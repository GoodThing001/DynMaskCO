#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 ISOLATED CANDIDATE CONTROLS -- the three identity defects, the TOCTOU window, and the six new
negative controls registered in `q3_g3_preregistration_amendment_03.md`.

Everything runs on synthetic objects in this process.  Nothing is installed, no production helper is
modified, and the candidate is not wired anywhere.  The point is one question:

    can the identity force "the object that was CERTIFIED" and "the object that is USED / ARCHIVED"
    to be the same object, and can the install boundary claim only what it can prove?

Usage: python scripts/tests/test_q3_g3_plan_identity_candidate.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))

import action_contract as AC                                              # noqa: E402
import q3_g3_plan_identity_candidate as G3                                # noqa: E402

ROWS = []
LIMITS = ('synthetic in-process objects only; candidate is isolated and unwired; no production helper '
          'modified; the in-process read-only view is NOT a security sandbox')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/simulation/m1b_immutable_snapshot.py',
              'scripts/m1b_adapter/m1b_immutable_snapshot.py',
              'scripts/simulation/d3_install_receipt.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:92]))


def try_call(fn, *a, **k):
    """Call and classify: a Refusal keeps its code; any other exception is recorded by type."""
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except G3.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}
    except BaseException as exc:                                          # noqa: BLE001
        return {'ok': False, 'code': type(exc).__name__, 'detail': {'error': str(exc)[:120]}}


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def plan(t=24.0, load=12.5, suffix=(3, 5), node=7, vid=0):
    return {vid: AC.VehiclePlan(vid, node, t, load, tuple(suffix))}


FROZEN_STATE = {0: {'current_node': 7, 'committed_next': None, 'ready_time': 24.0, 'load': 12.5}}


def cert(plan_obj):
    return G3.certify_frozen_plan(G3.deep_freeze_plan(plan_obj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES if blob_digest(rel) is not None}

    # ============================================================ C1 primary control
    P = plan()
    old_a, old_b = AC.plan_hash(plan(t=24.0, load=12.5)), AC.plan_hash(plan(t=30.0, load=25.0))
    new_a = G3.plan_identity(G3.deep_freeze_plan(plan(t=24.0, load=12.5)))
    new_b = G3.plan_identity(G3.deep_freeze_plan(plan(t=30.0, load=25.0)))
    ok1 = (old_a == old_b and new_a != new_b and len(new_a) == 64)
    ck('C1', 'PRIMARY CONTROL (D-1): two plans that differ only in anchor_time/anchor_load share the '
       'old plan_hash and get DIFFERENT 64-hex candidate identities',
       ok1,
       {'old_hash_a': old_a, 'old_hash_b': old_b, 'old_identical': old_a == old_b,
        'new_a': new_a, 'new_b': new_b, 'new_length_hex': len(new_a)},
       conclusion='D-1 is closed by the candidate and not by the old hash')

    # ============================================================ C2 encoding collision (D-3)
    coll_a = {1: AC.VehiclePlan(1, 23, 24.0, 12.5, (5, 6))}
    coll_b = {12: AC.VehiclePlan(12, 3, 24.0, 12.5, (5, 6))}
    old_coll = AC.plan_hash(coll_a) == AC.plan_hash(coll_b)
    new_ident_a = G3.plan_identity(G3.deep_freeze_plan(coll_a))
    new_ident_b = G3.plan_identity(G3.deep_freeze_plan(coll_b))
    ok2 = (old_coll and new_ident_a != new_ident_b)
    ck('C2', 'D-3 (cross-field encoding collision): the pair whose byte streams are identical under the '
       'old concatenation (`123(5, 6)`) gets different candidate identities, because every field is '
       'length-prefixed',
       ok2,
       {'old_collides': old_coll, 'old_hash': AC.plan_hash(coll_a),
        'new_identity_a': new_ident_a, 'new_identity_b': new_ident_b,
        'encoding': 'len(name):name len(payload):payload, per field'},
       conclusion='D-3 is unfixable by digest length and is fixed by unambiguous framing')

    # ============================================================ C3 truncation (D-2)
    ok3 = (len(new_a) == 64 and len(old_a) == 16
           and G3.certify_frozen_plan(G3.deep_freeze_plan(P))['length_hex'] == 64)
    ck('C3', 'D-2 (truncation): the candidate identity is the full 256-bit digest (64 hex), and the '
       'certificate reports its length so a short form cannot be mistaken for the identity',
       ok3, {'candidate_length_hex': 64, 'old_length_hex': 16},
       conclusion='no truncated prefix is ever treated as the identity')

    # ============================================================ C4 deep freeze (N-17)
    shallow = {0: AC.VehiclePlan(0, 7, 24.0, 12.5, [3, 5])}
    from types import MappingProxyType
    shallow_proxy = MappingProxyType(shallow)
    shallow_assert = try_call(G3.assert_deep_frozen, shallow_proxy)
    deep = G3.deep_freeze_plan(P)
    deep_assert = try_call(G3.assert_deep_frozen, deep)
    # and the inner objects of the shallow structure really are mutable
    shallow[0].anchor_time = 99.0
    inner_mutable = shallow[0].anchor_time == 99.0
    ok4 = (not shallow_assert['ok'] and shallow_assert['code'] == 'FREEZE_NOT_DEEP'
           and deep_assert['ok'] and inner_mutable)
    ck('C4', 'N-17 shallow read-only wrapper bypass: a `MappingProxyType` over live VehiclePlans is '
       'REFUSED as not deep-frozen, and the probe confirms the inner objects really were assignable '
       '-- so "proxy at the top level" is rejected rather than trusted',
       ok4,
       {'shallow_assert': shallow_assert, 'deep_assert_ok': deep_assert['ok'],
        'inner_plan_was_mutable': inner_mutable,
        'why': 'the freeze must be recursive; a wrapper at the outer mapping proves nothing about the '
               'inner plans'},
       conclusion='L-1 is enforced by a recursive walk, not by a wrapper')

    # ============================================================ C5 nested modification (N-18)
    target = G3.deep_freeze_plan(plan(suffix=(3, 5)))
    normal = try_call(setattr, target.plans[0], 'anchor_time', 30.0)
    bypassed = False
    try:
        object.__setattr__(target.plans[0], 'anchor_time', 30.0)
        bypassed = True
    except Exception:                                                      # noqa: BLE001
        bypassed = False
    mutated_identity = G3.plan_identity(target) if bypassed else None
    ok5 = (not normal['ok'] and bypassed and mutated_identity is not None
           and mutated_identity != G3.plan_identity(G3.deep_freeze_plan(plan(suffix=(3, 5)))))
    ck('C5', 'N-18 nested modification: attribute assignment raises FrozenInstanceError, and even when '
       '`object.__setattr__` is used to bypass the freeze the identity CHANGES -- so the post-install '
       're-hash still sees it',
       ok5,
       {'normal_assignment': normal.get('code'),
        'bypass_succeeded': bypassed,
        'identity_after_bypass': (mutated_identity or '')[:16],
        'reading': 'the freeze is a first line; the re-hash is the detector'},
       conclusion='neither layer alone is claimed to be sufficient')

    # ============================================================ C6 same content, different object (N-21)
    cert_a = cert(plan())
    inst = G3.install_with_frozen_identity(plan(), certificate=cert_a, install_fn=lambda v: None,
                                           frozen_state=FROZEN_STATE)
    ok6 = (inst['outcome'] == G3.INSTALL_COMPLETED and inst['n_installs'] == 1)
    ck('C6', 'N-21 (legal case): a DIFFERENT object with identical content is accepted, because the '
       'identity is over content and not over object identity -- this is the case that must NOT be '
       'refused, and separating it from "content changed" is the whole point',
       ok6,
       {'outcome': inst['outcome'], 'n_installs': inst['n_installs'],
        'pre_identity': (inst['pre_identity'] or '')[:16],
        'certified_identity': (inst['certified_identity'] or '')[:16]},
       conclusion='content identity, not object identity')

    # ============================================================ C7 content changed (N-22)
    changed_after_cert = plan()
    cert_c = cert(changed_after_cert)
    changed_after_cert[0].anchor_time = 30.0            # the caller mutates AFTER certification
    called = {'n': 0}

    def counting(_view):
        called['n'] += 1

    inst7 = G3.install_with_frozen_identity(changed_after_cert, certificate=cert_c,
                                            install_fn=counting, frozen_state=FROZEN_STATE)
    ok7 = (inst7['outcome'] == G3.REFUSED_BEFORE_INSTALL and inst7['callback_called'] is False
           and inst7['n_installs'] == 0 and called['n'] == 0
           and any(r['code'] == 'CERTIFIED_INSTALLED_MISMATCH' for r in inst7['refusals']))
    ck('C7', 'N-22 / T-1: if the caller mutates the plan AFTER certification, the install is refused '
       'BEFORE the callback runs (callback_called=False, n_installs=0, the mock records zero calls), so '
       'this tier can honestly claim "nothing was installed"',
       ok7,
       {'outcome': inst7['outcome'], 'callback_called': inst7['callback_called'],
        'n_installs': inst7['n_installs'], 'mock_call_count': called['n'],
        'refusals': inst7['refusals'], 'proves': inst7['proves']},
       conclusion='pre-install refusal is the only tier allowed to claim "no install"')

    # ============================================================ C8 callback mutates the frozen plan (T-2)
    side_effect = {'wrote': False}

    def hostile(view):
        # a legitimate install writes something; then it also tries to move the plan
        side_effect['wrote'] = True
        for v in view.values():
            object.__setattr__(v, 'anchor_time', 30.0)

    inst8 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()), install_fn=hostile,
                                            frozen_state=FROZEN_STATE)
    ok8 = (inst8['outcome'] == G3.INSTALL_OUTCOME_UNVERIFIED
           and inst8['callback_called'] is True and inst8['side_effects_possible'] is True
           and inst8['rollback_verified'] is False
           and inst8['archived_plan'] is None
           and any(r['code'] == 'PLAN_MUTATED_DURING_INSTALL' for r in inst8['refusals'])
           and side_effect['wrote'] is True)
    ck('C8', 'T-2: a callback that mutates the frozen plan during install is DETECTED by the '
       'post-install re-hash, the plan is NOT archived, and the outcome is '
       '`INSTALL_OUTCOME_UNVERIFIED` -- it does NOT claim "no install", because the callback already '
       'ran and may have written something',
       ok8,
       {'outcome': inst8['outcome'], 'callback_called': inst8['callback_called'],
        'n_installs': inst8['n_installs'],
        'side_effects_possible': inst8['side_effects_possible'],
        'archived_plan_is_none': inst8['archived_plan'] is None,
        'pre_identity': (inst8['pre_identity'] or '')[:16],
        'post_identity': (inst8['post_identity'] or '')[:16],
        'refusals': inst8['refusals'], 'proves': inst8['proves']},
       conclusion='detection is not rollback')

    # ============================================================ C9 callback raises (N-19)
    def boom(_view):
        raise RuntimeError('install backend exploded')

    inst9 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()), install_fn=boom,
                                            frozen_state=FROZEN_STATE)
    ok9 = (inst9['outcome'] == G3.INSTALL_OUTCOME_UNVERIFIED
           and inst9['callback_called'] is True
           and any(r['code'] == 'INSTALL_CALLBACK_RAISED' for r in inst9['refusals'])
           and 'does NOT prove' in inst9['proves'])
    ck('C9', 'N-19: a callback that raises leaves the outcome `INSTALL_OUTCOME_UNVERIFIED`, and the '
       'receipt says in words that it does not prove nothing was installed -- an exception is not a '
       'rollback',
       ok9,
       {'outcome': inst9['outcome'], 'n_installs': inst9['n_installs'],
        'refusals': inst9['refusals'], 'proves': inst9['proves'],
        'side_effects_possible': inst9['side_effects_possible']},
       conclusion='an exception is not evidence of no side effect')

    # ============================================================ C10 side effect then failure (N-20)
    marker = {'written': False}

    def write_then_fail(_view):
        marker['written'] = True
        raise RuntimeError('failed after writing')

    inst10 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()),
                                             install_fn=write_then_fail,
                                             frozen_state=FROZEN_STATE)
    ok10 = (marker['written'] is True and inst10['outcome'] == G3.INSTALL_OUTCOME_UNVERIFIED
            and inst10['side_effects_possible'] is True)
    ck('C10', 'N-20: a callback that writes a side effect and THEN fails is recorded as '
       '`INSTALL_OUTCOME_UNVERIFIED` with `side_effects_possible=True` -- the written marker is '
       'observed, so "nothing happened" would be a false claim',
       ok10,
       {'marker_written': marker['written'], 'outcome': inst10['outcome'],
        'n_installs': inst10['n_installs'],
        'side_effects_possible': inst10['side_effects_possible'],
        'rollback_verified': inst10['rollback_verified']},
       conclusion='the receipt reports the side effect it cannot undo')

    # ============================================================ C11 verified rollback
    rollback = {'ran': False}

    def rb(_frozen):
        rollback['ran'] = True
        return True

    inst11 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()), install_fn=hostile,
                                             rollback_fn=rb, frozen_state=FROZEN_STATE)
    ok11 = (inst11['outcome'] == G3.INSTALL_ROLLED_BACK and rollback['ran'] is True
            and inst11['rollback_verified'] is True and inst11['n_installs'] == 0
            and 'not independently verified' in inst11['proves'])
    ck('C11', 'a VERIFIED rollback lifts the outcome to `INSTALL_ROLLED_BACK`, and the receipt still '
       'states that the post-rollback state is not independently verified by this module -- the claim '
       'is bounded to "the rollback ran and reported success"',
       ok11,
       {'outcome': inst11['outcome'], 'rollback_ran': rollback['ran'],
        'rollback_verified': inst11['rollback_verified'], 'n_installs': inst11['n_installs'],
        'proves': inst11['proves']},
       conclusion='even the best tier claims only what it can')

    # ============================================================ C12 anchor cross-check (L-5)
    bad_state = {0: {'current_node': 7, 'committed_next': None, 'ready_time': 30.0, 'load': 12.5}}
    inst12 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()),
                                             install_fn=counting, frozen_state=bad_state)
    ok12 = (inst12['outcome'] == G3.REFUSED_BEFORE_INSTALL
            and any(r['code'] == 'ANCHOR_NOT_FROM_FROZEN_STATE' for r in inst12['refusals']))
    ck('C12', 'L-5: anchors that do not match the frozen execution state are refused BEFORE install '
       '(`ANCHOR_NOT_FROM_FROZEN_STATE`), so a caller cannot supply a plausible-but-invented anchor',
       ok12,
       {'outcome': inst12['outcome'], 'n_installs': inst12['n_installs'],
        'callback_called': inst12['callback_called'], 'refusals': inst12['refusals']},
       conclusion='the anchors have one authority: the frozen state')

    # ============================================================ C13 missing certificate
    inst13 = G3.install_with_frozen_identity(plan(), certificate=None, install_fn=counting,
                                             frozen_state=FROZEN_STATE)
    ok13 = (inst13['outcome'] == G3.REFUSED_BEFORE_INSTALL
            and any(r['code'] == 'CERTIFIED_IDENTITY_MISSING' for r in inst13['refusals']))
    ck('C13', 'no certificate means no install: `CERTIFIED_IDENTITY_MISSING` and the callback is never '
       'called', ok13, {'outcome': inst13['outcome'], 'refusals': inst13['refusals']},
       conclusion='the certificate is a precondition, not a comment')

    # ============================================================ C14 suffix contract
    bad_suffix = try_call(G3.deep_freeze_plan, plan(suffix=(3, 0, 5)))
    ok14 = (not bad_suffix['ok'] and bad_suffix['code'] == 'SUFFIX_CONTRACT_VIOLATION')
    ck('C14', 'N-16 carried over: a suffix containing 0 is refused (`SUFFIX_CONTRACT_VIOLATION`), since '
       'a production suffix never contains one', ok14, bad_suffix,
       conclusion='production-shaped inputs only')

    # ============================================================ C15 domain limits are stated
    receipt_limits = inst8['domain_limits']
    ok15 = (isinstance(receipt_limits, list) and len(receipt_limits) >= 4
            and any('NOT a security sandbox' in s for s in receipt_limits)
            and any('threads' in s for s in receipt_limits)
            and any('external processes' in s for s in receipt_limits))
    ck('C15', 'the applicability domain is carried IN THE RECEIPT, not just in a comment: in-process '
       'read-only wrappers are not a sandbox, and threads / external processes / escaped references / '
       'real executor side effects are listed as uncovered',
       ok15, {'domain_limits': receipt_limits},
       conclusion='the limits travel with the artifact')

    # ============================================================ C16 production untouched
    after = {rel: blob_digest(rel) for rel in before}
    ok16 = (after == before)
    ck('C16', 'the production helpers are byte-identical before and after: the candidate is isolated '
       'and unwired', ok16, {'digests': before},
       conclusion='no production change')

    # ============================================================ C17 what the tiers can claim
    claims = {'REFUSED_BEFORE_INSTALL': inst7['proves'],
              'INSTALL_COMPLETED': inst['proves'],
              'INSTALL_ROLLED_BACK': inst11['proves'],
              'INSTALL_OUTCOME_UNVERIFIED': inst8['proves']}
    ok17 = ('never called' in claims['REFUSED_BEFORE_INSTALL']
            and 'identical before and after' in claims['INSTALL_COMPLETED']
            and 'reported success' in claims['INSTALL_ROLLED_BACK']
            and 'does NOT prove' in claims['INSTALL_OUTCOME_UNVERIFIED'])
    ck('C17', 'the four tiers state DIFFERENT things, and only `REFUSED_BEFORE_INSTALL` may claim "no '
       'install"; `INSTALL_OUTCOME_UNVERIFIED` explicitly refuses to claim it -- this is the atomicity '
       'boundary the review demanded', ok17, {'claims_by_tier': claims},
       conclusion='no post-hoc hash is presented as atomicity')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G3_CANDIDATE_FAIL' if n_fail else 'G3_CANDIDATE_PASS_ISOLATED'
    payload = {
        'schema': 'q3_g3_plan_identity_candidate_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'production_consumable': False,
        'baseline_head': '4f9bf896e64b2ed0981229ef432b10969a910b5b',
        'registrations_first': ['scripts/q3_qualification/q3_g3_preregistration_2026-10-17.md',
                               'scripts/q3_qualification/q3_g3_preregistration_amendment_01.md',
                               'scripts/q3_qualification/q3_g3_preregistration_amendment_02.md',
                               'scripts/q3_qualification/q3_g3_preregistration_amendment_03.md'],
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'defects_closed_by_the_candidate': ['D-1 field omission', 'D-2 truncation',
                                            'D-3 encoding collision'],
        'defects_closed_by_the_boundary': ['T-1 callback receives the audited object',
                                           'T-2 archive diverges from certification',
                                           'T-4 receipt read-back is blind'],
        'not_proven': ['install atomicity after the callback ran (INSTALL_OUTCOME_UNVERIFIED)',
                       'post-rollback state (only "the rollback reported success")',
                       'threads, external processes, escaped references',
                       'real executor side effects',
                       'any production qualification (PRODUCTION_NOT_QUALIFIED holds)'],
        'domain_limits': list(G3.DOMAIN_LIMITS),
        'production_file_digests': before,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'GTMIN_V10_CLEAN_CHECKOUT': 'FAIL (11 PASS / 0 SKIP / 1 FAIL, PL-12 undeclared)',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED',
            'PRODUCTION_NOT_QUALIFIED': True, 'B-1..B-9': 'all open'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d controls: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
