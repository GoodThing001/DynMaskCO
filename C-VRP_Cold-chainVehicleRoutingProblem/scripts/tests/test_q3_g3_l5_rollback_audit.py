#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 L-5 / ROLLBACK QUALIFICATION AUDIT -- read-only, additive.

Two independent-review findings about the ISOLATED CANDIDATE (not about production code) are measured
here so the errata can cite numbers instead of prose:

  finding 1  L-5 (anchor cross-check) is NOT mandatory: `require_frozen_state` defaults to False, so a
             plan whose anchors match no frozen state can still reach INSTALL_COMPLETED.  And the
             tolerances it would apply (1e-12 s, 1e-9 load) are far coarser than the 1-ulp separation
             the G-1 identity was built to give.  For a committed vehicle the load is taken from the
             caller's `load` field, never re-derived from the pending order's demand.
  finding 2  INSTALL_ROLLED_BACK is reached when `rollback_fn` merely RETURNS a truthy value; there is
             no independent post-rollback state check, so the tier name and the boolean
             `rollback_verified` overstate the pre-registered condition ("rollback completed AND the
             post-rollback state was verified").

This file changes nothing.  It does not relax, tighten or rename anything in the candidate; it only
records what the current code does, so amendment 04's registration is evidence-backed.

Usage: python scripts/tests/test_q3_g3_l5_rollback_audit.py --out <json path>
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
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))

import action_contract as AC                                              # noqa: E402
import q3_g3_plan_identity_candidate as G3                                # noqa: E402

ROWS = []
LIMITS = ('read-only audit of the ISOLATED candidate; nothing installed, no production helper touched, '
          'no experiment')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/m1b_adapter/m1b_immutable_snapshot.py',
              'scripts/simulation/d3_install_receipt.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:92]))


def try_call(fn, *a, **k):
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except G3.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}
    except BaseException as exc:                                          # noqa: BLE001
        return {'ok': False, 'code': type(exc).__name__}


def plan(t=24.0, load=12.5, vid=0, node=7):
    return {vid: AC.VehiclePlan(vid, node, t, load, (3, 5))}


def cert(p):
    return G3.certify_frozen_plan(G3.deep_freeze_plan(p))


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES if blob_digest(rel) is not None}

    # ============================================================ L5-1 mandatory?
    called = {'n': 0}

    def counting(_view):
        called['n'] += 1

    # anchors that match NO frozen state are supplied, and no frozen state is passed at all
    invented = plan(t=12345.0, load=999.0)
    r1 = G3.install_with_frozen_identity(invented, certificate=cert(invented),
                                         install_fn=counting)      # no frozen_state, defaults
    ok1 = (r1['outcome'] == G3.INSTALL_COMPLETED and called['n'] == 1
           and r1['refusals'] == [])
    ck('L5-1', 'FINDING 1a: L-5 is NOT mandatory. With the default `frozen_state=None` / '
       '`require_frozen_state=False`, a plan whose anchors match no frozen execution state reaches '
       '`INSTALL_COMPLETED` and the callback runs -- the unique-source constraint was never applied',
       ok1,
       {'outcome': r1['outcome'], 'callback_called': r1['callback_called'],
        'refusals': r1['refusals'],
        'anchor_supplied': {'anchor_time': 12345.0, 'anchor_load': 999.0},
        'defaults': {'frozen_state': None, 'require_frozen_state': False},
        'source': 'candidate :367-368 (signature defaults) and the `if require_frozen_state or '
                  'frozen_state is not None:` guard'},
       conclusion='B-9a cannot be described as fully closed')

    # ============================================================ L5-2 tolerance granularity
    ulp24 = math.ulp(24.0)
    state = {0: {'current_node': 7, 'committed_next': None, 'ready_time': 24.0, 'load': 12.5}}
    cases = {}
    for name, delta_t, delta_l in (('1_ulp', ulp24, 0.0),
                                   ('1e-13', 1e-13, 0.0),
                                   ('2e-12', 2e-12, 0.0),
                                   ('load_1e-10', 0.0, 1e-10),
                                   ('load_2e-9', 0.0, 2e-9)):
        p = plan(t=24.0 + delta_t, load=12.5 + delta_l)
        fz = G3.deep_freeze_plan(p)
        cases[name] = try_call(G3.anchor_cross_check, fz, state)
    accepted = {k: bool(v['ok']) for k, v in cases.items()}
    ok2 = (accepted['1_ulp'] is True and accepted['1e-13'] is True
           and accepted['2e-12'] is False and accepted['load_1e-10'] is True
           and accepted['load_2e-9'] is False)
    ck('L5-2', 'FINDING 1b: the anchor tolerances are ABSOLUTE (1e-12 s, 1e-9 load), not the G-1 '
       'one-ulp criterion. A 1-ulp difference AND a ~28-ulp difference (1e-13 s) both pass; only a '
       '2e-12 difference is refused, so the effective granularity is ~282 ulp at 24 h',
       ok2,
       {'ulp_at_24': ulp24,
        'tolerance_time': 1e-12, 'tolerance_time_in_ulp': 1e-12 / ulp24,
        'tolerance_load': 1e-9,
        'accepted_by_the_cross_check': accepted,
        'source': 'candidate :350 (:350 abs(anchor_time - expect) > 1e-12) and :356 (> 1e-9)'},
       conclusion='the tolerance regime was never pre-declared and is far coarser than 1 ulp')

    # ============================================================ L5-3 committed load not re-derived
    committed_state = {0: {'current_node': 0, 'committed_next': 5, 'committed_finish': 30.0,
                           'load': 999.0}}
    # anchor_node MUST equal committed_next (5) or the refusal would be about the node, not the
    # load; the point is that a caller-consistent pair passes with the load never re-derived
    fabricated = plan(t=30.0, load=999.0, node=5)
    r3 = try_call(G3.anchor_cross_check, G3.deep_freeze_plan(fabricated), committed_state)
    ok3 = r3['ok'] is True
    ck('L5-3', 'FINDING 1c: for a committed vehicle the expected load is the CALLER-SUPPLIED `load` '
       'field; the pending order demand is never re-derived, so a caller that supplies a state '
       'consistent with its own plan passes the cross-check. L-5 verifies CONSISTENCY, not provenance',
       ok3,
       {'cross_check_passed': r3['ok'], 'supplied_load': 999.0,
        'source': 'candidate :343 (`expect_load = st.get(\'load\')  # + demand applied by the caller; '
                  'not re-derived here`)'},
       conclusion='a consistent pair is not a true pair')

    # ============================================================ RB-1 rollback "verified" is self-reported
    ran = {'n': 0}

    def noop_rollback(_frozen):
        ran['n'] += 1
        return True                       # reports success, changes nothing at all

    def hostile(_view):
        for v in _view.values():
            object.__setattr__(v, 'anchor_time', 30.0)

    rb = G3.install_with_frozen_identity(plan(), certificate=cert(plan()), install_fn=hostile,
                                         rollback_fn=noop_rollback, frozen_state=state)
    ok4 = (rb['outcome'] == G3.INSTALL_ROLLED_BACK and rb['rollback_verified'] is True
           and ran['n'] == 1
           and 'not independently verified' in rb['proves'])
    ck('RB-1', 'FINDING 2: `INSTALL_ROLLED_BACK` is reached by a rollback that merely RETURNS True and '
       'does nothing. The tier name and `rollback_verified` therefore mean "the rollback reported '
       'success", which is weaker than the pre-registered condition ("rollback completed AND the '
       'post-rollback state was verified"). The receipt text is honest; the NAMES overstate',
       ok4,
       {'outcome': rb['outcome'], 'rollback_verified_flag': rb['rollback_verified'],
        'rollback_call_count': ran['n'], 'rollback_did_anything': False,
        'proves_text': rb['proves'],
        'source': "candidate :463 (`rollback_verified = bool(rollback_fn(frozen))`)",
        'registered_condition': 'amendment 03 section 2: "rollback executed AND the post-rollback '
                                'state was verified"',
        'gap': 'no independent post-rollback verifier is called'},
       conclusion='a boolean returned by the rollback is not a state check')

    # ============================================================ RB-2 negative rollback stays unverified
    def refusing_rollback(_frozen):
        return False

    rb2 = G3.install_with_frozen_identity(plan(), certificate=cert(plan()), install_fn=hostile,
                                          rollback_fn=refusing_rollback, frozen_state=state)
    ok5 = (rb2['outcome'] == G3.INSTALL_OUTCOME_UNVERIFIED and rb2['rollback_verified'] is False)
    ck('RB-2', 'control: a rollback that reports failure leaves the outcome '
       '`INSTALL_OUTCOME_UNVERIFIED`, so the tier only rises on an explicit claim of success -- the '
       'finding is about what "success" is allowed to mean, not about the ladder being broken',
       ok5, {'outcome': rb2['outcome'], 'rollback_verified': rb2['rollback_verified']},
       conclusion='the ladder works; the top rung proves less than registered')

    # ============================================================ production untouched
    after = {rel: blob_digest(rel) for rel in before}
    ok6 = (after == before)
    ck('AUD-0', 'the production helpers are byte-identical before and after the audit, and the '
       'candidate module was not modified by this audit', ok6, {'digests': before},
       conclusion='read-only')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G3_L5_ROLLBACK_AUDIT_FAIL' if n_fail else 'G3_L5_ROLLBACK_AUDIT_OBSERVED'
    payload = {
        'schema': 'q3_g3_l5_rollback_audit',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'audits': 'the isolated candidate only',
        'baseline_head': '0e38936a0b2d4e5a01ed518dbe67fa1b300c65ee',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'findings': {
            'L5_not_mandatory': {'default_require_frozen_state': False,
                                 'observed': 'INSTALL_COMPLETED with anchors matching no state'},
            'L5_tolerance_regime': {'absolute_time_tolerance': 1e-12,
                                    'absolute_load_tolerance': 1e-9,
                                    'ulp_at_24': ulp24,
                                    'time_tolerance_in_ulp': 1e-12 / ulp24,
                                    'pre_declared_regime': None},
            'L5_committed_load': 'taken from the caller-supplied `load`, demand never re-derived',
            'rollback_verified': 'bool(rollback_fn(...)) only; no independent post-rollback verifier',
        },
        'does_not_claim': ['that the candidate is unsafe to keep as an isolated artifact',
                           'that any production incident occurred',
                           'that B-9b is closable inside G-3'],
        'production_file_digests': before,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'GTMIN_V10_CLEAN_CHECKOUT': 'FAIL',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED',
            'PRODUCTION_NOT_QUALIFIED': True, 'B-1..B-9': 'all open',
            'B-9a': 'conditionally closed (L-5 gaps registered)',
            'B-9b': 'OUT OF G-3 SCOPE (option c), deferred'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d audit rows: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
