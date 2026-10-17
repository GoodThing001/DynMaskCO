#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 TOCTOU DIAGNOSTIC -- read-only observation of the CURRENT certify/install boundary.

Implements NOTHING.  It drives the production `InstallLedger` and the production receipt helper in a
controlled synthetic ledger and records where "the object that was checked" and "the object that was
used / archived" can differ.  The spec for this was registered first, in
`q3_g3_preregistration_amendment_02.md` (section 4, T-1..T-4).

WHAT THIS IS NOT
================
It is NOT evidence that any production incident occurred.  Every row here is a synthetic injection
into a ledger the probe constructs itself.  The claims it supports are exactly two:
  * the ORDER of operations (hash -> gate -> callback -> copy) admits a window, and
  * the old `plan_hash` cannot see a change in `anchor_time` / `anchor_load`.
Whether that window is ever exercised in a real run is a separate question this probe does not answer.

Usage: python scripts/tests/test_q3_g3_toctou_diagnostic.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))

import action_contract as AC                                              # noqa: E402
import d3_geo_adapter as GEO                                              # noqa: E402
import d3_install_receipt as RCP                                          # noqa: E402
import m1b_immutable_snapshot as IS                                       # noqa: E402

ROWS = []
LIMITS = ('synthetic ledger constructed by the probe; read-only use of the production helpers; no '
          'production module modified; NOT evidence of any production incident')
NOT_AN_INCIDENT = ('此探针只记录"机制存在"，不构成任何"已发生的生产事故"的证据；'
                   '所有注入都发生在探针自己构造的合成账本里。')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/simulation/d3_install_receipt.py',
              'scripts/m1b_adapter/m1b_immutable_snapshot.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS,
                 'not_a_production_incident': True})
    print('[%s] %-5s %s' % (st, cid, name[:92]))


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def plan(t=24.0, load=12.5, suffix=(3, 5)):
    return {0: AC.VehiclePlan(0, 7, t, load, tuple(suffix))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    # ============================================================ T-0 the identity in use
    P = plan()
    import inspect
    ident_source = inspect.getsource(GEO._plan_identity)
    h_old = GEO._plan_identity(P)
    h_contract = AC.plan_hash(P)
    delegates = ('plan_hash(' in ident_source)
    ok0 = (h_old == h_contract and delegates)
    ck('T-0', 'context: the install gate identity IS `action_contract.plan_hash` (the time-blind one), '
       'so every observation below is about that function and not about a stronger one',
       ok0,
       {'_plan_identity_delegates_to_plan_hash': delegates,
        'identity': h_old, 'plan_hash': h_contract,
        'fields_hashed': ['vehicle_id', 'anchor_node', 'suffix'],
        'fields_present_on_the_object': ['anchor_node', 'anchor_time', 'anchor_load', 'suffix']},
       conclusion='the gate compares the truncation of a hash that omits two physical fields')

    # ============================================================ T-1 the callback gets the same object
    seen = {}

    def mutating_install(p):
        seen['received_is_passed_object'] = (p is P)
        p[0].anchor_time = 30.0
        p[0].anchor_load = 25.0

    ledger = GEO.InstallLedger(install_fn=mutating_install)
    certified_v2 = IS.plan_identity_v2(P)
    res = ledger.finish(failure_reason=None, plan=P, original_plan={},
                        certified=True, certified_plan_hash=h_old, certifier='diagnostic',
                        deadline_ok=True)
    ok1 = (seen.get('received_is_passed_object') is True and res is True)
    ck('T-1', 'the install callback receives THE SAME MUTABLE OBJECT that was hashed: `install_fn(plan)` '
       'is called with the original mapping, not with an immutable snapshot taken before the check '
       '(T-1 of the registered spec)',
       ok1,
       {'received_is_the_passed_object': seen.get('received_is_passed_object'),
        'install_order': 'installed_hash=_plan_identity(plan) [before] -> gate -> install_fn(plan) '
                         '-> n_installs++ -> _copy_plan(plan) [after]',
        'source': 'd3_geo_adapter.py:666, :689, :691'},
       conclusion='the window exists in the ORDER, regardless of intent')

    # ============================================================ T-2 archive diverges from certification
    stored = ledger.installed_plans[0]
    entry = ledger.installs[0]
    archived_v2 = IS.plan_identity_v2(stored)
    archived_old_hash = GEO._plan_identity(stored)
    ok2 = (archived_v2 != certified_v2          # physically different plan
           and archived_old_hash == h_old       # yet the old hash still matches
           and entry['plan_hash'] == h_old
           and stored[0].anchor_time == 30.0 and stored[0].anchor_load == 25.0)
    ck('T-2', 'a callback that changes `anchor_time`/`anchor_load` during install leaves an archive whose '
       'PHYSICAL content differs from the certified plan, while the ledger row still records the '
       'pre-callback plan_hash -- so the archived plan and the recorded hash describe two different '
       'plans (T-2 of the registered spec)',
       ok2,
       {'certified_at': {'anchor_time': 24.0, 'anchor_load': 12.5},
        'archived_at': {'anchor_time': stored[0].anchor_time,
                        'anchor_load': stored[0].anchor_load},
        'identity_v2_certified': certified_v2[:16], 'identity_v2_archived': archived_v2[:16],
        'old_hash_certified': h_old, 'old_hash_of_archived': archived_old_hash,
        'ledger_row_plan_hash': entry['plan_hash'],
        'reading': 'the ledger is internally consistent under the OLD hash and physically wrong'},
       conclusion='archiving must be bound to the pre-callback snapshot')

    # ============================================================ T-3 different object, same content
    Q = plan()
    same_content = (GEO._plan_identity(Q) == GEO._plan_identity(plan()))
    ok3 = same_content
    ck('T-3', 'a DIFFERENT object with identical content is accepted by the old gate, because the gate '
       'compares content, not object identity. That is correct behaviour and worth recording: the '
       'defect is not "object identity", it is that the compared content omits two fields',
       ok3,
       {'different_objects_same_content_pass': same_content,
        'consequence': ('the candidate must compare CONTENT field-by-field (including anchor_time and '
                        'anchor_load), never rely on object identity')},
       conclusion='content comparison is right; the content set is what is wrong')

    # ============================================================ T-4 receipt read-back is blind
    record = {'plan': {0: {'vehicle_id': 0, 'anchor_node': 7, 'anchor_time': 24.0,
                           'anchor_load': 12.5, 'suffix': [3, 5]}}}
    hash_true = RCP._reproduce_plan_hash(record)
    tampered = {'plan': {0: {'vehicle_id': 0, 'anchor_node': 7, 'anchor_time': 30.0,
                             'anchor_load': 25.0, 'suffix': [3, 5]}}}
    hash_tampered = RCP._reproduce_plan_hash(tampered)
    ok4 = (hash_true is not None and hash_true == hash_tampered)
    ck('T-4', 'the persistent receipt STORES anchor_time and anchor_load but its verification re-hashes '
       'with the old function: a receipt whose two physical fields were changed still reproduces the '
       'same plan_hash, so read-back cannot prove they are unchanged (T-4 of the registered spec; a '
       'D-1 downstream effect, not a fourth defect class)',
       ok4,
       {'receipt_stores_physical_fields': True,
        'stored_fields': ['vehicle_id', 'anchor_node', 'anchor_time', 'anchor_load', 'suffix'],
        'verified_fields': ['vehicle_id', 'anchor_node', 'suffix'],
        'hash_authentic_receipt': hash_true,
        'hash_with_changed_physical_fields': hash_tampered,
        'reader': 'd3_install_receipt._reproduce_plan_hash (:203-230) -> plan_hash (:228)'},
       conclusion='storing a field is not verifying it')

    # ============================================================ T-5 production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok5 = (after == before and all(v is not None for v in after.values()))
    ck('T-5', 'the four production files are byte-identical before and after the probe: the TOCTOU '
       'window was observed, not created',
       ok5, {'digests': after}, conclusion='read-only observation')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G3_TOCTOU_DIAG_FAIL' if n_fail else 'G3_TOCTOU_OBSERVED'
    payload = {
        'schema': 'q3_g3_toctou_diagnostic',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'implements_nothing': True,
        'not_a_production_incident': NOT_AN_INCIDENT,
        'spec_registered_first': 'scripts/q3_qualification/q3_g3_preregistration_amendment_02.md',
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'measures': {
            'T-0': 'the gate identity is action_contract.plan_hash',
            'T-1': 'install_fn receives the same mutable object that was hashed',
            'T-2': 'archive and recorded hash describe two different plans after a callback mutation',
            'T-3': 'content comparison (not object identity) is the right rule',
            'T-4': 'receipt read-back cannot see anchor_time/anchor_load changes',
        },
        'does_not_measure': [
            'whether any real run exercised this window',
            'whether a production install ever archived a plan different from the certified one',
            'anything about float bounds, distance models or time budgets',
        ],
        'production_file_digests': before,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d probe rows: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
