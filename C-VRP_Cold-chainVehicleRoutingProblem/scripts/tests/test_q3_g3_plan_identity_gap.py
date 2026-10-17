#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 DEFECT REPRODUCTION -- read-only.  Does the CURRENT plan identity bind the whole plan?

This probe implements NOTHING.  It calls the two production helpers exactly as the codebase does and
records where they stop being injective:

    action_contract.plan_hash()                 vehicle_id + anchor_node + suffix, SHA256 truncated to 16 hex
    m1b_immutable_snapshot.plan_identity_v2()   all four fields, but anchor_time / anchor_load rounded to 12

The question is not "is there a hash"; it is "can a plan that was CERTIFIED differ from the plan that is
INSTALLED while both still carry the same identity".  Every row below is a witness for or against that.

Nothing is modified: `action_contract.py`, `d3_geo_adapter.py`, `m1b_immutable_snapshot.py`, the frozen
trees and the data are only read.

Usage: python scripts/tests/test_q3_g3_plan_identity_gap.py --out <json path>
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
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))

import action_contract as AC                                              # noqa: E402
import m1b_immutable_snapshot as IS                                       # noqa: E402

ROWS = []
LIMITS = ('read-only reproduction with the production helpers; no candidate identity is implemented, '
          'no production module is modified, no data or frozen tree is touched')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/m1b_adapter/m1b_immutable_snapshot.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:94]))


class P(object):
    """The attribute surface both helpers read: anchor_node / anchor_time / anchor_load / suffix."""

    def __init__(self, anchor_node=7, anchor_time=24.0, anchor_load=12.5, suffix=(3, 5, 0)):
        self.anchor_node = anchor_node
        self.anchor_time = anchor_time
        self.anchor_load = anchor_load
        self.suffix = tuple(suffix)


def one(p, vid=0):
    return {vid: p}


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    base = P()
    h0 = AC.plan_hash(one(base))
    i0 = IS.plan_identity_v2(one(base))

    # ============================================================ G-01 anchor_time is not in plan_hash
    t_ulp = math.nextafter(24.0, 25.0)
    a_time_ulp = AC.plan_hash(one(P(anchor_time=t_ulp)))
    a_time_big = AC.plan_hash(one(P(anchor_time=24.0 + 6.0)))     # six hours later
    ok1 = (a_time_ulp == h0 and a_time_big == h0)
    ck('G-01', 'plan_hash ignores anchor_time completely: a plan departing at the same instant, at one '
       'ulp later, and SIX HOURS later all carry the same plan identity',
       ok1,
       {'baseline_hash': h0, 'plus_1_ulp': a_time_ulp, 'plus_6_hours': a_time_big,
        'source': 'action_contract.py:301-309 -- h.update is called for vid, anchor_node, suffix only',
        'fields_present_on_the_object': ['anchor_node', 'anchor_time', 'anchor_load', 'suffix']},
       conclusion='the identity is blind to the departure time')

    # ============================================================ G-02 anchor_load is not in plan_hash
    a_load_small = AC.plan_hash(one(P(anchor_load=12.5 + 1e-9)))
    a_load_big = AC.plan_hash(one(P(anchor_load=12.5 * 2)))       # double the load
    ok2 = (a_load_small == h0 and a_load_big == h0)
    ck('G-02', 'plan_hash ignores anchor_load completely: doubling the carried load leaves the plan '
       'identity unchanged',
       ok2,
       {'baseline_hash': h0, 'load_plus_1e-9': a_load_small, 'load_doubled': a_load_big},
       conclusion='the identity is blind to what the vehicle carries')

    # ============================================================ G-03 the certified/installed gap
    cert = P(anchor_time=24.0, anchor_load=12.5, suffix=(3, 5, 0))
    inst = P(anchor_time=30.0, anchor_load=25.0, suffix=(3, 5, 0))   # a materially different plan
    ok3 = (AC.plan_hash(one(cert)) == AC.plan_hash(one(inst))
           and IS.plan_identity_v2(one(cert)) != IS.plan_identity_v2(one(inst)))
    ck('G-03', 'PRIMARY COUNTEREXAMPLE: a plan certified at 24.0 h with load 12.5 and a plan installed '
       'at 30.0 h with load 25.0 share the SAME plan_hash, while the stricter helper separates them -- '
       'so "certified A, installed B" is invisible to any gate that trusts plan_hash',
       ok3,
       {'certified': {'anchor_time': cert.anchor_time, 'anchor_load': cert.anchor_load,
                      'plan_hash': AC.plan_hash(one(cert)),
                      'plan_identity_v2': IS.plan_identity_v2(one(cert))[:16]},
        'installed': {'anchor_time': inst.anchor_time, 'anchor_load': inst.anchor_load,
                      'plan_hash': AC.plan_hash(one(inst)),
                      'plan_identity_v2': IS.plan_identity_v2(one(inst))[:16]},
        'same_plan_hash': True, 'different_plan_identity_v2': True},
       conclusion='the two identities disagree on the exact pair a real gate cares about')

    # ============================================================ G-04 truncation
    full = hashlib.sha256(b'probe').hexdigest()
    ok4 = (len(h0) == 16 and len(full) == 64)
    ck('G-04', 'the plan identity is a SHA256 TRUNCATED to 16 hex characters (64 bits): the collision '
       'budget of a 64-bit truncated digest is ~2^32 plans, and a truncated digest cannot be promoted '
       'to a full identity later without changing every recorded value',
       ok4,
       {'plan_hash_length': len(h0), 'full_sha256_length': len(full),
        'truncation_bits': 64, 'birthday_budget': 2 ** 32,
        'note': 'truncation is a property of the encoding, not of any certificate built on it'},
       conclusion='a 64-bit prefix is not a full identity')

    # ============================================================ G-05 plan_identity_v2 rounding window
    for_12 = 0.5 * 10 ** -12
    absorbed = IS.plan_identity_v2(one(P(anchor_time=24.0 + 1e-13)))
    caught = IS.plan_identity_v2(one(P(anchor_time=24.0 + 1e-6)))
    load_absorbed = IS.plan_identity_v2(one(P(anchor_load=12.5 + 1e-13)))
    ok5 = (absorbed == i0 and load_absorbed == i0 and caught != i0)
    ck('G-05', 'plan_identity_v2 is stricter than plan_hash but still LOSSY: `round(x, 12)` absorbs a '
       '1e-13 change in anchor_time and in anchor_load, so a sub-1e-12 drift is not a difference; a '
       '1e-6 change IS caught',
       ok5,
       {'baseline': i0[:16], 'time_plus_1e-13': absorbed[:16], 'load_plus_1e-13': load_absorbed[:16],
        'time_plus_1e-6': caught[:16],
        'rounding_radius': for_12,
        'source': 'm1b_immutable_snapshot.py:47-53 -- round(anchor_time, 12), round(anchor_load, 12)',
        'ulp_at_24': math.ulp(24.0),
        'ulp_vs_rounding_radius': math.ulp(24.0) / for_12},
       conclusion='the G-1 requirement (lossless, bit-exact) is not met by either helper')

    # ============================================================ G-06 the install path trusts it
    src = (EXT / 'scripts/simulation/d3_geo_adapter.py').read_text(encoding='utf-8')
    uses = [{'line': i, 'text': ln.strip()}
            for i, ln in enumerate(src.splitlines(), 1)
            if 'plan_hash' in ln and 'def ' not in ln]
    ok6 = any('certified_plan_hash' in u['text'] or 'installed_hash' in u['text'] for u in uses)
    ck('G-06', 'the certification and install transaction is wired to this truncated, time-blind '
       'identity: `d3_geo_adapter.py` uses plan_hash for the certified plan hash and for the installed '
       'plan hash, so the equality check between them cannot see a time or load change',
       ok6,
       {'call_sites': uses[:8], 'n_call_sites': len(uses),
        'reading': ('the ledger compares certified_plan_hash with installed_hash; both are computed by '
                    'plan_hash, so the comparison is exactly as strong as plan_hash is')},
       conclusion='the defect reaches the install boundary, which is why G-3 exists')

    # ============================================================ G-07 fields present but unbound
    p = P(anchor_time=30.0, anchor_load=25.0)
    ok7 = (hasattr(p, 'anchor_time') and hasattr(p, 'anchor_load')
           and AC.plan_hash(one(p)) == h0)
    ck('G-07', 'the omission is not "the field does not exist": `VehiclePlan` carries anchor_time and '
       'anchor_load (action_contract.py:67-73) and they are simply not hashed, so the fix is a binding '
       'question rather than a data-availability question',
       ok7,
       {'vehicle_plan_fields': ['anchor_node', 'anchor_time', 'anchor_load', 'suffix'],
        'hashed_by_plan_hash': ['vehicle_id', 'anchor_node', 'suffix'],
        'hashed_by_plan_identity_v2': ['vehicle_id', 'anchor_node', 'anchor_time(rounded)',
                                       'anchor_load(rounded)', 'suffix'],
        'omitted_by_plan_hash': ['anchor_time', 'anchor_load']},
       conclusion='the data is there; the identity is what is missing')

    # ============================================================ G-08 the probes are disjoint
    # a suffix change IS caught by plan_hash; a time-or-load change is not.  Stating what the current
    # identity does see keeps the finding narrow and falsifiable.
    seen_suffix = AC.plan_hash(one(P(suffix=(3, 6, 0)))) != h0
    seen_node = AC.plan_hash(one(P(anchor_node=8))) != h0
    seen_vid = AC.plan_hash({0: base, 1: P()}) != h0
    ok8 = (seen_suffix and seen_node and seen_vid
           and AC.plan_hash(one(P(anchor_time=30.0))) == h0)
    ck('G-08', 'the finding is narrow: plan_hash DOES separate a suffix change, an anchor_node change '
       'and a vehicle-set change; it does not separate a time or load change.  G-3 therefore needs to '
       'bind the two missing fields, not to discard the existing ones',
       ok8,
       {'suffix_change_detected': seen_suffix, 'anchor_node_change_detected': seen_node,
        'vehicle_set_change_detected': seen_vid,
        'anchor_time_change_detected': False, 'anchor_load_change_detected': False},
       conclusion='bind what is missing; keep what works')

    # ============================================================ G-09 production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok9 = (after == before and all(v is not None for v in after.values()))
    ck('G-09', 'the three production files are byte-identical before and after the probe: this round '
       'reproduced the defect without modifying any helper',
       ok9, {'digests': after}, conclusion='read-only reproduction')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G3_DEFECT_REPRO_FAIL' if n_fail else 'G3_DEFECT_REPRODUCED'
    payload = {
        'schema': 'q3_g3_plan_identity_gap',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'implements_nothing': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'defect': {
            'plan_hash_binds': ['vehicle_id', 'anchor_node', 'suffix'],
            'plan_hash_omits': ['anchor_time', 'anchor_load'],
            'plan_hash_truncation_hex': 16,
            'plan_identity_v2_rounding_decimals': 12,
            'install_gate_depends_on': 'd3_geo_adapter plan_hash equality',
        },
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
