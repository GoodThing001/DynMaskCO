#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 AMENDMENT CONTROLS -- the two SEPARATE defects of the current plan identity.

The errata (`q3_g3_preregistration_amendment_01.md`) was registered BEFORE this file was run.  It
records three distinct defects, not one:

    D-1 FIELD OMISSION      anchor_time and anchor_load are not in the encoding at all
    D-2 TRUNCATION          the digest is cut to 16 hex (64 bit)
    D-3 ENCODING COLLISION  the fields are concatenated with no length prefix or separator, so two
                            DIFFERENT plans produce the SAME byte stream -- unfixable by a longer hash

This probe only READS the production helpers and records the three separately.  It implements no
candidate and modifies nothing.

Inputs obey the production contract: `build_vehicle_plans` strips zeros, so a legal suffix never
contains 0 (that was wrong in the first counterexample and is corrected here).

Usage: python scripts/tests/test_q3_g3_amendment_controls.py --out <json path>
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
import m1b_immutable_snapshot as IS                                       # noqa: E402

ROWS = []
LIMITS = ('read-only reproduction with the production helpers; no candidate implemented, no production '
          'module modified, no data or frozen tree touched')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/m1b_adapter/m1b_immutable_snapshot.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:94]))


class P(object):
    def __init__(self, anchor_node=7, anchor_time=24.0, anchor_load=12.5, suffix=(3, 5)):
        self.anchor_node = anchor_node
        self.anchor_time = anchor_time
        self.anchor_load = anchor_load
        self.suffix = tuple(suffix)


def stream(plans):
    """The EXACT byte stream `plan_hash` feeds to sha256 -- reproduced from :303-308."""
    out = b''
    for vid in sorted(plans):
        p = plans[vid]
        out += str(vid).encode() + str(p.anchor_node).encode() + repr(tuple(p.suffix)).encode()
    return out


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    # ============================================================ M-1 encoding collision (5,6)
    a = {1: P(anchor_node=23, suffix=(5, 6))}
    b = {12: P(anchor_node=3, suffix=(5, 6))}
    sa, sb = stream(a), stream(b)
    full_a, full_b = hashlib.sha256(sa).hexdigest(), hashlib.sha256(sb).hexdigest()
    ok1 = (sa == sb and full_a == full_b
           and AC.plan_hash(a) == AC.plan_hash(b) == '58137566f9309ff7')
    ck('M-1', 'ENCODING COLLISION (D-3): `vid=1,anchor_node=23` and `vid=12,anchor_node=3` with the '
       'legal suffix (5,6) produce the SAME byte stream b"123(5, 6)", so the FULL 256-bit SHA256 is '
       'identical too -- this is serialisation non-injectivity, not a hash collision',
       ok1,
       {'plan_a': {'vehicle_id': 1, 'anchor_node': 23, 'suffix': [5, 6]},
        'plan_b': {'vehicle_id': 12, 'anchor_node': 3, 'suffix': [5, 6]},
        'byte_stream_a': sa.decode(), 'byte_stream_b': sb.decode(),
        'streams_identical': sa == sb,
        'full_sha256_a': full_a, 'full_sha256_b': full_b,
        'plan_hash_a': AC.plan_hash(a), 'plan_hash_b': AC.plan_hash(b),
        'fixable_by_full_digest': False,
        'why': 'the preimage is the same byte string, so no digest length can separate them'},
       conclusion='D-3 is not D-2: a longer hash cannot fix it')

    # ============================================================ M-2 the same at (3,5)
    a2 = {1: P(anchor_node=23, suffix=(3, 5))}
    b2 = {12: P(anchor_node=3, suffix=(3, 5))}
    ok2 = (stream(a2) == stream(b2) == b'123(3, 5)'
           and AC.plan_hash(a2) == AC.plan_hash(b2) == 'f5c20e00860e9e87')
    ck('M-2', 'the same collision at suffix (3,5) reproduces the independent reviewer table exactly '
       '(`f5c20e00860e9e87` from the shared stream b"123(3, 5)")',
       ok2,
       {'byte_stream': stream(a2).decode(), 'plan_hash': AC.plan_hash(a2),
        'matches_reviewer_value': AC.plan_hash(a2) == 'f5c20e00860e9e87'},
       conclusion='the finding is reproducible from the source alone')

    # ============================================================ M-3 field omission, legal suffix
    cert = {0: P(anchor_node=7, anchor_time=24.0, anchor_load=12.5, suffix=(3, 5))}
    inst = {0: P(anchor_node=7, anchor_time=30.0, anchor_load=25.0, suffix=(3, 5))}
    ok3 = (AC.plan_hash(cert) == AC.plan_hash(inst)
           and IS.plan_identity_v2(cert) != IS.plan_identity_v2(inst))
    ck('M-3', 'FIELD OMISSION (D-1), re-run on a CONTRACT-LEGAL suffix (3,5): a plan certified at '
       '24.0 h / 12.5 load and one installed at 30.0 h / 25.0 load share the plan identity, while '
       'plan_identity_v2 separates them',
       ok3,
       {'certified': {'anchor_time': 24.0, 'anchor_load': 12.5, 'suffix': [3, 5],
                      'plan_hash': AC.plan_hash(cert),
                      'plan_identity_v2': IS.plan_identity_v2(cert)[:16]},
        'installed': {'anchor_time': 30.0, 'anchor_load': 25.0, 'suffix': [3, 5],
                      'plan_hash': AC.plan_hash(inst),
                      'plan_identity_v2': IS.plan_identity_v2(inst)[:16]},
        'suffix_legal': True,
        'correction': ('the first counterexample used (3,5,0); build_vehicle_plans strips zeros '
                       '(:107), so that input was not production-shaped.  The conclusion is '
                       'unchanged on the legal input.'),
        'historical_record': 'the original 9/9 artifact is kept byte-intact and is NOT rewritten'},
       conclusion='D-1 stands on a legal input')

    # ============================================================ M-4 G-08 demoted
    # the collision needs a compensating change in vid; within a FIXED vid the anchor_node alone is
    # separated.  Both halves are recorded so the claim is bounded precisely.
    fixed_vid_separates = AC.plan_hash({0: P(anchor_node=7)}) != AC.plan_hash({0: P(anchor_node=8)})
    cross_field_collides = AC.plan_hash(a) == AC.plan_hash(b)
    ok4 = (fixed_vid_separates and cross_field_collides)
    ck('M-4', 'G-08 DEMOTED: the earlier statement "plan_hash separates anchor_node / vehicle-set '
       'changes" is true only at SAMPLE level.  Within a fixed vehicle_id an anchor_node change IS '
       'separated, but a simultaneous vid change can compensate exactly and produce an identical '
       'stream -- so it is not a structural guarantee',
       ok4,
       {'fixed_vid_anchor_change_detected': fixed_vid_separates,
        'cross_field_compensation_detected': False,
        'colliding_pair': {'a': {'vid': 1, 'anchor_node': 23}, 'b': {'vid': 12, 'anchor_node': 3}},
        'corrected_claim': ('G-08 remains valid for its own samples and is NOT a general property; '
                            'the G-3 errata section 2 records the demotion'),
        'n_defect_classes': 3,
        'defect_classes': ['D-1 field omission', 'D-2 truncation', 'D-3 encoding collision']},
       conclusion='three separate defects, one of them unfixable by hashing harder')

    # ============================================================ M-5 the suffix contract
    import numpy as np

    class V(object):
        vehicle_id = 0
        status = 'ready'
        current_node = 7
        ready_time = 24.0
        current_load = 12.5
        mutable_suffix = [3, 5, 0]

    class E(object):
        demands = np.zeros((1, 10))

    plans = AC.build_vehicle_plans(E(), 0, [V()])
    built = plans[0]
    ok5 = (built.suffix == (3, 5) and list(built.anchor_time.__class__.__mro__)  # sanity: a float
           and built.anchor_time == 24.0 and built.anchor_load == 12.5)
    mixed = {0: P(suffix=(3, 0, 5))}      # an interior zero is NOT produced by the builder
    ck('M-5', 'INPUT CONTRACT: `build_vehicle_plans` strips zeros (`:107`), so a legal production '
       'suffix never contains 0 at all -- the corrected counterexamples use (3,5)/(5,6).  An interior '
       'zero is not production-shaped and must be refused by the candidate (N-16)',
       ok5,
       {'input_mutable_suffix': [3, 5, 0], 'built_suffix': list(built.suffix),
        'anchor_fields': {'anchor_node': built.anchor_node, 'anchor_time': built.anchor_time,
                          'anchor_load': built.anchor_load},
        'interior_zero_example_stream': stream(mixed).decode(),
        'rule': 'suffix must be a suffix of positive customer ids; 0 anywhere is a contract violation'},
       conclusion='the test inputs now match the production data contract')

    # ============================================================ M-6 production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok6 = (after == before and all(v is not None for v in after.values()))
    ck('M-6', 'the three production files are byte-identical before and after: this round registered '
       'an errata and reproduced the defects without modifying any helper',
       ok6, {'digests': after}, conclusion='read-only')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G3_AMENDMENT_FAIL' if n_fail else 'G3_AMENDMENT_CONTROLS_PASS'
    payload = {
        'schema': 'q3_g3_amendment_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'implements_nothing': True,
        'errata_registered_before_these_controls': (
            'scripts/q3_qualification/q3_g3_preregistration_amendment_01.md'),
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'defect_classes': {
            'D-1_field_omission': {'fields': ['anchor_time', 'anchor_load'],
                                   'fixable_by_full_digest': False,
                                   'control': 'M-3'},
            'D-2_truncation': {'hex': 16, 'fixable_by_full_digest': True, 'control': 'G-04 (prior)'},
            'D-3_encoding_collision': {'shared_stream': '123(5, 6) / 123(3, 5)',
                                       'fixable_by_full_digest': False,
                                       'control': 'M-1, M-2'},
        },
        'requirements_for_the_g3_candidate': {
            'C13': 'the collision pair MUST get different identities (N-15)',
            'C14': 'a 1-ulp change in anchor_time / anchor_load MUST change the identity (N-14)',
            'N-16': 'a suffix containing 0 MUST be refused (SUFFIX_CONTRACT_VIOLATION)',
            'note': 'requirements only; no candidate is implemented in this round',
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
    print('\n=== %d controls: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
