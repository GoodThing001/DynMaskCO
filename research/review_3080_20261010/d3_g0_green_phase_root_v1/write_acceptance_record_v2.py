#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Emit the v2 acceptance record with THREE SEPARATE machine-readable fields.

Why this exists (user review of 51896c3, 2026-10-14):
`G0_SAFETY_DELTA.json` carries a single boolean `criterion_2_historical_suites: true`.  That is
ambiguous for an independent reviewer: it can be read as "all historical tests passed", whereas
the reality is "A1-A4 is 20/3 and the three failures are registered safety-contract updates".

The user's instruction was explicit: do NOT rewrite the old matrix (frozen record), and record
the three propositions SEPARATELY in a NEW acceptance report:
    historical_suites_all_pass          = false
    superseded_assertions_accounted_for = true
    updated_safety_contract_accepted    = true

So this script DERIVES each field from the measured numbers in `G0_SAFETY_DELTA.json` instead of
hard-coding them -- a hand-written `false` would be just as unverifiable as the hand-written
`true` it replaces.  It refuses to emit anything if the numbers and the registered
superseded-assertion list disagree.

Read-only w.r.t. everything except its own output file.
Run: python research/review_3080_20261010/d3_g0_green_phase_root_v1/write_acceptance_record_v2.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'G0_SAFETY_DELTA.json'
OUT = ROOT / 'ACCEPTANCE_RECORD_batch1_v2.json'


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = json.loads(OLD.read_text(encoding='utf-8'))
    suites = old['historical_suites']
    superseded = old['superseded_assertions']

    # ---- derive the three propositions from measurements, do not assert them ----------------
    live_failed_total = sum((s['live']['failed'] or 0) for s in suites.values())
    live_passed_total = sum((s['live']['passed'] or 0) for s in suites.values())
    prefix_failed_total = sum((s['prefix']['failed'] or 0) for s in suites.values())

    historical_suites_all_pass = (live_failed_total == 0)

    live_failed_cases_raw = sorted(c for s in suites.values() for c in s['live']['failed_cases'])
    # the suite's human output appends '   <- <detail>' to a failing case name; the registered
    # superseded-assertion list carries the bare name.  Normalise ONLY for the comparison and
    # keep the raw strings in the record, so nothing is hidden by the normalisation.
    live_failed_cases = sorted(c.split('   <- ')[0].strip() for c in live_failed_cases_raw)
    registered_cases = sorted(a['case'] for a in superseded)
    superseded_accounted = (live_failed_cases == registered_cases and len(registered_cases) > 0)

    # `accepted` is the USER's act, not the assistant's claim -> carried with provenance.
    accepted = old.get('historical_suites', {}).get('test_d3_assembly_invariants', {}) \
                  .get('live', {}).get('failed') == len(superseded)

    # fail closed: a v2 record that contradicts its own source must not be written
    problems = []
    if historical_suites_all_pass:
        problems.append('derived historical_suites_all_pass is True, expected False '
                        '(A1-A4 has %d failures)' % live_failed_total)
    if not superseded_accounted:
        problems.append('failed cases %r != registered superseded assertions %r'
                        % (live_failed_cases, registered_cases))
    if not accepted:
        problems.append('the superseded-assertion count does not match the live failure count')
    if problems:
        for p in problems:
            print('REFUSING:', p)
        return 1

    rec = {
        'record': 'D3_G0_G0-Safety_batch1_acceptance_v2',
        'date': '2026-10-14',
        'base_commit': old['base_commit'],
        'purpose': ('separate three propositions that the v1 single boolean conflated; the v1 '
                    'file is retained BYTE-INTACT as the frozen record and is NOT corrected in '
                    'place (user instruction: 保留旧矩阵原件，不追改历史记录)'),
        'supersedes_for_machine_reading': {
            'file': 'G0_SAFETY_DELTA.json',
            'field': 'acceptance.criterion_2_historical_suites',
            'v1_value': old['acceptance']['criterion_2_historical_suites'],
            'why_superseded': ('one boolean cannot distinguish "historical tests all passed" from '
                               '"historical tests have 3 registered, accounted-for failures"'),
            'v1_file_sha256_at_this_record': sha256_of(OLD),
            'v1_file_modified_in_place': False,
        },

        # ---- the three separate propositions (user-specified) --------------------------------
        'historical_suites_all_pass': False,
        'superseded_assertions_accounted_for': True,
        'updated_safety_contract_accepted': True,
        'updated_safety_contract_accepted_by': ('user, review of 51896c3 on 2026-10-14: "我认可三条'
                                                '失败属于已经解释的安全契约更新"'),

        # ---- the measurements the three fields are derived from ------------------------------
        'derivation': {
            'live_failed_total': live_failed_total,
            'live_passed_total': live_passed_total,
            'prefix_failed_total': prefix_failed_total,
            'live_failed_case_names': live_failed_cases,
            'live_failed_case_names_raw': live_failed_cases_raw,
            'registered_superseded_case_names': registered_cases,
            'sets_equal': superseded_accounted,
        },
        'per_suite': {name: {'prefix': '%s/%s' % (s['prefix']['passed'],
                                                  (s['prefix']['passed'] or 0)
                                                  + (s['prefix']['failed'] or 0)),
                             'live': '%s/%s' % (s['live']['passed'],
                                                (s['live']['passed'] or 0)
                                                + (s['live']['failed'] or 0)),
                             'delta_failed_cases': s['live']['failed_cases']}
                      for name, s in suites.items()},
        'unchanged_assertions': old['unchanged'],
        'superseded_assertions': superseded,
        'residuals_still_open': old['residuals_still_open'],
        'red_phase_harness_defect': old['red_phase_harness_defect'],
        'still_not_established': [
            'D3 G0 is NOT passed; only batch-1 local safety fixes have engineering acceptance '
            'evidence',
            'G1/G2 not started',
            'no attributable F-D learning gain',
            'I3 formal negative result, old D2 not_qualified, stable_online_gain=False unchanged',
        ],
    }
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8', newline='\n')
    print('historical_suites_all_pass          =', rec['historical_suites_all_pass'])
    print('superseded_assertions_accounted_for =', rec['superseded_assertions_accounted_for'])
    print('updated_safety_contract_accepted    =', rec['updated_safety_contract_accepted'],
          '(by %s)' % 'user')
    print('live=%d/%d  prefix_failed=%d  superseded=%d' % (
        live_passed_total, live_passed_total + live_failed_total, prefix_failed_total,
        len(registered_cases)))
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
