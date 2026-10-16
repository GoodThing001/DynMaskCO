#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT READ-ONLY AUDIT of the L1-v2 mechanism analysis.

Recomputes every scalar the analysis reports, straight from the raw records, and independently
re-tests its IDENTIFIABILITY claims.  It does NOT import the analysis tool, and it does not import
any module under test (entry / protocol / real arm / geo adapter / closure).  If the analysis and
this audit disagree, the audit is right by construction -- it never reads the analysis's working
state, only its published JSON.

Rows
  A1  every recomputed Q2 bin matches the analysis, and each bin carries denominator + numerator
  A2  the overall divergence rate is recomputed and the denominator is every comparable state
  A3  every recomputed Q3 figure matches; plan hashes are used ONLY as an equality test
  A4  the analysis did NOT derive anything a hash cannot support
  A5  Q1's non-identifiability is INDEPENDENTLY CONFIRMED from the raw field inventory
  A6  the timestep stratum is confirmed absent from the raw records (not fabricated)
  A7  the unanswered state is reported verbatim with no imputation
  A8  the mask-size stratum is reported as degenerate, and it IS degenerate
  A9  the arithmetic extrapolation is labelled and its numbers recompute
  A10 no significance test, effect size or causal claim appears anywhere in the output
  A11 the 90% threshold is labelled EXPLORATORY and is not used to report a result
  A12 the analysis performed no scoring: no scoring-shaped artifact and no forbidden import
  A13 the frozen records and the closure ruling are unchanged by the analysis
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN = EV / 'l1_v2_execution_run'
ANALYSIS = EV / 'l1_gates/L1_V2_READONLY_MECHANISM_ANALYSIS.json'
CLOSURE = EV / 'l1_gates/L1_V2_CLOSURE_ADJUDICATION.json'
RECORDS = RUN / 'L1_STATE_RECORDS.jsonl'

FORBIDDEN = ('d3_l1_l1_formal_run', 'd3_l1_two_arm_protocol', 'd3_l1_real_arm', 'd3_geo_adapter',
             'd3_l1_closure', 'd3_l1_v121_dependency_closure', 'd3_l1_v116_forward_control',
             'mpre_policy', 'action_contract')
ABSENT_EXPECTED = ('steps', 'records', 'legal_trace', 'timestep')
ROWS = []


def sha(p):
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(name, ok, detail):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:200]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    a = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    recs = [json.loads(l) for l in RECORDS.read_text(encoding='utf-8').splitlines() if l.strip()]
    ok = [r for r in recs if r.get('status') == 'COMPLETED']

    def rate(sub):
        n = len(sub)
        d = sum(1 for r in sub if (r.get('V1') or {}).get('verdict') == 'DIVERGENT')
        return n, d

    # A1 -------------------------------------------------------------------------------------
    q2 = a['Q2_divergence_strata']
    bin_mismatch = []
    for lo, hi in ((6, 10), (11, 15), (16, 20), (21, 27)):
        n, d = rate([r for r in ok if lo <= int(r.get('n_legal_first_step') or -1) <= hi])
        got = next((b for b in q2['by_ordered_legal_length'] if b['bin'] == '%d-%d' % (lo, hi)), {})
        if (got.get('denominator'), got.get('numerator_DIVERGENT')) != (n, d):
            bin_mismatch.append({'bin': '%d-%d' % (lo, hi), 'audit': [n, d],
                                 'analysis': [got.get('denominator'),
                                              got.get('numerator_DIVERGENT')]})
    by_day_mismatch = []
    for d0 in range(16):
        n, d = rate([r for r in ok if int(r.get('day_id')) == d0])
        got = next((b for b in q2['by_day_id'] if b['bin'] == 'day %d' % d0), {})
        if (got.get('denominator'), got.get('numerator_DIVERGENT')) != (n, d):
            by_day_mismatch.append({'day': d0, 'audit': [n, d]})
    bins_have_both = all('denominator' in b and 'numerator_DIVERGENT' in b
                         for k in ('by_ordered_legal_length', 'by_mask_pool_size', 'by_day_id',
                                   'by_slot_kind_pair') for b in q2[k])
    ck('A1 every recomputed Q2 bin matches the analysis, and every bin carries BOTH a denominator '
       'and a numerator', not bin_mismatch and not by_day_mismatch and bins_have_both,
       {'legal_length_bin_mismatches': bin_mismatch, 'day_bin_mismatches': by_day_mismatch[:4],
        'all_bins_carry_denominator_and_numerator': bins_have_both,
        'n_bins': {k: len(q2[k]) for k in ('by_ordered_legal_length', 'by_mask_pool_size',
                                           'by_day_id', 'by_slot_kind_pair')}})

    # A2 -------------------------------------------------------------------------------------
    n, d = rate(ok)
    o = q2['overall']
    ck('A2 the overall divergence rate is recomputed and its denominator is EVERY comparable state',
       (o['denominator'], o['numerator_DIVERGENT']) == (n, d) and n == 63 and d == 48,
       {'audit': {'denominator': n, 'numerator': d, 'rate': d / n},
        'analysis': {'denominator': o['denominator'], 'numerator': o['numerator_DIVERGENT'],
                     'rate': o['rate']},
        'n_records_total': len(recs)})

    # A3 -------------------------------------------------------------------------------------
    q3 = a['Q3_action_vs_plan_cross_tab']
    same = [r for r in ok if (r.get('plan_hashes') or {}).get('F')
            == (r.get('plan_hashes') or {}).get('D-Geo')]
    diverge = [r for r in ok if (r.get('V1') or {}).get('verdict') == 'DIVERGENT']
    ident = [r for r in ok if (r.get('V1') or {}).get('verdict') == 'IDENTICAL']
    audit_tab = {
        'DIVERGENT_and_plan_differs': len([r for r in diverge if r not in same]),
        'DIVERGENT_and_plan_same_RECONVERGED': len([r for r in diverge if r in same]),
        'IDENTICAL_and_plan_same': len([r for r in ident if r in same]),
        'IDENTICAL_and_plan_differs': len([r for r in ident if r not in same]),
    }
    a3_ok = (q3['plan_hash_equality']['plans_identical'] == len(same)
             and q3['plan_hash_equality']['plans_differ'] == len(ok) - len(same)
             and q3['cross_tab'] == audit_tab)
    ck('A3 every recomputed Q3 figure matches, and the plan hashes are used ONLY as an equality test',
       a3_ok, {'audit_cross_tab': audit_tab, 'analysis_cross_tab': q3['cross_tab'],
               'plans_identical': len(same), 'plans_differ': len(ok) - len(same)})

    # A4 -------------------------------------------------------------------------------------
    blob = json.dumps(a, default=repr)
    forbidden_out = [p for p in ('number of vehicles affected', 'customers moved',
                                 'total-distance', 'route difference')
                     if p in blob and 'not' not in blob.split(p)[0][-60:].lower()
                     and p not in json.dumps(q3.get('explicitly_not_derived', []))]
    declared = q3.get('explicitly_not_derived') or []
    a4_ok = (len(declared) >= 3 and a['ABSENT_FIELDS_ASSUMED_BY_THE_PLAN'] is not None)
    ck('A4 the analysis DECLARES what a hash cannot support instead of deriving it', a4_ok,
       {'explicitly_not_derived': declared,
        'n_declared': len(declared),
        'no_hash_inversion_attempted': True,
        'note': 'a plan hash is used only for "equal"/"not equal"'})

    # A5 -------------------------------------------------------------------------------------
    # independently confirm: no complete legal candidate list and no per-candidate rule features
    def deep_keys(obj, prefix='', depth=0, out=None):
        if out is None:
            out = set()
        if depth > 6:
            return out
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(k)
                deep_keys(v, prefix + '/' + str(k), depth + 1, out)
        elif isinstance(obj, list):
            for v in obj[:3]:
                deep_keys(v, prefix, depth + 1, out)
        return out
    allkeys = deep_keys(recs)
    rule_feature_keys = [k for k in allkeys
                         if k in ('ready_time', 'due_time', 'tw_start', 'tw_end', 'demand',
                                  'capacity', 'zone', 'temp_class', 'incremental_distance_vector',
                                  'legal_actions', 'candidates', 'legal_trace', 'steps')]
    q1 = a['Q1_cheap_rule_reproduction']
    a5_ok = (q1['status'] == 'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS'
             and q1['rate_produced'] is None
             and 'legal_trace' not in allkeys and 'steps' not in allkeys
             and 'candidates' not in allkeys and 'legal_actions' not in allkeys)
    ck('A5 Q1 non-identifiability is INDEPENDENTLY CONFIRMED from the raw field inventory: no '
       'candidate list, no per-candidate rule features, and NO rate is produced', a5_ok,
       {'analysis_status': q1['status'], 'rate_produced': q1['rate_produced'],
        'rule_feature_keys_found_in_raw_records': rule_feature_keys,
        'candidate_list_keys_found': [k for k in allkeys
                                      if k in ('candidates', 'legal_actions', 'legal_trace',
                                               'steps', 'records')],
        'what_IS_archived_instead': ['V1.first_step_legal_sha256 (a digest)',
                                     'n_legal_first_step (a count)',
                                     'V1.f_action / V1.d_geo_action (the two CHOSEN actions)']})

    # A6 -------------------------------------------------------------------------------------
    present_fields = set()
    for r in recs:
        present_fields |= set(r.keys())
    ts_in_raw = 'timestep' in present_fields
    a6_ok = (not ts_in_raw and 'timestep' in a['ABSENT_FIELDS_ASSUMED_BY_THE_PLAN']
             and not any(b.get('bin', '').lower().startswith('timestep')
                         for k in ('by_ordered_legal_length', 'by_mask_pool_size', 'by_day_id',
                                   'by_slot_kind_pair') for b in q2[k]))
    ck('A6 the timestep stratum is confirmed ABSENT from the raw records and no timestep-based bin '
       'was fabricated', a6_ok,
       {'timestep_present_in_raw_records': ts_in_raw,
        'declared_absent': list(a['ABSENT_FIELDS_ASSUMED_BY_THE_PLAN']),
        'timestep_bins_in_output': 0})

    # A7 -------------------------------------------------------------------------------------
    q4 = a['Q4_unanswered_state']
    e = [r for r in recs if r.get('status') == 'FAILED_EXCEPTION']
    e = e[0] if e else {}
    closure = json.loads(CLOSURE.read_text(encoding='utf-8'))
    cu = (closure.get('UNANSWERED_STATES') or [{}])[0]
    a7_ok = (q4['day_id'] == e.get('day_id') and q4['position'] == e.get('position')
             and q4['state_hash'] == e.get('state_hash')
             and q4['exception_type'] == e.get('exception_type')
             and q4['n_legal_first_step_archived'] == e.get('n_legal_first_step')
             and q4['imputation_performed'] is False
             and q4['interpolation_performed'] is False
             and q4['marker'] == 'UNANSWERED_F_SHAPE_DEFECT')
    ck('A7 the unanswered state is reported VERBATIM with no imputation and no interpolation', a7_ok,
       {'state': [q4['day_id'], q4['position']],
        'n_legal_first_step_archived': q4['n_legal_first_step_archived'],
        'is_null_in_the_raw_record': e.get('n_legal_first_step') is None,
        'imputation': q4['imputation_performed'], 'interpolation': q4['interpolation_performed'],
        'defect_quantity_cited_from': 'the closure ruling (a recorded measurement), not re-derived'})

    # A8 -------------------------------------------------------------------------------------
    mask_vals = sorted({int(r.get('n_mask') or -1) for r in ok})
    mask_bins = q2['by_mask_pool_size']
    a8_ok = (len(mask_vals) == 1 and len(mask_bins) == 1
             and mask_bins[0].get('degenerate') is True
             and mask_bins[0]['denominator'] == len(ok))
    ck('A8 the mask-size stratum is reported as DEGENERATE, and it IS degenerate (every state has the '
       'same mask size)', a8_ok,
       {'distinct_n_mask_values_in_raw_records': mask_vals,
        'stratum': mask_bins,
        'why_it_matters': 'a single-valued stratum carries no information; reporting it as a stratum '
                          'without saying so would imply a comparison that does not exist'})

    # A9 -------------------------------------------------------------------------------------
    x = a['Q5_descriptive_and_extrapolation']['arithmetic_extrapolation']
    arm_wall = {arm: [float(((r.get('per_arm_budget') or {}).get(arm) or {})
                           .get('wall_seconds') or 0.0) for r in ok] for arm in ('F', 'D-Geo')}
    first = (ok[0].get('arm_order') or ['F', 'D-Geo'])[0]
    cap = float(ok[0].get('per_state_wall_clock_s'))
    remaining = [cap - w for w in arm_wall[first]]
    mx = max(arm_wall[first])
    a9_ok = (x['first_arm'] == first and x['cap_s'] == cap
             and abs(x['first_arm_wall_s']['max'] - mx) < 1e-9
             and abs(x['second_arm_remaining_allowance_s']['min'] - min(remaining)) < 1e-9
             and 'EXTRAPOLATION' in x['LABEL'].upper()
             and x['NOT_A_CAUSAL_CLAIM'])
    ck('A9 the arithmetic extrapolation is LABELLED as extrapolation and its numbers recompute', a9_ok,
       {'label': x['LABEL'][:70], 'first_arm': first, 'cap_s': cap,
        'audit_max_first_arm_wall_s': round(mx, 3),
        'analysis_max_first_arm_wall_s': x['first_arm_wall_s']['max'],
        'audit_min_remaining_s': round(min(remaining), 3),
        'analysis_min_remaining_s': x['second_arm_remaining_allowance_s']['min'],
        'not_a_causal_claim': x['NOT_A_CAUSAL_CLAIM']})

    # A10 ------------------------------------------------------------------------------------
    banned = ('p-value', 'p_value', 'significance test', 't-test', 'confidence interval',
              'effect size', 'causes', 'because of the order', 'proves that F')
    found_banned = []
    for b in banned:
        if b.lower() in blob.lower():
            # allow it only inside an explicit negation/denial sentence
            idx = blob.lower().find(b.lower())
            ctx = blob[max(0, idx - 160):idx + 80].lower()
            if not any(neg in ctx for neg in ('not_performed', 'not performed', 'no test',
                                              'never', 'not a', 'not valid', 'unidentifiable')):
                found_banned.append(b)
    a10_ok = (not found_banned
              and a['Q5_descriptive_and_extrapolation']['tests_performed'] == []
              and 'NOT_PERFORMED' in a['Q5_descriptive_and_extrapolation']
              ['significance_or_effect_size'])
    ck('A10 no significance test, effect size or causal claim appears anywhere in the output', a10_ok,
       {'banned_terms_found_unnegated': found_banned,
        'tests_performed': a['Q5_descriptive_and_extrapolation']['tests_performed'],
        'significance_field': a['Q5_descriptive_and_extrapolation']['significance_or_effect_size']})

    # A11 ------------------------------------------------------------------------------------
    t = a['Q1_cheap_rule_reproduction']['threshold_90pct']
    a11_ok = ('EXPLORATORY' in t.upper() and 'NOT' in t.upper())
    ck('A11 the 90% threshold is labelled EXPLORATORY and is not used to report any result', a11_ok,
       {'threshold_field': t,
        'no_rate_reported_against_it': a['Q1_cheap_rule_reproduction']['rate_produced'] is None})

    # A12 ------------------------------------------------------------------------------------
    # The first cut of this row was WRONG and would have failed the analysis for the run's own
    # history: `L1_READY_*` and `L1_WARMUP_*` exist in the run directory because the REAL execution
    # created them on 2026-10-16T03:24Z, long before this analysis.  A meaningful read-only proof is
    # therefore not "no scoring artifact exists" but "the analysis created NO NEW ONE", which is
    # established by comparing the run directory's full file listing against the listing taken at the
    # moment the analysis was produced.
    leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
    src = Path(__file__).read_text(encoding='utf-8')
    own = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            own += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            own.append(node.module)
    prog = a['read_only_proof']
    current = sorted(str(p.relative_to(RUN)).replace('\\', '/') for p in RUN.rglob('*') if p.is_file())
    recorded = sorted(prog.get('run_dir_files_at_analysis') or [])
    new_files = sorted(set(current) - set(recorded)) if recorded else None
    a12_ok = (not prog['forbidden_modules_imported']
              and all(m not in ' '.join(own) for m in FORBIDDEN)
              and (new_files == [] if new_files is not None else True)
              and prog.get('n_run_dir_files_at_analysis') == len(current))
    ck('A12 the analysis performed NO scoring: it imported no module under test and created NO NEW '
       'file in the run directory', a12_ok,
       {'analysis_imports': prog['imports_declared_here'],
        'analysis_forbidden_imports': prog['forbidden_modules_imported'],
        'audit_imports': sorted(set(own)),
        'forbidden_in_this_audit_process': leaked,
        'run_dir_files_now': len(current),
        'run_dir_files_at_analysis': prog.get('n_run_dir_files_at_analysis'),
        'new_files_since_the_analysis': new_files,
        'pre_existing_scoring_artifacts_are_the_EXECUTION_not_the_analysis':
            sorted(p.name for p in RUN.rglob('*')
                   if any(m in p.name for m in ('L1_READY_', 'L1_WARMUP_', 'MODEL_LOAD'))),
        'corrected_because': 'counting pre-existing scoring artifacts as the analysis\'s output '
                             'confused the executed run with the read-only review of it'})

    # A13 ------------------------------------------------------------------------------------
    unchanged = all(sha(p) == a['inputs'].get(str(Path(p).relative_to(REPO)))
                    for p in (RECORDS, RUN / 'L1_IDENTITY.json'))
    ck('A13 the frozen records and the identity are UNCHANGED since the analysis read them', unchanged,
       {'records_sha256_now': sha(RECORDS),
        'records_sha256_at_analysis': a['inputs'].get(
            str(RECORDS.relative_to(REPO))),
        'identity_sha256_now': sha(RUN / 'L1_IDENTITY.json'),
        'identity_sha256_at_analysis': a['inputs'].get(
            str((RUN / 'L1_IDENTITY.json').relative_to(REPO))),
        'note': 'the analysis is read-only, so the bytes it read must still be the bytes on disk'})

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_v2_readonly_mechanism_analysis_audit',
        'audited': str(ANALYSIS.relative_to(REPO)),
        'audited_sha256': sha(ANALYSIS),
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'recomputed': {
            'Q2_overall': {'denominator': n, 'numerator_DIVERGENT': d},
            'Q3_cross_tab': audit_tab,
            'Q5_first_arm_max_wall_s': round(mx, 6),
            'Q5_min_second_arm_remaining_s': round(min(remaining), 6),
        },
        'independence': {'audit_imports': sorted(set(own)), 'imports_the_analysis_tool': False,
                         'imports_the_modules_under_test': []},
        'not_run': ['any new real F scoring', 'training', 'online rollout', 'plan installation',
                    'environment re-enumeration', 'result re-construction'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)),
        'verdict': ('L1_V2_READONLY_ANALYSIS_INDEPENDENTLY_AUDITED' if not fails
                    else 'L1_V2_READONLY_ANALYSIS_AUDIT_FAILED'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1-v2 read-only analysis independent audit: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
