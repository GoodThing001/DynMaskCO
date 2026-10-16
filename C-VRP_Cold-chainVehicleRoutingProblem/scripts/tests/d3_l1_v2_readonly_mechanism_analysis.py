#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1-v2 READ-ONLY MECHANISM ANALYSIS (data-identifiability first).

READ-ONLY.  This tool reads ONLY the already-collected L1-v2 records and the frozen identity
manifest.  It imports NEITHER the L1 run entry, NOR the two-arm protocol, NOR the real arm, NOR the
geo adapter, NOR the closure; it instantiates no model, calls no scorer, re-enumerates no
environment and reconstructs no result.  It emits no `L1_READY` / `MODEL_LOAD` / `SCORES_*` artifact,
because it performs no scoring at all.

Its FIRST product is an identifiability audit, because the analysis plan assumed the archived records
carry `steps`, `records` and `legal_trace`, and THEY DO NOT.  Every question is therefore resolved
against the fields that actually exist:

  Q1  cheap-rule reproduction of F's first step  -> NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS
      (no complete ordered legal list per state, no per-candidate rule features; the record keeps only
       the legal list's SHA-256, the legal COUNT, and the two CHOSEN actions)
  Q2  divergence strata, denominator + numerator per bin   -> COMPUTABLE
  Q3  first-step action vs plan-hash cross tabulation      -> COMPUTABLE (descriptive only; nothing is
      inferred from a hash beyond "equal" / "not equal")
  Q4  the unanswered state, verifiable frozen inputs only  -> PARTIAL (whose values are archived)
  Q5  labelled descriptive counts + arithmetic projection  -> COMPUTABLE

Terminology is locked: a "cheap rule" hit rate is NOT computable here, so none is produced; anything
arithmetically extrapolated is labelled as such; a differing plan hash is an UNCERTIFIED difference
and never a quality statement.
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
PREREG = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
MANIFEST = EV / 'l1_gates/L1_V2_PREREGISTRATION_MANIFEST.json'

FORBIDDEN_IMPORTS = ('d3_l1_l1_formal_run', 'd3_l1_two_arm_protocol', 'd3_l1_real_arm',
                     'd3_geo_adapter', 'd3_l1_closure', 'd3_l1_v121_dependency_closure',
                     'd3_l1_v116_forward_control', 'mpre_policy', 'action_contract',
                     'd3_l1_v110_input_qualification')
# artifacts that only a REAL scoring run can leave behind: their absence is the proof of read-only
SCORING_ARTIFACT_MARKERS = ('L1_READY_', 'MODEL_LOAD', 'L1_WARMUP_', 'SCORES_')


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def field_inventory(records):
    """Every top-level field with its presence count and a shape summary -- no assumptions."""
    out = {}
    for r in records:
        for k, v in r.items():
            rec = out.setdefault(k, {'present': 0, 'types': Counter(), 'example': None})
            rec['present'] += 1
            rec['types'][type(v).__name__] += 1
            if rec['example'] is None and v is not None:
                rec['example'] = str(v)[:90]
    for k, rec in out.items():
        rec['types'] = dict(rec['types'])
    return out


def nested_field_inventory(records, key):
    """Presence + sub-keys of a nested dict field, across records."""
    subs = Counter()
    present = 0
    for r in records:
        v = r.get(key)
        if isinstance(v, dict):
            present += 1
            for kk in v:
                subs[kk] += 1
    return {'present': present, 'subkeys': dict(subs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()

    records = read_jsonl(RUN / 'L1_STATE_RECORDS.jsonl') if (RUN / 'L1_STATE_RECORDS.jsonl') else []
    ok = [r for r in records if r.get('status') == 'COMPLETED']
    exc = [r for r in records if r.get('status') == 'FAILED_EXCEPTION']
    frozen = json.loads((RUN / 'L1_IDENTITY.json').read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.is_file() else {}

    inv = field_inventory(records)
    inv_completed = field_inventory(ok)

    # ---- THE MATRIX -----------------------------------------------------------------------
    def row(q, item, source, computable, note):
        return {'question': q, 'item': item, 'source': source, 'computable': computable,
                'note': note}

    matrix = [
        # Q1 -- the headline non-identifiability
        row('Q1', 'complete ordered legal action list per state',
            'ABSENT: the record keeps only `V1.first_step_legal_sha256` (a digest) and '
            '`n_legal_first_step` (a count). `steps`, `records` and `legal_trace` are NOT archived.',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'a digest cannot be inverted to the candidate set, and a count does not name candidates'),
        row('Q1', 'per-candidate rule features (ready_time, distance, capacity, zone)',
            'ABSENT: no such field exists at any level of `L1_STATE_RECORDS.jsonl`.',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            're-enumerating the environment to regenerate them is FORBIDDEN by the task ruling'),
        row('Q1', 'cheap-rule reproduction RATE of F\'s first step',
            'requires the two items above',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'NO rate is produced. Producing one would require inventing the candidate set.'),
        row('Q1', 'the 90% rule-capture threshold',
            'n/a -- the governing statistic is uncomputable',
            'EXPLORATORY_STANDARD_NOT_APPLIED',
            'EXPLORATORY, set AFTER the fact in the analysis plan; it is NOT a formal L1 '
            'pre-registration criterion and no result is ever reported against it here'),
        row('Q1', 'the two arms\' CHOSEN first-step actions',
            '`V1.f_action` and `V1.d_geo_action` (7 fields each, incl. incremental_distance)',
            'COMPUTABLE',
            'the chosen actions ARE archived; only their candidate set is not'),
        row('Q1', 'D-Geo\'s choice equals the minimum-incremental-distance choice',
            '`V1.d_geo_action.incremental_distance` is archived, but the other candidates\' '
            'distances are not',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'cannot verify minimality without the full distance vector'),
        # Q2
        row('Q2', 'divergence rate over ALL comparable states',
            '`V1.verdict` over records with `status == COMPLETED` (n = 63)',
            'COMPUTABLE', 'denominator is every state where BOTH arms produced a first step'),
        row('Q2', 'stratum by ordered-legal length',
            '`n_legal_first_step` (archived, integer 6..26)',
            'COMPUTABLE', 'each bin reports denominator and numerator'),
        row('Q2', 'stratum by mask-pool size',
            '`n_mask` (archived); NOTE all 64 states have n_mask = 2, so this stratum is degenerate',
            'COMPUTABLE_BUT_DEGENERATE',
            'reported as such rather than dropped silently'),
        row('Q2', 'stratum by timestep / retained-edge ratio',
            'ABSENT: `timestep` lives in the scorer state and was never persisted to the record',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'NOT fabricated, NOT interpolated, NOT regenerated'),
        row('Q2', 'stratum by day_id',
            '`day_id` (archived)',
            'COMPUTABLE_DESCRIPTIVE_ONLY',
            'day index is a stratum label, never evidence of a temporal trend'),
        row('Q2', 'stratum by slot_kind / same-vehicle',
            '`V1.f_action.slot_kind`, `V1.d_geo_action.slot_kind`, both `vehicle_id`',
            'COMPUTABLE', 'descriptive only'),
        # Q3
        row('Q3', 'cross tabulation of first-step actions vs plan-hash equality',
            '`V1.*_action` and `plan_hashes.F` / `plan_hashes.D-Geo`',
            'COMPUTABLE', 'the plan hash is used ONLY as an equality test'),
        row('Q3', 'number of vehicles affected by the plan difference',
            'NOT DERIVABLE: `plan_hashes` holds an opaque 16-hex identity per arm',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'a hash cannot be inverted to a route structure; forbidden to attempt'),
        row('Q3', 'number of customers moved / route or distance delta',
            'NOT DERIVABLE: the per-arm `plan_after` structures were never persisted',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'only the two CHOSEN first-step `incremental_distance` values exist, which are per-action '
            'and NOT a plan distance'),
        row('Q3', 'divergence that converged back to an identical plan',
            '`plan_hashes.F == plan_hashes.D-Geo` while `V1.verdict == DIVERGENT`',
            'COMPUTABLE', 'counts "diverged then reconverged"; uncertified'),
        # Q4
        row('Q4', 'the unanswered state\'s identity and failure',
            '`state_hash`, `exception_type`, `error`, `failure_phase`, `traceback` (archived)',
            'COMPUTABLE', 'taken verbatim; nothing imputed'),
        row('Q4', 'the unanswered state\'s mask-pool size (the defect quantity)',
            '`manifest.watch`-independent: measured during the frozen-identity review and recorded in '
            'the closure adjudication, NOT in `L1_STATE_RECORDS.jsonl`',
            'COMPUTABLE_FROM_THE_CLOSURE_RECORD',
            'the value 27 is taken from the recorded measurement, not re-derived here'),
        row('Q4', 'the unanswered state\'s `n_legal_first_step`',
            'ABSENT: the record carries the key but the state never ran, so it is null',
            'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'NOT interpolated from neighbouring states'),
        row('Q4', 'any imputation of the state\'s V1/V2/V3',
            'explicitly forbidden',
            'NOT_PERFORMED', 'the state stays UNANSWERED_F_SHAPE_DEFECT'),
        # Q5
        row('Q5', 'labelled descriptive counts (outcomes, resources, attribution)',
            '`status`, `wall_s`, `per_arm_budget`, `per_arm_n_steps`, `v2_order_attribution`, '
            '`ledger_cross_check`',
            'COMPUTABLE', 'every figure carries its denominator'),
        row('Q5', 'second-arm remaining-allowance arithmetic',
            '`per_arm_budget.*.wall_seconds` and `per_state_wall_clock_s` (both archived)',
            'COMPUTABLE_ARITHMETIC_EXTRAPOLATION',
            'labelled an extrapolation from observed costs, NOT a measurement and NOT a prediction '
            'about any other run'),
        row('Q5', 'any significance test, effect size or causal claim',
            'n/a',
            'NOT_PERFORMED',
            'the 63 states were used for development; no test is run and none is reported'),
    ]

    # ---- Q2 -----------------------------------------------------------------------------
    def rate(subset):
        n = len(subset)
        d = sum(1 for r in subset if (r.get('V1') or {}).get('verdict') == 'DIVERGENT')
        return {'denominator': n, 'numerator_DIVERGENT': d,
                'rate': (d / n) if n else None,
                'IDENTICAL': sum(1 for r in subset
                                 if (r.get('V1') or {}).get('verdict') == 'IDENTICAL')}

    legal_bins = [(6, 10), (11, 15), (16, 20), (21, 27)]
    q2 = {
        'overall': rate(ok),
        'by_ordered_legal_length': [],
        'by_mask_pool_size': [],
        'by_day_id': [],
        'by_slot_kind_pair': [],
    }
    for lo, hi in legal_bins:
        sub = [r for r in ok if lo <= int(r.get('n_legal_first_step') or -1) <= hi]
        q2['by_ordered_legal_length'].append(dict(bin='%d-%d' % (lo, hi), **rate(sub)))
    mask_vals = sorted({int(r.get('n_mask') or -1) for r in ok})
    for m in mask_vals:
        sub = [r for r in ok if int(r.get('n_mask') or -1) == m]
        q2['by_mask_pool_size'].append(dict(bin=str(m), degenerate=len(mask_vals) == 1, **rate(sub)))
    for d in range(16):
        sub = [r for r in ok if int(r.get('day_id')) == d]
        q2['by_day_id'].append(dict(bin='day %d' % d, **rate(sub)))
    kinds = Counter()
    for r in ok:
        f = (r.get('V1') or {}).get('f_action') or {}
        g = (r.get('V1') or {}).get('d_geo_action') or {}
        kinds[(f.get('slot_kind'), g.get('slot_kind'))] += 1
    for k, n in sorted(kinds.items(), key=lambda x: repr(x[0])):
        sub = [r for r in ok
               if (((r.get('V1') or {}).get('f_action') or {}).get('slot_kind'),
                   ((r.get('V1') or {}).get('d_geo_action') or {}).get('slot_kind')) == k]
        q2['by_slot_kind_pair'].append(dict(bin='%s/%s' % k, **rate(sub)))

    # ---- Q3 -----------------------------------------------------------------------------
    diverge = [r for r in ok if (r.get('V1') or {}).get('verdict') == 'DIVERGENT']
    identical = [r for r in ok if (r.get('V1') or {}).get('verdict') == 'IDENTICAL']
    plan_same = [r for r in ok
                 if (r.get('plan_hashes') or {}).get('F')
                 == (r.get('plan_hashes') or {}).get('D-Geo')]
    plan_diff = [r for r in ok
                 if (r.get('plan_hashes') or {}).get('F')
                 != (r.get('plan_hashes') or {}).get('D-Geo')]
    q3 = {
        'plan_hash_equality': {'n_comparable': len(ok), 'plans_identical': len(plan_same),
                               'plans_differ': len(plan_diff)},
        'cross_tab': {
            'DIVERGENT_and_plan_differs': len([r for r in diverge if r in plan_diff]),
            'DIVERGENT_and_plan_same_RECONVERGED': len([r for r in diverge if r in plan_same]),
            'IDENTICAL_and_plan_same': len([r for r in identical if r in plan_same]),
            'IDENTICAL_and_plan_differs': len([r for r in identical if r in plan_diff]),
        },
        'same_customer_same_vehicle_among_DIVERGENT':
            len([r for r in diverge
                 if ((r['V1'].get('f_action') or {}).get('customer')
                     == (r['V1'].get('d_geo_action') or {}).get('customer'))
                 and ((r['V1'].get('f_action') or {}).get('vehicle_id')
                      == (r['V1'].get('d_geo_action') or {}).get('vehicle_id'))]),
        'first_step_incremental_distance': {
            'note': 'per-ACTION incremental distance of each arm\'s CHOSEN first action; this is NOT '
                    'a plan distance and NOT a route difference',
            'F': {'min': min(float(r['V1']['f_action']['incremental_distance']) for r in ok),
                  'median': statistics.median([float(r['V1']['f_action']['incremental_distance'])
                                               for r in ok]),
                  'max': max(float(r['V1']['f_action']['incremental_distance']) for r in ok)},
            'D_Geo': {'min': min(float(r['V1']['d_geo_action']['incremental_distance']) for r in ok),
                      'median': statistics.median([float(r['V1']['d_geo_action']
                                                         ['incremental_distance']) for r in ok]),
                      'max': max(float(r['V1']['d_geo_action']['incremental_distance'])
                                 for r in ok)},
            'n_states_D_geo_strictly_smaller': len(
                [r for r in ok if float(r['V1']['d_geo_action']['incremental_distance'])
                 < float(r['V1']['f_action']['incremental_distance'])]),
            'n_states_F_strictly_smaller': len(
                [r for r in ok if float(r['V1']['f_action']['incremental_distance'])
                 < float(r['V1']['d_geo_action']['incremental_distance'])]),
            'CAUTION': 'a smaller first-step incremental distance is NOT a better plan, NOT a quality '
                       'difference and NOT a gain; it is one scalar of one action.',
        },
        'explicitly_not_derived': [
            'number of vehicles affected by the plan difference',
            'number of customers moved',
            'route or total-distance difference between the two plans',
        ],
    }

    # ---- Q4 -----------------------------------------------------------------------------
    e = exc[0] if exc else {}
    closure = None
    for cand in sorted((EV / 'l1_gates').glob('L1_V2_CLOSURE_ADJUDICATION.json')):
        closure = json.loads(cand.read_text(encoding='utf-8'))
    cu = ((closure or {}).get('UNANSWERED_STATES') or [{}])[0]
    q4 = {
        'marker': 'UNANSWERED_F_SHAPE_DEFECT',
        'day_id': e.get('day_id'), 'position': e.get('position'),
        'state_hash': e.get('state_hash'),
        'exception_type': e.get('exception_type'), 'error': e.get('error'),
        'failure_phase': e.get('failure_phase'),
        'mask_archived': e.get('mask'), 'n_mask_archived': e.get('n_mask'),
        'n_legal_first_step_archived': e.get('n_legal_first_step'),
        'defect_quantity_from_the_closure_record': {
            'M_max_frozen': (cu.get('measured') or {}).get('M_max_frozen'),
            'mask_pool_max_over_64_states': (cu.get('measured') or {}).get('mask_pool_max_over_64_states'),
            'source': 'l1_gates/L1_V2_CLOSURE_ADJUDICATION.json (a recorded measurement, cited)',
        },
        'imputation_performed': False,
        'interpolation_performed': False,
        're_derivation_performed': False,
        'note': 'the record carries n_legal_first_step as a key but the state never ran, so its value '
                'is null; that null is REPORTED, not filled',
    }

    # ---- Q5 -----------------------------------------------------------------------------
    spans = sorted(float(r['wall_s']) for r in ok if r.get('wall_s') is not None)
    arm_wall = {a: [float(((r.get('per_arm_budget') or {}).get(a) or {}).get('wall_seconds') or 0.0)
                    for r in ok] for a in ('F', 'D-Geo')}
    cap = float((ok[0].get('per_state_wall_clock_s') if ok else 10.0) or 10.0)
    first_arm = (ok[0].get('arm_order') or ['F', 'D-Geo'])[0] if ok else 'F'
    # the SECOND arm's remaining allowance is `cap - (what the first arm spent)`
    remaining = [cap - w for w in arm_wall[first_arm]]
    max_first = max(arm_wall[first_arm])
    mult = 2.0
    q5 = {
        'labelled_descriptive': {
            'n_records': len(records), 'n_completed': len(ok), 'n_exception': len(exc),
            'outcome_counts': dict(Counter(r.get('status') for r in records)),
            'n_ledger_ok': sum(1 for r in ok if r.get('ledger_ok')),
            'n_ledger_problems': sum(len(r.get('ledger_problems') or []) for r in ok),
            'order_attribution_counts': dict(Counter(r.get('v2_order_attribution')
                                                     for r in records)),
            'per_arm_full_repair_completed': {a: sum(1 for r in ok
                                                     if ((r.get('V2') or {}).get(a) or {})
                                                     .get('full_repair_completed'))
                                              for a in ('F', 'D-Geo')},
            'per_arm_scorer_calls': {a: sum(int(((r.get('ledger_cross_check') or {})
                                                 .get('per_arm_calls') or {}).get(a, 0))
                                            for r in ok) for a in ('F', 'D-Geo')},
            'per_state_wall_s': {'min': spans[0], 'median': statistics.median(spans),
                                 'max': spans[-1], 'cap': cap},
        },
        'arithmetic_extrapolation': {
            'LABEL': 'ARITHMETIC EXTRAPOLATION from observed costs -- NOT a measurement and NOT a '
                     'prediction about any other run',
            'first_arm': first_arm, 'second_arm': (ok[0].get('arm_order') or ['F', 'D-Geo'])[1],
            'cap_s': cap,
            'first_arm_wall_s': {'min': min(arm_wall[first_arm]),
                                 'median': statistics.median(arm_wall[first_arm]),
                                 'max': max_first},
            'second_arm_remaining_allowance_s': {'min': min(remaining),
                                                 'median': statistics.median(remaining),
                                                 'max': max(remaining)},
            'observed_scenario': 'the first arm consumed at most %.3f s of the %.1f s allowance, so '
                                 'the second arm always retained at least %.3f s' % (max_first, cap,
                                                                                    min(remaining)),
            'hypothetical': 'if the first arm cost %gx the observed maximum (%.3f s), the second arm '
                            'would have %.3f s left' % (mult, max_first * mult,
                                                        cap - max_first * mult),
            'would_exceed_the_cap_at': cap / max_first if max_first else None,
            'would_exceed_label': 'multiple of the observed maximum first-arm cost at which the '
                                  'second arm would be left with zero allowance',
            'NOT_A_CAUSAL_CLAIM': 'this says nothing about what any conclusion would have been under '
                                  'a different budget; the observed run had zero timeouts and zero '
                                  'order-attributable states',
        },
        'tests_performed': [],
        'significance_or_effect_size': 'NOT_PERFORMED (development states; no test is run)',
    }

    # ---- independence + read-only proof ---------------------------------------------------
    src = Path(__file__).read_text(encoding='utf-8')
    own_imports = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            own_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            own_imports.append(node.module)
    leaked = sorted(m for m in FORBIDDEN_IMPORTS if m in sys.modules)
    # A read-only proof that is actually meaningful: the run directory ALREADY contains scoring
    # artifacts (`L1_READY_*`, `L1_WARMUP_*`) because the real execution created them.  Claiming
    # "no scoring artifact exists" would therefore be false and would fail the analysis for the run's
    # own history.  What this tool can prove is that it added NOTHING: it records the complete file
    # listing of the run directory, which the independent audit re-computes and diffs.
    run_dir_files = sorted(str(p.relative_to(RUN)).replace('\\', '/')
                           for p in RUN.rglob('*') if p.is_file())
    pre_existing_scoring = sorted(p for p in run_dir_files
                                  if any(m in p for m in SCORING_ARTIFACT_MARKERS))
    indep_ok = (not leaked
                and not any(m in ' '.join(own_imports) for m in FORBIDDEN_IMPORTS))

    payload = {
        'schema': 'd3_l1_v2_readonly_mechanism_analysis',
        'run_id': 'd3l1-l1-two-arm-20261016-l1v2',
        'read_only_proof': {
            'imports_declared_here': sorted(set(own_imports)),
            'forbidden_modules_in_sys_modules': leaked,
            'forbidden_modules_imported': [m for m in FORBIDDEN_IMPORTS
                                           if m in ' '.join(own_imports)],
            'n_run_dir_files_at_analysis': len(run_dir_files),
            'run_dir_files_at_analysis': run_dir_files,
            'pre_existing_scoring_artifacts_from_the_EXECUTION': pre_existing_scoring,
            'scoring_artifacts_created_by_this_tool': [],
            'how_no_new_file_is_proven': 'the complete run-directory file listing is recorded here; '
                                         'the independent audit re-lists the directory and requires '
                                         'the diff to be EMPTY',
            'note': 'the run directory is pre-existing and untouched; this tool writes only its JSON '
                    'output outside the run directory',
            'independent': indep_ok,
        },
        'field_inventory_all_records': inv,
        'field_inventory_completed_records': inv_completed,
        'nested_inventory_V1': nested_field_inventory(records, 'V1'),
        'nested_inventory_V2': nested_field_inventory(records, 'V2'),
        'nested_inventory_audit': nested_field_inventory(records, 'audit'),
        'ABSENT_FIELDS_ASSUMED_BY_THE_PLAN': {
            'steps': 'ABSENT from every record',
            'records': 'ABSENT from every record (only `plan_hashes` survives, per arm)',
            'legal_trace': 'ABSENT from every record',
            'timestep': 'ABSENT from every record',
            'consequence': 'the plan\'s assumption that the JSONL carries complete per-step action '
                           'lists, per-step legal traces and timestep is FALSE; Q1 and parts of Q2/Q3 '
                           'are therefore unidentifiable rather than merely inconvenient',
        },
        'identifiability_matrix': matrix,
        'Q1_cheap_rule_reproduction': {
            'status': 'NOT_IDENTIFIABLE_FROM_ARCHIVED_RECORDS',
            'rate_produced': None,
            'why': 'no complete ordered legal candidate list and no per-candidate rule features are '
                   'archived; a rate would require inventing the candidate set, and re-enumerating '
                   'the environment is forbidden',
            'threshold_90pct': 'EXPLORATORY_STANDARD_NOT_APPLIED -- set after the fact in the plan; it '
                               'is NOT a formal L1 pre-registration criterion and no result is '
                               'reported against it',
        },
        'Q2_divergence_strata': q2,
        'Q3_action_vs_plan_cross_tab': q3,
        'Q4_unanswered_state': q4,
        'Q5_descriptive_and_extrapolation': q5,
        'terminology_locks': {
            'cheap_rule_hit_rate': 'NOT PRODUCED (unidentifiable)',
            'plan_difference': 'UNCERTIFIED difference; never "better"/"worse"/"gain"/"loss"',
            'first_step_incremental_distance': 'one scalar of one action; NOT a plan distance',
            'significance': 'NOT PERFORMED',
        },
        'not_run': ['any new real F scoring', 'training', 'online rollout', 'plan installation',
                    'L2', 'L3', 'the 40-day batch', 'the final test',
                    'environment re-enumeration', 'result re-construction'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)),
        'inputs': {str(p.relative_to(REPO)): sha(p) for p in
                   (RUN / 'L1_STATE_RECORDS.jsonl', RUN / 'L1_IDENTITY.json', MANIFEST)},
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print(json.dumps({
        'Q1': payload['Q1_cheap_rule_reproduction']['status'],
        'Q2_overall': q2['overall'],
        'Q3_plan_hash': q3['plan_hash_equality'],
        'Q3_cross_tab': q3['cross_tab'],
        'Q5_outcomes': q5['labelled_descriptive']['outcome_counts'],
        'independent': indep_ok,
        'absent_fields': list(payload['ABSENT_FIELDS_ASSUMED_BY_THE_PLAN']),
        'wrote': str(out),
    }, indent=1, default=repr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
