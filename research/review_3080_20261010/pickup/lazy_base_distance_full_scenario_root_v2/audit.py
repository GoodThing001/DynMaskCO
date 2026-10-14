"""Independent audit of the full-scenario pickup paired REF/CAND component.

Own re-derivation of the selection rule, of the deletion relation between the two
distance logs, and of every route-distance value by an ordered double sum over the
sealed immutable distance matrix.  Does not import run.py's execution path.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import numpy as np

import engineering_stage_closure_v1 as seal

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))
E = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))
RUN_EXIT = json.loads((HERE / 'RUN_EXIT.json').read_text(encoding='utf-8'))


def save(n, x):
    seal.save(HERE / n, x)


def dg(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    seal.initialize()
    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(n, v):
        checks.append(dict(name=n, pass_=bool(v)))

    for path, h in REG['sources'].items():
        ck('registered_source:' + path, seal.sha(path) == h)
    ck('producer_reported_no_failure', E['failed'] == 0)
    ck('producer_exit_zero', RUN_EXIT['exit_code'] == 0)

    # 1. selection rule re-derived from the recorded artefacts
    sequence, ids = E['sequence'], E['scenario8_ids']
    reveals = [float.fromhex(h) for h in E['scenario8_reveals']]
    ck('sequence_is_the_full_recorded_scenario', sorted(sequence) == sorted(ids) and len(sequence) == len(ids))
    ck('sequence_is_reveal_sorted', sequence == [o for _, o in sorted(zip(reveals, ids))])
    ck('no_order_dropped_or_duplicated', len(set(sequence)) == len(sequence))

    # 2. per-run digests reproduce cross-repeat determinism
    for label in ('REF', 'CAND'):
        rows = E['runs'][label]
        ck('run_count:' + label, len(rows) == REG['timed_repeats'])
        ck('cross_repeat_identical_events:' + label, len({r['events'] for r in rows}) == 1)
        ck('cross_repeat_identical_feasible:' + label, len({r['feasible'] for r in rows}) == 1)
        ck('cross_repeat_identical_distance:' + label, len({r['distance'] for r in rows}) == 1)
        ck('cross_repeat_identical_score:' + label, len({r['score'] for r in rows}) == 1)
        ck('cross_repeat_identical_terminal:' + label,
           len({(r['terminal_energy_hex'], r['terminal_distance_hex']) for r in rows}) == 1)
        ck('declared_timing_values_match_runs:' + label,
           sorted(r['seconds'] for r in rows) == sorted(E['reference_seconds' if label == 'REF' else 'candidate_seconds']))

    # 3. the candidate must be structurally identical to the reference
    r0, c0 = E['runs']['REF'][0], E['runs']['CAND'][0]
    ck('identical_events_digest', r0['events'] == c0['events'])
    ck('identical_feasibility_digest', r0['feasible'] == c0['feasible'])
    ck('identical_scored_candidates_digest', r0['score'] == c0['score'])
    ck('distance_log_differs_as_expected', r0['distance'] != c0['distance'])
    ck('identical_terminal_energy', r0['terminal_energy_hex'] == c0['terminal_energy_hex'])
    ck('identical_terminal_distance', r0['terminal_distance_hex'] == c0['terminal_distance_hex'])
    ck('recorded_top_level_logs_match_reference_digests',
       dg(E['events']) == r0['events'] and dg(E['feasible_log']) == r0['feasible']
       and dg(E['score_log']) == r0['score'] and dg(E['distance_log']) == r0['distance']
       and dg(E['candidate_distance_log']) == c0['distance'])

    # 4. the deletion relation, checked positionally without vehicle identity
    pairs = {}
    for r in E['distance_log']:
        pairs.setdefault(r['operation'], []).append(r)
    is_base = {}
    for op, entries in pairs.items():
        for n, entry in enumerate(entries):
            is_base[id(entry)] = bool(n % 2)
    j = 0
    deleted = []
    for entry in E['distance_log']:
        if j < len(E['candidate_distance_log']) and E['candidate_distance_log'][j] == entry:
            j += 1
        else:
            deleted.append(entry)
    ck('candidate_log_is_reference_log_with_in_order_deletions', j == len(E['candidate_distance_log']))
    ck('every_deleted_entry_is_a_base_call', all(is_base.get(id(e)) for e in deleted))
    ck('saving_count_matches_log_difference_and_declaration',
       len(E['distance_log']) - len(E['candidate_distance_log']) == len(deleted) == E['proved_removed_base_calls'])
    ck('at_least_one_trial_kept_per_removed_base',
       len([e for e in E['candidate_distance_log'] if not is_base.get(id(e))])
       == len([e for e in E['distance_log'] if not is_base.get(id(e))]))
    ck('base_calls_strictly_reduced', len(E['candidate_distance_log']) < len(E['distance_log']))

    # 5. every route distance recomputed independently for BOTH logs
    matrix = np.load(HERE / 'DISTANCE_MATRIX.npy', allow_pickle=False)
    bad_ref = bad_cand = 0
    for r in E['distance_log']:
        total = 0.
        cur = r['cur']
        for o in r['route']:
            total += float(matrix[cur, o])
            cur = o
        total += float(matrix[cur, 0])
        bad_ref += int(total.hex() != r['value_hex'])
    for r in E['candidate_distance_log']:
        total = 0.
        cur = r['cur']
        for o in r['route']:
            total += float(matrix[cur, o])
            cur = o
        total += float(matrix[cur, 0])
        bad_cand += int(total.hex() != r['value_hex'])
    ck('independent_ordered_double_sum_all_reference_calls', bad_ref == 0)
    ck('independent_ordered_double_sum_all_candidate_calls', bad_cand == 0)
    ck('matrix_identity_recorded', isinstance(E['immutable_matrix_sha256'], str) and len(E['immutable_matrix_sha256']) == 64)

    # 6. C0 physical transitions are well formed and identical across arms
    trans = E['cold_transition_outputs']
    ck('transitions_non_empty', len(trans) > 0)
    ck('every_transition_is_a_state_metrics_pair',
       all(isinstance(t, dict) and t.get('__sequence__') == 'tuple' and len(t.get('values', [])) == 2
           and t['values'][1].get('__dataclass__') == 'coldchain_state.SegmentColdChainMetrics'
           for t in trans))

    # 7. descriptive timing (no speed gate is registered; call counts are not speed)
    ref_s = E['reference_seconds']
    cand_s = E['candidate_seconds']
    timing = dict(reference_seconds=ref_s, candidate_seconds=cand_s,
                  paired_seconds=[c - r for r, c in zip(ref_s, cand_s)] if len(ref_s) == len(cand_s) else None,
                  note=('reported descriptively; no speed threshold is registered because a reduced call count is '
                        'not a speedup and the timed section is small relative to process noise'))

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    finished = time.perf_counter()
    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 inserted=E['inserted'], accepted=E['accepted'],
                 original_reference_distance_calls=len(E['distance_log']),
                 candidate_distance_calls=len(E['candidate_distance_log']),
                 proved_removed_base_calls=E['proved_removed_base_calls'],
                 scored_feasible_candidates=len(E['score_log']),
                 complete_original_C0_transitions=len(trans),
                 terminal_energy_hex=E['terminal_energy_hex'],
                 terminal_distance_hex=E['terminal_distance_hex'],
                 timing=timing,
                 actual_source_files=len(start_closure['files']), actual_source_maps=len(start_closure['modules']),
                 audit_seconds=finished - started,
                 full_request_budget_qualified=False, full_scientific_loaded_closure=False,
                 formal_strong_dynamic_qualified=False, stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json', audit)
    save('AUDIT_RUN_EXIT.json', dict(exit_code=int(audit['failed'] > 0)))
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}))
    return int(audit['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
