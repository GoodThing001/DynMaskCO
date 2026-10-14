"""Read-only analysis of the already sealed two timed partial requests.

No model, C0, source closure, or previous auditor is executed.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/STAGE_ADJUDICATION.json'
OUT = ROOT / 'research/review_3080_20261010/goal_round_20261013_r9_strong_owner_resource/DELIVERY_SAVED_TIMING_DIAGNOSIS.json'

def main():
    old = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = []
    for row in old['rows']:
        exact = {int(k): Fraction(*v) for k, v in row['completed_phase_exact_seconds'].items()}
        # Top-level phase 7 includes phases 20..23. These are never added together.
        recording = sum((exact.get(k, Fraction(0)) for k in (8, 10, 11)), Fraction(0))
        completed = row['completed_insertions']
        rows.append({
            'arm': row['arm'],
            'completed_insertions': completed,
            'unfinished_insertions': 25 - completed,
            'outer_elapsed_seconds': row['elapsed_seconds'],
            'completed_recording_phases_8_10_11_seconds': float(recording),
            'completed_head_phase22_seconds': float(exact[22]),
            'completed_decode_phase20_seconds': float(exact[20]),
            'completed_encoder_phase4_seconds': float(exact[4]),
            'recording_fraction_of_outer_elapsed': float(recording) / row['elapsed_seconds'],
            'head_fraction_of_outer_elapsed': float(exact[22]) / row['elapsed_seconds'],
            'source_gate_reached': row['dependency_gate_reached'],
            'parameter_gate_reached': row['parameter_gate_reached'],
            'selected': row['selected'],
        })
    checks = {
        'exactly_previously_sealed_two_arms': [x['arm'] for x in rows] == ['pretrained_initial','random888_initial'],
        'both_incomplete_and_unselected': all(x['unfinished_insertions'] > 0 and not x['selected'] for x in rows),
        'all_publication_gates_unreached': all(not x['source_gate_reached'] and not x['parameter_gate_reached'] for x in rows),
        'recording_less_than_head_each': all(x['completed_recording_phases_8_10_11_seconds'] < x['completed_head_phase22_seconds'] for x in rows),
        'old_independent_audit_not_rerun': old['independent_failures'] == 0,
    }
    result = {
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'scope': 'read-only completed-span mechanism diagnosis; not a speed counterfactual',
        'rows': rows, 'checks': checks,
        'failed': [k for k, v in checks.items() if not v],
        'interpretation': 'Completed score-head spans dominate recorded serialization phases in both killed requests; deleting recording alone is insufficient evidence for a complete 8+2 request. Missing insertion steps and publication gates have unknown positive cost.',
        'model_run': False, 'old_QA_rerun': False,
        'full_request_budget_qualified': False,
        'stable_online_gain': False,
    }
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'rows': len(rows), 'failed': result['failed'],
                      'recording_seconds': [r['completed_recording_phases_8_10_11_seconds'] for r in rows],
                      'head_seconds': [r['completed_head_phase22_seconds'] for r in rows]}))
    return int(bool(result['failed']))

if __name__ == '__main__':
    raise SystemExit(main())
