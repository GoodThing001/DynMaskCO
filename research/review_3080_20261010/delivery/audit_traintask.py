"""Independent read-only arithmetic and implementation-boundary audit.

Writes only a NEW caller-selected report; never edits source result artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, out):
    base = root / 'results/research_20261010_continuation_v2/delivery'
    names = ['train_run_v1', 'dev_gate_v1', 'guide_diag_v1', 'infer_ident_run1', 'infer_ident_run2']
    artifacts = {}
    pins = {}
    for name in names:
        path = base / name / 'result.json'
        if not path.is_file():
            raise ValueError('Missing original evidence: ' + str(path))
        artifacts[name] = json.loads(path.read_text(encoding='utf-8'))
        pins[str(path.relative_to(root))] = sha(path)
    dev = artifacts['dev_gate_v1']
    rows = dev['per_instance']
    ids = [r['instance'] for r in rows]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(32)):
        raise ValueError('Not exactly 32 unique actual integer identifiers')
    means = {a: float(np.mean([r[a] for r in rows])) for a in dev['mean_final_cost']}
    contrasts = {}
    for control in ['regret2_nolearn', 'zeroshot_guide', 'discard_control']:
        ref = np.asarray([r[control] for r in rows], np.float64)
        treatment = np.asarray([r['trained'] for r in rows], np.float64)
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(treatment)) or np.any(ref <= 0):
            raise ValueError('Invalid retained costs')
        d = (ref - treatment) / ref
        rng = np.random.default_rng(dev['bootstrap_seed'])
        draws = d[rng.integers(0, len(d), size=(dev['bootstrap_repetitions'], len(d)))].mean(axis=1)
        ci = np.percentile(draws, [2.5, 97.5]).tolist()
        original = dev['paired_contrasts']['trained_minus_' + control]
        contrasts[control] = {'mean_relative_reduction': float(d.mean()), 'ci95': ci,
            'improved': int(np.sum(d > 1e-9)), 'worse': int(np.sum(d < -1e-9)),
            'mean_error': abs(float(d.mean()) - original['mean_relative_cost_reduction']),
            'ci_max_error': float(np.max(np.abs(np.asarray(ci) - original['ci95'])))}
    train = artifacts['train_run_v1']
    diagnostic = artifacts['guide_diag_v1']
    source_paths = ['delivery_traintask_train.py', 'delivery_trained_guide.py', 'delivery_traintask_guide_diagnostic.py', 'delivery_traintask_dev_gate.py', 'delivery_traintask_recipe.json']
    for name in source_paths:
        path = root / 'scripts/research_20261010' / name
        pins[str(path.relative_to(root))] = sha(path)
    result = {'kind': 'independent_traintask_evidence_scope_audit', 'pins': pins,
        'means_recomputed': means, 'paired_contrasts': contrasts,
        'numeric_recomputation_consistent': all(v['mean_error'] < 1e-12 and v['ci_max_error'] < 1e-12 for v in contrasts.values()),
        'train_certified_available': train['n_certified_available'], 'train_cycles_used': train['n_cycles_used'],
        'actual_optimization_insertion_records': train['n_training_steps'],
        'full_rule_steps_audited': train['fidelity']['full_rule_steps_checked'],
        'per_customer_option_steps_audited': train['fidelity']['per_customer_steps_checked'],
        'base_hash_equal': train['base_param_hash_before'] == train['base_param_hash_after'],
        'head_hash_changed': train['head_param_hash_before'] != train['head_param_hash_after'],
        'diagnostic_summary_retained': diagnostic['summary'],
        'interpretation': {'full_encoder_decoder_task_training_tested': False,
            'head_mask_context_consumed': False, 'signed_advantage_weighted_ce_matches_recipe': False,
            'instant_min_delta_win_test_can_disprove_global_utility': False,
            'independent_full_route_replay_qualified': False,
            'fresh_confirmation_gain_established': False,
            'stable_online_gain_established': False},
        'limits': ['Arithmetic uses stored costs; original DEV per_instance has no routes or replay traces.',
                   'Inference identity artifacts are pinned and preserved; no model re-inference in this audit.',
                   'Signed gain coefficient inside score is different from signed weighting of CE.']}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'numeric_consistency': result['numeric_recomputation_consistent'], 'train_cycles_used': train['n_cycles_used']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    main(args.root, args.out)
