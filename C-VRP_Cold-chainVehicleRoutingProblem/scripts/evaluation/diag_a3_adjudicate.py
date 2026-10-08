# -*- coding: utf-8 -*-
"""A3 independent, fail-closed adjudicator for nine MaskCO batches.

Recomputes paired statistics from all 40 day rows and checks source, data,
contract, configuration and model identities. --baseline-map can pair the
original, immutable A3 MaskCO outputs with newly rerun same-source baselines;
the original gate.json files are never changed. `formal_passed` is separate
from the numeric threshold. C-B is a combined training/inference contrast.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

MAIN_PENALTY = 'p_c=(5,10,15)'
DELTA_3 = 20.0
TIERS = ('p_c=0', MAIN_PENALTY, 'p_c=(10,20,30)')
ARMS = ('B', 'C', 'D')
SEEDS = ('20260926', '20260930', '20261001')
_PAIR_FIELDS = ('shared_source_sha256', 'contract_sha256', 'data_sha256',
                'data_meta_sha256', 'budget_B', 'cooling_share', 'shared_config')


def _check_rows(rows, expected=40):
    """Check the exact day set before any statistic is computed."""
    if not isinstance(rows, list):
        return ['rows_missing']
    problems = []
    if len(rows) != expected:
        problems.append('expected_%d_rows_got_%d' % (expected, len(rows)))
    try:
        ids = [r['i'] for r in rows]
        if any(type(i) is not int for i in ids) or sorted(ids) != list(range(expected)):
            problems.append('day_ids_not_exact_0_to_%d' % (expected - 1))
        if any(r.get('utility') is None or not np.isfinite(r['utility']) for r in rows):
            problems.append('nonfinite_utility')
        if any(r.get('hard_feasible') is not True for r in rows):
            problems.append('hard_violation')
    except (KeyError, TypeError, ValueError):
        problems.append('malformed_day_row')
    return problems


def _check_identity(ident, baseline_ident):
    """Independently compare the two sealed execution identities."""
    problems = []
    for label, obj in (('maskco', ident), ('baseline', baseline_ident)):
        if not isinstance(obj, dict):
            problems.append(label + '_identity_missing')
            continue
        if obj.get('source_stable') is not True or not obj.get('source_sha256') \
                or obj.get('source_sha256') != obj.get('source_end_sha256'):
            problems.append(label + '_source_not_stable')
        if not isinstance(obj.get('shared_source_sha256'), dict):
            problems.append(label + '_shared_source_missing')
    if not isinstance(ident, dict) or not isinstance(baseline_ident, dict):
        return problems
    for key in _PAIR_FIELDS:
        if not ident.get(key) or not baseline_ident.get(key):
            problems.append(key + '_missing')
        elif ident[key] != baseline_ident[key]:
            problems.append(key + '_mismatch')
    for name in ('train', 'gate'):
        x = (ident.get('seeds') or {}).get(name)
        y = (baseline_ident.get('seeds') or {}).get(name)
        if x is None or y is None or int(x) != int(y):
            problems.append(name + '_seed_mismatch')
    return problems


def _stat_exact(a_rows, b_rows, seed, expected=40):
    """Recompute the registered 2,000-draw day bootstrap from validated rows."""
    problems = _check_rows(a_rows, expected) + _check_rows(b_rows, expected)
    if problems:
        return {'mean': None, 'ci_lo': None, 'ci_hi': None, 'n_paired': 0,
                'adjudicable': False, 'problems': sorted(set(problems))}
    aa = {r['i']: float(r['utility']) for r in a_rows}
    bb = {r['i']: float(r['utility']) for r in b_rows}
    d = np.array([aa[i] - bb[i] for i in range(expected)], dtype=float)
    rng = np.random.default_rng(int(seed) + 777)
    means = np.array([d[rng.integers(0, expected, expected)].mean()
                      for _ in range(2000)])
    return {'mean': float(d.mean()), 'ci_lo': float(np.percentile(means, 2.5)),
            'ci_hi': float(np.percentile(means, 97.5)), 'n_paired': expected,
            'adjudicable': True, 'problems': []}


def _baseline_block(doc, seed, tier):
    if 'gate' in doc:
        return (doc.get('gate') or {}).get(tier), (doc.get('identity') or {})
    result = (doc.get('results') or {}).get(str(seed)) or {}
    return result.get(tier), result.get('_identity') or {}


def _rows(blk, arm):
    return blk.get('per_day', {}).get(arm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--lock', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline-map', default=None,
                    help='可选：同源码补跑基线 JSON，须含 gate_json_by_seed；原 gate 不改写')
    ap.add_argument('--allow-partial', action='store_true',
                    help='仅用于诊断预演；输出永不标记为正式资格')
    args = ap.parse_args(argv)

    with open(args.lock, encoding='utf-8') as f:
        lock = json.load(f)
    baseline_map = None
    if args.baseline_map:
        with open(args.baseline_map, encoding='utf-8') as f:
            baseline_map = json.load(f).get('gate_json_by_seed')
        if not isinstance(baseline_map, dict):
            raise SystemExit('--baseline-map must contain gate_json_by_seed')
    if lock.get('baseline_policy') != 'density':
        raise SystemExit('A3 registered baseline_policy must be density')

    docs = {}
    batch_keys = {}
    global_problems = []
    for d in args.dirs:
        p = os.path.join(d, 'gate.json')
        if not os.path.isfile(p):
            global_problems.append('gate_missing_%s' % d)
            continue
        with open(p, encoding='utf-8') as f:
            doc = json.load(f)
        docs[d] = doc
        parts = os.path.basename(os.path.normpath(d)).split('_')
        arms = [x for x in parts if x in ARMS]
        seeds = [x for x in parts if x in SEEDS]
        if len(arms) != 1 or len(seeds) != 1:
            global_problems.append('unrecognized_arm_seed_%s' % d)
            continue
        key = (arms[0], seeds[0])
        if key in batch_keys:
            global_problems.append('duplicate_batch_%s_%s' % key)
        batch_keys[key] = d
    expected_keys = {(a, s) for a in ARMS for s in SEEDS}
    for a, s in sorted(expected_keys - set(batch_keys)):
        global_problems.append('batch_missing_%s_%s' % (a, s))
    if set(batch_keys) - expected_keys:
        global_problems.append('unexpected_batches')

    baseline_cache = {}
    per_batch = {}
    batch_identity = {}
    table = {}
    maskco_rows = {}
    for (arm, seed), d in sorted(batch_keys.items()):
        doc = docs[d]
        cfg = doc.get('config') or {}
        s_blk = (doc.get('results') or {}).get(seed) or {}
        ident = s_blk.get('_identity') or {}
        batch_identity[(arm, seed)] = ident
        errors = []
        if set((doc.get('results') or {})) != {seed}:
            errors.append('result_seed_set_mismatch')
        if (doc.get('seeds') or {}).get('ran') != [int(seed)]:
            errors.append('ran_seed_mismatch')
        expected_cfg = {'dev_instances': 40, 'train_instances': 200,
                        'gate_seed': int(seed), 'K': 10, 'time_limit': 10.0,
                        'future_policy': 'density', 'shadow_mode': 'greedy',
                        'energy_pricing': 'marginal'}
        for name, value in expected_cfg.items():
            if cfg.get(name) != value:
                errors.append('config_%s_mismatch' % name)
        arm_cfg = {'B': ('pretrained', False),
                   'C': ('pretrained', True),
                   'D': ('random_cvrp', False)}
        expected_model_arm, expected_iterative = arm_cfg[arm]
        if cfg.get('arm') != expected_model_arm or \
                cfg.get('iterative') is not expected_iterative:
            errors.append('arm_training_or_inference_config_mismatch')
        if arm == 'C' and (cfg.get('rounds') != 1 or
                           cfg.get('remask_frac') != 0.5):
            errors.append('iterative_config_mismatch')
        if doc.get('model_generation_check') != 'ok':
            errors.append('model_generation_check_failed')
        run_exit = os.path.join(d, 'run.exit')
        if not os.path.isfile(run_exit) or 'RUN_EXIT=0' not in open(
                run_exit, encoding='utf-8').read().splitlines():
            errors.append('run_exit_not_zero')
        model_path = ident.get('model_bin')
        model_hash = ident.get('model_bin_sha256')
        if not model_path or not model_hash or not os.path.isfile(model_path):
            errors.append('model_file_or_hash_missing')
        elif hashlib.sha256(open(model_path, 'rb').read()).hexdigest() != model_hash:
            errors.append('model_hash_mismatch')

        base_path = ((baseline_map or {}).get(seed) if baseline_map is not None
                     else cfg.get('baseline_from'))
        if not base_path or not os.path.isfile(base_path):
            errors.append('baseline_gate_missing')
            baseline = {}
        else:
            if base_path not in baseline_cache:
                with open(base_path, encoding='utf-8') as f:
                    baseline_cache[base_path] = json.load(f)
            baseline = baseline_cache[base_path]
        base_seed = (baseline.get('seeds') or {}).get('gate')
        if base_seed is None or int(base_seed) != int(seed):
            errors.append('baseline_seed_mismatch')
        base_cfg = baseline.get('config') or {}
        expected_base_cfg = {
            'train_instances': 200, 'gate_instances': 40,
            'gate_seed': int(seed), 'n_orders': 200, 'capacity': 50.0,
            'num_vehicles': 15, 'rho': 0.6, 'K': 10, 'time_limit': 10.0,
            'k_neighbors': 10, 'future_policy': 'density',
            'shadow_mode': 'greedy', 'energy_pricing': 'marginal',
            'standby_orders': None, 'overage_price': None,
        }
        for name, value in expected_base_cfg.items():
            if base_cfg.get(name) != value:
                errors.append('baseline_config_%s_mismatch' % name)
        if baseline_map is not None and base_path:
            seal_path = os.path.join(os.path.dirname(base_path), 'run.exit')
            seal_lines = (open(seal_path, encoding='utf-8').read().splitlines()
                          if os.path.isfile(seal_path) else [])
            if not {'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1'}.issubset(seal_lines):
                errors.append('baseline_completion_or_seal_failed')
        _, base_ident = _baseline_block(baseline, seed, MAIN_PENALTY)
        identity_problems = _check_identity(ident, base_ident)
        errors.extend(identity_problems)
        per_batch[os.path.basename(d)] = {
            'arm': arm, 'seed': seed, 'baseline_gate': base_path,
            'model_generation_check': doc.get('model_generation_check'),
            'source_stable': ident.get('source_stable'),
            'identity_verified': not identity_problems,
            'problems': sorted(set(errors))}

        for pname in TIERS:
            blk = s_blk.get(pname) or {}
            base_blk, _ = _baseline_block(baseline, seed, pname)
            base_blk = base_blk or {}
            a_rows = _rows(blk, 'maskco') or []
            b_rows = _rows(base_blk, 'cond_hist') or []
            tier_problems = list(errors)
            tier_problems += ['maskco_' + x for x in _check_rows(a_rows)]
            tier_problems += ['cond_' + x for x in _check_rows(b_rows)]
            if not blk:
                tier_problems.append('a3_tier_missing')
            if not base_blk:
                tier_problems.append('baseline_tier_missing')
            stat = _stat_exact(a_rows, b_rows, seed)
            tier_problems += stat['problems']
            numeric_passed = bool(stat['adjudicable'] and stat['mean'] >= DELTA_3
                                  and stat['ci_lo'] > 0)
            formal = not tier_problems
            table[(arm, seed, pname)] = {
                'mean': stat['mean'], 'ci_lo': stat['ci_lo'], 'ci_hi': stat['ci_hi'],
                'n_paired': stat['n_paired'], 'adjudicable': stat['adjudicable'],
                'delta_3': DELTA_3, 'numeric_passed': numeric_passed,
                'identity_verified': not identity_problems,
                'formal_adjudication': formal,
                'formal_passed': formal and numeric_passed,
                'problems': sorted(set(tier_problems)),
                'baseline_gate': base_path,
                'original_driver_result': (blk.get('main_comparison') or {}).get(
                    'maskco_minus_cond_hist'),
                'mean_timeouts_maskco': ((blk.get('arms') or {}).get('maskco') or {}).get(
                    'mean_timeouts')}
            maskco_rows[(arm, seed, pname)] = a_rows

    # 消融（跨批 maskco 逐日配对，日聚类 bootstrap）
    ablations = {}
    for seed in SEEDS:
        for pname in TIERS:
            for (a1, a2, name) in (('B', 'D', 'B_minus_D'), ('C', 'B', 'C_minus_B')):
                r1 = maskco_rows.get((a1, seed, pname))
                r2 = maskco_rows.get((a2, seed, pname))
                if not r1 or not r2:
                    continue
                st_ab = _stat_exact(r1, r2, seed)
                cross_problems = _check_identity(batch_identity.get((a1, seed)),
                                                 batch_identity.get((a2, seed)))
                for a in (a1, a2):
                    batch = per_batch.get(os.path.basename(batch_keys[(a, seed)]), {})
                    cross_problems += [a + '_' + p for p in batch.get('problems', [])
                                       if p.startswith(('model_', 'run_exit_', 'config_',
                                                        'arm_', 'iterative_', 'ran_',
                                                        'result_'))]
                st_ab['identity_verified'] = not cross_problems
                st_ab['formal_adjudication'] = bool(st_ab['adjudicable'] and
                                                    not cross_problems)
                st_ab['problems'] = sorted(set(st_ab['problems'] + cross_problems))
                st_ab['interpretation'] = (
                    'training_and_inference_confounded' if name == 'C_minus_B'
                    else 'training_recipe_comparison')
                ablations[(seed, pname, name)] = st_ab

    # 主门汇总（A3 预声明口径）：主档 (5,10,15)，每臂每种子
    summary = {}
    for arm in ARMS:
        for seed in SEEDS:
            k = (arm, seed, MAIN_PENALTY)
            entry = table.get(k)
            summary.setdefault(arm, {})[seed] = {
                'main_tier': (entry if entry else None),
                'tiers': {p: table[(arm, seed, p)] for p in
                          TIERS
                          if (arm, seed, p) in table}}

    partial = (set(batch_keys) != expected_keys or bool(global_problems))
    formal_all = bool(not partial and not args.allow_partial and
                      all(v['formal_adjudication'] for v in table.values()) and
                      len(table) == len(expected_keys) * len(TIERS))
    out = {
        'lock': {'baseline_policy': lock.get('baseline_policy'),
                 'caveats': lock.get('caveats')},
        'per_batch': per_batch,
        'delta_3': DELTA_3,
        'main_penalty': MAIN_PENALTY,
        'n_batches': int(len(docs)),
        'partial': partial,
        'formal_adjudicable': formal_all,
        'problems': sorted(set(global_problems)),
        'baseline_override_used': bool(args.baseline_map),
        'comparison_table': {str(k): v for k, v in table.items()},
        'ablations': {str(k): v for k, v in ablations.items()},
        'summary_by_arm_seed': summary,
        'note': '逐日数据和跨运行身份从原始 gate 独立重验。'
                'formal_passed = numeric_passed 且 formal_adjudication；'
                'C-B 同时变化训练谱系与推理方式，不是迭代独立贡献。',
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({'n_batches': out['n_batches'], 'partial': partial,
                      'formal_adjudicable': formal_all,
                      'problems': out['problems'], 'out': args.out},
                     ensure_ascii=False))
    return out


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result['formal_adjudicable'] or '--allow-partial' in sys.argv else 2)
