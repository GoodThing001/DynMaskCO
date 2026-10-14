"""Frozen-state teacher-objective diagnosis, never an online utility gate.

Only visible tokens/clock reach the sampler. Private future is used by the
sealed evaluator. All registered states are retained; ineligible risk is zero
by the TRAINING objective, not a claim that those states have no online regret.
No weights, source, checkpoint selection or primary protocol are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sealed(spec):
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('External artifact identity mismatch: ' + spec['path'])
    return json.loads(Path(spec['path']).read_text(encoding='utf-8'))


def decision_metrics(preference, margins, *, tau, weight_cap, supported):
    if tau <= 0 or weight_cap <= 0:
        raise ValueError('Positive registered tau/weight cap required')
    enabled = bool(preference['decision_eligible'] and supported)
    if not enabled:
        if margins is not None:
            raise ValueError('Do not fabricate sampled margins for an ineligible state')
        return {'evaluated': False, 'preference_risk': 0., 'eligible_teacher_regret': 0.,
                'direction_accuracy': None}
    m = np.asarray(margins, dtype=float)
    delta = preference['teacher_delta']
    if (not preference['teacher_valid'] or delta is None or not np.isfinite(delta) or delta == 0 or
            m.ndim != 1 or not len(m) or not np.isfinite(m).all()):
        raise ValueError('Invalid eligible teacher/margins; no filtering')
    disagreement = (m > 0) != (delta > 0)  # SAA ties reject.
    return {'evaluated': True,
            'preference_risk': float(min(abs(delta), weight_cap) * np.logaddexp(0., -np.sign(delta)*m/tau).mean()),
            'eligible_teacher_regret': float(abs(delta) * disagreement.mean()),
            'direction_accuracy': float((~disagreement).mean())}


def aggregate(rows, expected_keys, *, bootstrap_samples, bootstrap_seed):
    keys = [(r['day_id'], r['ordinal']) for r in rows]
    if (len(keys) != len(expected_keys) or len(set(keys)) != len(keys) or
            set(keys) != {tuple(k) for k in expected_keys}):
        raise ValueError('Missing/duplicate/unregistered diagnostic states')
    if not rows or bootstrap_samples <= 0:
        raise ValueError('Nonempty diagnostic day blocks required')
    arms = set(rows[0]['arms'])
    if any(set(row['arms']) != arms for row in rows):
        raise ValueError('Missing diagnostic arms')
    days = sorted({r['day_id'] for r in rows})
    if any(not np.isfinite(row['arms'][arm][metric]) for row in rows for arm in arms
           for metric in ('preference_risk', 'eligible_teacher_regret')):
        raise ValueError('Nonfinite risk; do not discard a state')
    by_day = {str(day): {arm: {metric: float(np.mean([
        row['arms'][arm][metric] for row in rows if row['day_id'] == day]))
        for metric in ('preference_risk', 'eligible_teacher_regret')} for arm in sorted(arms)} for day in days}
    rng = np.random.default_rng(bootstrap_seed)
    picks = rng.integers(0, len(days), size=(bootstrap_samples, len(days)))
    comparisons = {}
    if 'maskco' in arms:
        for control in sorted(arms - {'maskco'}):
            comparisons[control] = {}
            for metric in ('preference_risk', 'eligible_teacher_regret'):
                differences = np.array([by_day[str(day)]['maskco'][metric] - by_day[str(day)][control][metric]
                                        for day in days])
                ci = np.quantile(differences[picks].mean(axis=1), [.025, .975])
                comparisons[control][metric] = {'mean': float(differences.mean()),
                    'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]), 'better_direction': 'negative'}
    return {'states': len(rows), 'days': len(days), 'valid_teachers': sum(r['teacher_valid'] for r in rows),
        'eligible_teachers': sum(r['decision_eligible'] for r in rows),
        'evaluated_states': sum(r['arms'][next(iter(arms))]['evaluated'] for r in rows),
        'teacher_signs': {'positive': sum(r['teacher_delta'] is not None and r['teacher_delta'] > 0 for r in rows),
                          'negative': sum(r['teacher_delta'] is not None and r['teacher_delta'] < 0 for r in rows)},
        'mean_by_arm': {arm: {metric: float(np.mean([by_day[str(day)][arm][metric] for day in days]))
            for metric in ('preference_risk', 'eligible_teacher_regret')} for arm in sorted(arms)},
        'by_day': by_day, 'maskco_minus_controls': comparisons,
        'bootstrap': {'samples': bootstrap_samples, 'seed': bootstrap_seed, 'unit': 'day'},
        'scope': 'Fixed-state teacher objective; zero ineligible weights; not online return or unseen-day validation'}


def run(config_path, config_sha, out_path):
    if sha(config_path) != config_sha:
        raise ValueError('Diagnostic config identity mismatch')
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg['qualification'] not in ('engineering', 'teacher_objective_diagnostic'):
        raise ValueError('Unknown diagnostic scope')
    os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    if cfg.get('cpu_set') and hasattr(os, 'sched_getaffinity'):
        selected = set(cfg['cpu_set'])
        if not selected <= os.sched_getaffinity(0):
            raise ValueError('Diagnostic CPUs unavailable')
        os.sched_setaffinity(0, selected)
    ext = Path(cfg['frozen_extension']).resolve()
    scripts = ext / 'scripts'
    for folder in ('', 'training', 'evaluation', 'simulation', 'models', 'coldchain'):
        sys.path.insert(0, str(scripts / folder))
    from flax import nnx
    from i3_actor_draws import draw_single_step_sets, validated_actor_arrays
    from i3_counterfactual_teacher import ReplaySafeSaaReplanner, scenario_margin
    from i3_teacher_archive import load_archive
    from i3_pool_lineage import pool_lineage
    from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools, load_model,
        clock_support_mask, VOCAB_SIZE, check_model_generation)
    from scenario_saa import build_history, CondHistoricalSampler, make_c0_contract_v2
    from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
    from run_identity import dataset_identity, dataset_meta_sha256
    source_start = {p.relative_to(ext).as_posix(): sha(p) for p in scripts.rglob('*.py')}
    train = read_sealed(cfg['training_config'])
    t = train['teacher']
    archive = load_archive(t['path'], archive_sha256=t['sha256'],
        expected_context=t['context'], expected_keys=t['expected_keys'])
    context, cc = archive['context'], archive['context']['collection_config']
    expected_keys = [tuple(k) for k in cfg['evaluation_keys']]
    if cfg['qualification'] == 'teacher_objective_diagnostic':
        if (train['qualification'] != 'formal_development' or train['split_scope'] != 'teacher_objective_only' or
                sorted(expected_keys) != sorted(tuple(k) for k in t['expected_keys'] if k[0] in train['holdout_days']) or
                set(cfg['models']) != {'maskco', 'warmstart', 'reconstruction_control'} or
                cfg['K'] != 10 or cfg['independent_sets'] != 2):
            raise ValueError('Diagnostic differs from registered objective holdout/arms')
    if not expected_keys or len(set(expected_keys)) != len(expected_keys):
        raise ValueError('Nonempty unique diagnostic keys required')
    entries_by_key = {(e['day_id'], e['ordinal']): e for e in archive['entries']}
    entries = [entries_by_key[key] for key in expected_keys]
    for rel, digest in context['source_sha256'].items():
        if sha(scripts / rel) != digest:
            raise ValueError('Diagnostic/teacher source differs: ' + rel)
    history_data = generate_dataset(cc['history_days'], cc['orders'], train['train_seed'])
    if (dataset_identity(history_data) != context['history_dataset_sha256'] or
            dataset_meta_sha256(history_data) != context['history_dataset_meta_sha256']):
        raise ValueError('Diagnostic history identity mismatch')
    history = build_history(history_data)
    lineage = pool_lineage(history_data, train_seed=train['train_seed'],
        excluded_days=train.get('actor_pool_excluded_days', train['holdout_days']))
    pools = HistoryPools([history[i] for i in lineage['included_days']])
    contract = make_c0_contract_v2()
    if contract.contract_hash != context['contract_hash']:
        raise ValueError('Diagnostic contract differs from teacher')
    spec = train['model']
    cvrp = None
    if spec['arm'] in ('pretrained', 'random_cvrp'):
        if sha(spec['cvrp_path']) != spec['cvrp_sha256']:
            raise ValueError('CVRP identity changed')
        from mpre import load_cvrp_model
        cvrp, cvrp_config, _ = load_cvrp_model(spec['cvrp_path'])
        if spec['arm'] == 'random_cvrp':
            from dataclasses import replace
            cvrp = replace(cvrp_config, rngs=spec['initialization_seed']).construct_model()
    snapshots = {}
    samplers = {}
    for name, binding in cfg['models'].items():
        if sha(binding['path']) != binding['sha256'] or check_model_generation(binding['path']) != 'ok':
            raise ValueError('Diagnostic model changed/unknown: ' + name)
        card = read_sealed({'path': binding['card'], 'sha256': binding['card_sha256']})
        if cfg['qualification'] == 'teacher_objective_diagnostic':
            if name == 'warmstart' and binding['sha256'] != spec['warmstart_sha256']:
                raise ValueError('Diagnostic warmstart differs from registered initialization')
            if name != 'warmstart':
                expected_settings = dict(train['settings'])
                if name == 'reconstruction_control':
                    expected_settings['objective'] = 'reconstruction_control'
                if card['i3_training']['settings'] != expected_settings:
                    raise ValueError('Diagnostic checkpoint differs from fixed training choices')
        if name != 'warmstart':
            if (not card['training_complete'] or card['history_pool_lineage'] != lineage or
                    not card['i3_training']['source_stable'] or not card['i3_training']['checkpoint_reload_verified'] or
                    card['i3_training']['model_bin_sha256'] != binding['sha256']):
                raise ValueError('Unqualified diagnostic training card: ' + name)
            if cfg['qualification'] == 'teacher_objective_diagnostic' and (
                    card['qualification'] != 'formal_development' or
                    card['i3_training']['steps_completed'] != train['settings']['steps'] or
                    card['i3_training']['teacher_archive_sha256'] != t['sha256']):
                raise ValueError('Wrong formal training/teacher/steps')
        model = MaskCOScenarioModel(dim=spec['dim'], arm=spec['arm'], cvrp_model=cvrp,
                                   rngs=nnx.Rngs(spec['initialization_seed']))
        load_model(model, binding['path'])
        samplers[name] = MaskCOScenarioSampler(model, pools)
        snapshots[name] = dict(binding)

    def factory(day):
        excluded = set(cc.get('pool_excluded_days', [])) | {day}
        pool = [h for i, h in enumerate(history) if i not in excluded]
        return lambda: ReplaySafeSaaReplanner(budget=cc['budget'], capacity=cc.get('capacity', 50.),
            booking_horizon=BOOKING_HORIZON, contract=contract, cooling_share=cc.get('cooling_share', 1),
            sampler=CondHistoricalSampler(pool), reject_penalty={0: 5, 1: 10, 2: 15}, K=cc['K'],
            time_limit=float('inf'), arm_seed=7001, shadow_mode='greedy',
            energy_pricing='marginal', future_policy='density')

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=False)
    out.joinpath('submitted_config.json').write_bytes(Path(config_path).read_bytes())
    rows = []
    started = time.perf_counter()
    with out.joinpath('progress.jsonl').open('x', encoding='utf-8') as stream:
        for entry in entries:
            day, ordinal = entry['day_id'], entry['ordinal']
            preference, record = entry['preference'], entry['record']
            supported = bool(np.any(clock_support_mask(record.visible.clock) & (np.arange(VOCAB_SIZE) != 0)))
            row = {'day_id': day, 'ordinal': ordinal, 'future_support': supported,
                   **preference, 'arms': {}}
            sets_by_arm = {}
            if preference['decision_eligible'] and supported:
                for name, sampler in samplers.items():
                    packet = draw_single_step_sets(sampler, record.visible,
                        np.random.default_rng([cfg['sampling_seed'], day, ordinal]),
                        independent_sets=cfg['independent_sets'], K=cfg['K'])
                    sets_by_arm[name] = packet['scenarios']
                    validated_actor_arrays(sampler, packet)
                policy = factory(day)()
                hist_rng = np.random.default_rng([cfg['sampling_seed'], day, ordinal])
                sets_by_arm['cond_hist'] = [policy.sampler.sample(record.visible, hist_rng, cfg['K'])
                                            for _ in range(cfg['independent_sets'])]
            for name in list(samplers) + ['cond_hist']:
                margins = None
                if name in sets_by_arm:
                    margins = [scenario_margin(record, archive['dataset'], factory(day), contract,
                        BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, scenes)['margin'] for scenes in sets_by_arm[name]]
                row['arms'][name] = {'margins': margins, **decision_metrics(preference, margins,
                    tau=train['settings']['tau'], weight_cap=train['settings']['weight_cap'], supported=supported)}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print('DIAGNOSTIC_STATE_COMPLETE day=%s ordinal=%s' % (day, ordinal), flush=True)
    source_end = {p.relative_to(ext).as_posix(): sha(p) for p in scripts.rglob('*.py')}
    if source_start != source_end or sha(config_path) != config_sha or sha(t['path']) != t['sha256']:
        raise ValueError('Diagnostic source/config/archive changed during run')
    for binding in snapshots.values():
        if sha(binding['path']) != binding['sha256'] or sha(binding['card']) != binding['card_sha256']:
            raise ValueError('Diagnostic model/card changed during run')
    report = {**aggregate(rows, expected_keys, bootstrap_samples=cfg['bootstrap_samples'],
                         bootstrap_seed=cfg['bootstrap_seed']),
        'qualification': cfg['qualification'], 'config_sha256': config_sha,
        'model_identities': snapshots, 'history_pool_lineage': lineage,
        'teacher_archive_sha256': t['sha256'], 'source_stable': True,
        'source_start': source_start, 'source_end': source_end, 'rows': rows,
        'elapsed_s': time.perf_counter()-started, 'checkpoint_selection_changed': False,
        'online_gain_established': False, 'iteration_gain_established': False,
        'warmstart_seen_all_history_days': train['split_scope'] == 'teacher_objective_only'}
    out.joinpath('report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('TEACHER_DIAGNOSTIC_COMPLETED states=%s days=%s online_gain=0' % (len(rows), report['days']))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run(args.config, args.config_sha256, args.out)
