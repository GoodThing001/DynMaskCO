"""Independent raw-day I3 adjudication; engineering smoke cannot become formal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(a, b, seed, days):
    if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
        raise ValueError('Positive expected day count is required')
    for rows in (a, b):
        if (len(rows) != days or
                any(not isinstance(r['i'], int) or isinstance(r['i'], bool) for r in rows) or
                sorted(r['i'] for r in rows) != list(range(days)) or
                any(r['utility'] is None or not np.isfinite(r['utility']) for r in rows)):
            raise ValueError('Missing/duplicate/nonfinite days; no filtering')
    aa, bb = {r['i']: r for r in a}, {r['i']: r for r in b}
    diff = np.array([aa[d]['utility'] - bb[d]['utility'] for d in range(days)])
    rng = np.random.default_rng(seed + 777)
    ci = np.quantile(diff[rng.integers(0, days, size=(2000, days))].mean(axis=1), [.025, .975])
    return {'mean': float(diff.mean()), 'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
            'n_paired': days, 'both_hard': all(r['hard_feasible'] is True for r in a + b)}


def adjudicate(gate, online_config, config_sha, protocol, protocol_sha):
    if sha(online_config) != config_sha or sha(protocol) != protocol_sha:
        raise ValueError('External online/protocol identity mismatch')
    cfg = json.loads(Path(online_config).read_text(encoding='utf-8'))
    reg = json.loads(Path(protocol).read_text(encoding='utf-8'))
    doc = json.loads(Path(gate).read_text(encoding='utf-8'))
    if cfg['qualification'] not in ('engineering', 'formal_development'):
        raise ValueError('Unknown experiment qualification')
    formal = cfg['qualification'] == 'formal_development'
    problems = []
    if not formal:
        problems.append('engineering_qualification')
    if reg['bootstrap'] != {'samples': 2000, 'seed_offset': 777, 'unit': 'day'}:
        raise ValueError('Unsupported registered bootstrap')
    seeds = doc['seeds']['ran']
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or
            any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds) or
            set(doc['results']) != {str(seed) for seed in seeds}):
        raise ValueError('Missing/duplicate development seeds or inconsistent result blocks')
    if set(cfg['models']) != {'maskco', 'warmstart', 'reconstruction_control'}:
        raise ValueError('Missing registered model/control identities')
    if formal:
        if (reg.get('predeclared') is not True or reg['main_comparison'] != 'I3_minus_cond_hist' or
                reg['delta_3'] != 20 or reg['online_days'] != 40 or
                cfg['online_parameters'] != reg['online_parameters']):
            raise ValueError('Formal problem or criterion differs from registered I3 protocol')
        if seeds != [cfg['expected_identity']['seeds']['gate']]:
            raise ValueError('Result seed differs from externally sealed expected data identity')
        exit_file = Path(gate).parent / 'run.exit'
        if not exit_file.exists() or 'RUN_EXIT=0' not in exit_file.read_text(encoding='utf-8').splitlines():
            problems.append('missing_successful_run_exit')
    for name, spec in cfg['models'].items():
        if sha(spec['path']) != spec['sha256'] or sha(spec['card']) != spec['card_sha256']:
            problems.append('model_or_card_changed:' + name)
            continue
        card = json.loads(Path(spec['card']).read_text(encoding='utf-8'))
        if formal and name != 'warmstart':
            if (card.get('qualification') != 'formal_development' or not card.get('training_complete') or
                    not card['i3_training']['source_stable'] or not card['i3_training']['checkpoint_reload_verified'] or
                    card['i3_training']['model_bin_sha256'] != spec['sha256'] or
                    card['history_pool_lineage'] != cfg['history_pool_lineage']):
                problems.append('unqualified_training_card:' + name)
    if doc['model_adapter']['config_sha256'] != config_sha:
        problems.append('online_config_identity_mismatch')
    for key, value in cfg['online_parameters'].items():
        if doc['config'].get(key) != value:
            problems.append('online_parameter_mismatch:' + key)
    stats = {}
    for seed in seeds:
        if seed not in (20260926, 20260930, 20261001):
            problems.append('unregistered_development_seed')
        block = doc['results'][str(seed)]
        ident = block['_identity']
        if not isinstance(ident['source_sha256'], dict) or not ident['source_sha256']:
            problems.append('missing_source_identity')
        if ident['source_stable'] is not True or ident['source_sha256'] != ident['source_end_sha256']:
            problems.append('source_changed')
        if ident['model_adapter'] != doc['model_adapter']:
            problems.append('model_adapter_changed')
        if formal:
            expected = cfg['expected_identity']
            for key, want in expected.items():
                if ident.get(key) != want:
                    problems.append('identity_mismatch:' + key)
            ext = Path(cfg['frozen_extension']).resolve()
            for rel, want in ident['source_sha256'].items():
                file = ext / rel
                if not file.resolve().is_relative_to(ext) or not file.exists() or sha(file) != want:
                    problems.append('source_file_changed:' + rel)
        tier = block['p_c=(5,10,15)']
        rows = tier['per_day']
        days = cfg['online_parameters']['dev_instances']
        if set(rows) != {'maskco', 'cond_hist', 'warmstart', 'reconstruction_control'}:
            raise ValueError('Missing fresh method/control arms')
        comparisons = {name: paired(rows['maskco'], rows[name], seed, days)
                       for name in ('cond_hist', 'warmstart', 'reconstruction_control')}
        primary = comparisons['cond_hist']
        qualified = bool(formal and not problems and days == reg['online_days'] and
                         tier['main_comparison']['formal_adjudication'] is True and primary['both_hard'])
        primary_pass = primary['mean'] >= reg['delta_3'] and primary['ci_lo'] > 0
        mechanism = comparisons['reconstruction_control']
        stats[str(seed)] = {'comparisons': comparisons, 'formal_adjudication': qualified,
            'numeric_passed': bool(primary_pass), 'formal_passed': bool(qualified and primary_pass),
            'decision_objective_attribution_passed': bool(qualified and mechanism['both_hard'] and
                mechanism['mean'] > 0 and mechanism['ci_lo'] > 0),
            'mean_timeouts': {name: float(np.mean([r['timeouts'] for r in rr])) for name, rr in rows.items()}}
    if problems:
        for value in stats.values():
            value['formal_adjudication'] = False
            value['formal_passed'] = False
            value['decision_objective_attribution_passed'] = False
    return {'formal_adjudicable': bool(formal and stats and not problems and all(v['formal_adjudication'] for v in stats.values())),
        'problems': problems, 'by_seed': stats, 'gate_sha256': sha(gate),
        'online_config_sha256': config_sha, 'protocol_sha256': protocol_sha,
        'bootstrap': reg['bootstrap'], 'iteration_gain_established': False,
        'note': 'Primary I3-cond; reconstruction control isolates training objective. No day or timeout filtering.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--online-config', required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = adjudicate(args.gate, args.online_config, args.config_sha256, args.protocol, args.protocol_sha256)
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'formal_adjudicable': report['formal_adjudicable'], 'by_seed': report['by_seed']}))
