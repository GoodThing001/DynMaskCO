# -*- coding: utf-8 -*-
"""Fail-closed tests for independent A3 adjudication and baseline override."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from diag_a3_adjudicate import ARMS, SEEDS, TIERS, main  # noqa: E402


def _identity(seed, model=None, shared='a' * 64):
    d = {'source_stable': True, 'source_sha256': {'s': 'b' * 64},
         'source_end_sha256': {'s': 'b' * 64},
         'shared_source_sha256': {'s': shared},
         'contract_sha256': 'c' * 64, 'data_sha256': 'd' * 64,
         'data_meta_sha256': 'e' * 64, 'budget_B': 100.0,
         'cooling_share': 4.0,
         'shared_config': {'time_limit': 10.0, 'capacity': 50.0,
                           'num_vehicles': 15, 'n_orders': 200},
         'seeds': {'train': 20260925, 'gate': int(seed)}}
    if model is not None:
        d['model_bin'] = str(model)
        d['model_bin_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
    return d


def _rows(offset=0.0):
    return [{'i': i, 'utility': float(100 + i + offset),
             'hard_feasible': True} for i in range(40)]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def _fixture(root):
    root = Path(root)
    lock = root / 'lock.json'
    _write(lock, {'baseline_policy': 'density'})
    base_map = {}
    dirs = []
    for seed in SEEDS:
        bp = root / ('baseline_%s' % seed) / 'gate.json'
        _write(bp, {'config': {'train_instances': 200, 'gate_instances': 40,
                              'gate_seed': int(seed), 'n_orders': 200,
                              'capacity': 50.0, 'num_vehicles': 15, 'rho': 0.6,
                              'K': 10, 'time_limit': 10.0, 'k_neighbors': 10,
                              'future_policy': 'density', 'shadow_mode': 'greedy',
                              'energy_pricing': 'marginal'},
                    'seeds': {'gate': int(seed)},
                    'identity': _identity(seed),
                    'gate': {tier: {'per_day': {'cond_hist': _rows()}}
                             for tier in TIERS}})
        base_map[seed] = str(bp)
        (bp.parent / 'run.exit').write_text(
            'RUN_EXIT=0\nSEAL_EXIT=0\nDONE=1\n', encoding='utf-8')
        for arm in ARMS:
            d = root / ('a1_a3_online_density_%s_%s_40_test' % (arm, seed))
            model = d / 'model.bin'
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes((arm + seed).encode('ascii'))
            (d / 'run.exit').write_text('RUN_EXIT=0\n', encoding='utf-8')
            ident = _identity(seed, model)
            offset = {'B': 30.0, 'C': 10.0, 'D': -10.0}[arm]
            doc = {'config': {'dev_instances': 40, 'train_instances': 200,
                              'gate_seed': int(seed), 'K': 10, 'time_limit': 10.0,
                              'future_policy': 'density', 'shadow_mode': 'greedy',
                              'energy_pricing': 'marginal',
                              'arm': 'random_cvrp' if arm == 'D' else 'pretrained',
                              'iterative': arm == 'C', 'rounds': 1,
                              'remask_frac': 0.5,
                              'baseline_from': str(bp)},
                   'seeds': {'ran': [int(seed)]},
                   'model_generation_check': 'ok',
                   'results': {seed: {'_identity': ident, **{
                       tier: {'per_day': {'maskco': _rows(offset)},
                              'arms': {'maskco': {'mean_timeouts': 0.0}}}
                       for tier in TIERS}}}}
            _write(d / 'gate.json', doc)
            dirs.append(str(d))
    mp = root / 'baseline_map.json'
    _write(mp, {'gate_json_by_seed': base_map})
    return dirs, lock, mp


def _run(root, dirs, lock, baseline_map=None):
    argv = dirs + ['--lock', str(lock), '--out', str(Path(root) / 'out.json')]
    if baseline_map is not None:
        argv += ['--baseline-map', str(baseline_map)]
    return main(argv)


def test_valid_then_fail_closed():
    with tempfile.TemporaryDirectory() as tmp:
        dirs, lock, mp = _fixture(tmp)
        good = _run(tmp, dirs, lock, mp)
        assert good['formal_adjudicable'] and not good['partial']
        b = good['summary_by_arm_seed']['B']['20260926']['main_tier']
        assert b['formal_passed'] and b['mean'] == 30.0
        c = good['summary_by_arm_seed']['C']['20260926']['main_tier']
        assert c['formal_adjudication'] and not c['formal_passed']
        assert good['ablations'][str(('20260926', TIERS[1], 'C_minus_B'))][
            'interpretation'] == 'training_and_inference_confounded'

        # The old baseline cannot be silently accepted after an override.
        seed = SEEDS[0]
        old = Path(tmp) / 'old_baseline' / 'gate.json'
        base = json.loads(Path(json.loads(mp.read_text())[
            'gate_json_by_seed'][seed]).read_text())
        base['identity']['shared_source_sha256'] = {'s': 'f' * 64}
        _write(old, base)
        d = Path(dirs[0]) / 'gate.json'
        doc = json.loads(d.read_text())
        doc['config']['baseline_from'] = str(old)
        _write(d, doc)
        bad = _run(tmp, dirs, lock)
        assert not bad['formal_adjudicable']
        assert 'shared_source_sha256_mismatch' in bad['summary_by_arm_seed'][
            'B'][seed]['main_tier']['problems']
        assert _run(tmp, dirs, lock, mp)['formal_adjudicable']

        # A duplicate ID, including 40/40 rows, is not an admissible paired day set.
        doc['results'][seed][TIERS[1]]['per_day']['maskco'][39]['i'] = 38
        _write(d, doc)
        dup = _run(tmp, dirs, lock, mp)
        assert not dup['formal_adjudicable']
        assert not dup['summary_by_arm_seed']['B'][seed]['main_tier']['adjudicable']

        # A missing batch cannot pass simply because directory count is eight.
        partial = _run(tmp, dirs[:-1], lock, mp)
        assert partial['partial'] and not partial['formal_adjudicable']
    print('A3 formal adjudicator fail-closed tests PASS')


def test_baseline_and_model_failures():
    with tempfile.TemporaryDirectory() as tmp:
        dirs, lock, mp = _fixture(tmp)
        seed = SEEDS[0]
        bp = Path(json.loads(mp.read_text())['gate_json_by_seed'][seed])
        original = json.loads(bp.read_text())
        checks = []
        d = json.loads(bp.read_text())
        for tier in TIERS:
            d['gate'][tier]['per_day']['cond_hist'].pop()
        checks.append(d)  # Shared missing day: never accept 39/39 intersection.
        d = json.loads(bp.read_text())
        d['seeds']['gate'] = 20260930
        checks.append(d)
        d = json.loads(bp.read_text())
        del d['gate'][TIERS[1]]
        checks.append(d)
        d = json.loads(bp.read_text())
        d['config']['future_policy'] = 'reveal'
        checks.append(d)
        for bad_base in checks:
            _write(bp, bad_base)
            assert not _run(tmp, dirs, lock, mp)['formal_adjudicable']
        _write(bp, original)
        seal = bp.parent / 'run.exit'
        seal.write_text('RUN_EXIT=0\nSEAL_EXIT=1\nDONE=0\n', encoding='utf-8')
        assert not _run(tmp, dirs, lock, mp)['formal_adjudicable']
        seal.write_text('RUN_EXIT=0\nSEAL_EXIT=0\nDONE=1\n', encoding='utf-8')
        assert not _run(tmp, dirs + [dirs[0]], lock, mp)['formal_adjudicable']
        model = Path(dirs[0]) / 'model.bin'
        model.write_bytes(b'tampered')
        bad = _run(tmp, dirs, lock, mp)
        assert not bad['formal_adjudicable']
        assert 'model_hash_mismatch' in bad['per_batch'][Path(dirs[0]).name]['problems']
        assert not bad['ablations'][str((seed, TIERS[1], 'B_minus_D'))][
            'formal_adjudication']
    print('A3 baseline configuration/model failure tests PASS')


if __name__ == '__main__':
    test_valid_then_fail_closed()
    test_baseline_and_model_failures()
