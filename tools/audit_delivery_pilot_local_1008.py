"""Independent local replay/paired recomputation of the fixed delivery pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'C-VRP_Cold-chainVehicleRoutingProblem'
sys.path.insert(0, str(ROOT / 'scripts/delivery'))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def main(run_name='delivery_pilot_1008_v1'):
    if run_name not in ['delivery_pilot_1008_v1', 'delivery_pilot_1008_v2']:
        raise ValueError('Unknown registered run')
    root = ROOT / 'results' / run_name
    reg = json.loads((root / 'protocol.json').read_text())
    result = json.loads((root / 'pilot/result.json').read_text())
    original = json.loads((root / 'PILOT_ADJUDICATION.json').read_text())
    identity = result['identity']
    assert identity == json.loads((root / 'pilot/identity_start.json').read_text())
    assert identity == json.loads((root / 'pilot/identity_end.json').read_text())
    assert identity['protocol_sha256'] == hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    assert identity['source_sha256'] == digest(manifest)
    checked_local_sources = 0
    for key, expected in manifest.items():
        prefix = 'C-VRP_Cold-chainVehicleRoutingProblem/'
        if not key.startswith(prefix):
            continue
        with zipfile.ZipFile(ROOT / ('tools_remote/' + run_name + '.zip')) as archive:
            raw = archive.read(key[len(prefix):])
        assert hashlib.sha256(raw).hexdigest() == expected, key
        checked_local_sources += 1
    contract = DeliveryContract(**reg['contract_overrides'])
    assert contract.identity() == identity['contract_hash']
    assert digest(reg['timing']) == identity['timing_hash']
    assert reg['weights']['sha256'] == identity['checkpoint_sha256']
    rows = result['rows']
    assert [r['instance_id'] for r in rows] == list(range(32))
    costs = {arm: [] for arm in reg['arms']}
    times = {arm: [] for arm in reg['arms']}
    cycles = {arm: [] for arm in reg['arms'] if arm.startswith('maskco')}
    timeouts = {arm: 0 for arm in reg['arms']}
    dataset = {}
    replayed = 0
    for row in rows:
        i = row['instance_id']
        rng = np.random.default_rng(np.random.SeedSequence([reg['dataset']['seed'], i]))
        n = reg['dataset']['customers']
        data = {'coords': np.r_[np.array([[5., 5.]]), rng.uniform(0, 10, (n, 2))].tolist(),
                'demands': np.r_[0, rng.integers(1, 6, n)].tolist(),
                'temp_class': np.r_[0, rng.integers(0, 3, n)].tolist(),
                'reveal_time': [0.] * (n + 1), 'ready_time': [0.] * (n + 1),
                'tw_start': [0.] * (n + 1), 'tw_end': [8.] * (n + 1),
                'service_time': [0.] + [.03] * n}
        initial = [list(range(1 + 10*k, 11 + 10*k)) for k in range(5)]
        expected_hash = digest([data, initial])
        assert row['data_initial_hash'] == expected_hash and row['identity'] == identity
        dataset[str(i)] = {'instance': data, 'initial_routes': initial,
                           'data_initial_hash': expected_hash}
        assert set(row['arms']) == set(reg['arms'])
        for arm in reg['arms']:
            record = row['arms'][arm]
            replay = evaluate(data, record['routes'], contract)
            assert replay['hard'] and replay['complete'] and len(replay['delivered']) == 50
            assert all(not t['terminal_manifest'] for t in replay['traces'])
            for field in ['cost', 'distance_km', 'energy_kwh', 'quality_loss_value']:
                assert np.isfinite(replay[field])
                assert abs(replay[field] - record['report'][field]) <= 1e-8
            for trace in replay['traces']:
                for event in trace['events']:
                    if event['kind'] == 'deliver':
                        assert event['load_after'] < event['load_before']
            wall = record['wall_s']
            assert np.isfinite(wall) and wall >= 0
            if wall >= reg['timing']['total_per_arm_per_instance_s']:
                assert record['timeout'] and record['routes'] == initial
            costs[arm].append(replay['cost'])
            times[arm].append(wall)
            timeouts[arm] += int(record['timeout'])
            if arm in cycles:
                cycles[arm].append(record['cycles'])
            replayed += 1
    assert digest(dataset) == identity['data_sha256']
    assert dataset == json.loads((root / 'pilot/dataset.json').read_text())
    rng = np.random.default_rng(reg['bootstrap']['seed'])
    indices = rng.integers(0, 32, (reg['bootstrap']['repetitions'], 32))
    comparisons = {}
    b = np.asarray(costs[reg['primary_method']])
    for control in reg['strong_controls'] + [reg['pretraining_control']]:
        a = np.asarray(costs[control])
        differences = (a-b)/a
        ci = np.quantile(differences[indices].mean(axis=1), [.025, .975])
        comparison = {'absolute_control_minus_single_mean': float((a-b).mean()),
                      'fractional_gain_mean': float(differences.mean()),
                      'fractional_gain_ci95': ci.tolist()}
        expected = original['comparisons'][control]
        for key in comparison:
            assert np.allclose(comparison[key], expected[key], rtol=0, atol=1e-12)
        comparisons[control] = comparison
    model_active = all(any(row['arms'][arm].get('decode_calls', row['arms'][arm]['cycles']) > 0
                           for row in rows) for arm in cycles)
    pilot_pass = all(comparisons[control]['fractional_gain_mean'] >=
                     (reg['practical_threshold'] if control in reg['strong_controls'] else 0.)
                     and comparisons[control]['fractional_gain_ci95'][0] > 0
                     for control in comparisons) and model_active
    iteration = np.asarray(costs['maskco_single']) - np.asarray(costs['maskco_iterative'])
    iteration_ci = np.quantile(iteration[indices].mean(axis=1), [.025, .975])
    assert abs(float(iteration.mean()) - original['same_weight_iteration_absolute_gain_mean']) <= 1e-12
    assert np.allclose(iteration_ci, original['same_weight_iteration_absolute_gain_ci95'], rtol=0, atol=1e-12)
    audit = {'unique_instances': 32, 'arm_instance_trajectories_replayed': replayed,
             'all_hard_and_complete': True, 'actual_local_sealed_sources_matched': checked_local_sources,
             'dataset_regenerated_and_byte_identity_matched': True,
             'source_manifest_and_contract_timing_identity_matched': True,
             'paired_bootstrap_matches_runner': True, 'mean_cost': {k: float(np.mean(v)) for k, v in costs.items()},
             'timeout_counts': timeouts, 'comparisons': comparisons,
             'cycles_per_instance': cycles,
             'model_arms_all_zero_cycles': all(not any(v) for v in cycles.values()),
             'mean_wall_seconds': {k: float(np.mean(v)) for k, v in times.items()},
             'operational_records_complete': True,
             'independent_method_comparison_qualified': model_active,
             'qualification_note': ('Active model comparison replayed; frozen-source/weight end hashes must also be checked independently on the server'
                                    if model_active else 'All MaskCO arms executed zero reconstruction cycles; engineering activation diagnosis is required'),
             'same_weight_iteration_absolute_gain_mean': float(iteration.mean()),
             'same_weight_iteration_absolute_gain_ci95': iteration_ci.tolist(),
             'pilot_pass': bool(pilot_pass), 'online_gain_established': False}
    (root / 'LOCAL_INDEPENDENT_REPLAY_AUDIT.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'cycles_per_instance'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default='delivery_pilot_1008_v1')
    main(p.parse_args().run_name)
