"""Actual 51-node compiled reconstruction barrier using synthetic engineering data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'delivery'))
from delivery_contract import DeliveryContract
from delivery_maskco import MaskCORouteProvider, improve_masked
from delivery_evaluator import evaluate


def main(checkpoint, upstream, out):
    contract = DeliveryContract()
    theta = np.arange(50) * (2 * np.pi / 50)
    data = {'coords': np.r_[np.array([[5., 5.]]),
                           np.stack([5+2*np.cos(theta), 5+2*np.sin(theta)], axis=1)].tolist(),
            'demands': [0.] + [1.] * 50, 'temp_class': [0] + [i % 3 for i in range(50)],
            'reveal_time': [0.] * 51, 'ready_time': [0.] * 51,
            'tw_start': [0.] * 51, 'tw_end': [8.] * 51,
            'service_time': [0.] + [.03] * 50}
    initial = [list(range(1+10*k, 11+10*k)) for k in range(5)]
    assert evaluate(data, initial, contract)['hard']
    reports = {}
    for label, random_seed, iterative in [('actual_single', None, False),
                                          ('actual_iterative', None, True),
                                          ('random_single', 42, False)]:
        provider = MaskCORouteProvider(checkpoint, upstream, random_seed=random_seed)
        provider.prewarm(51, contract.capacity)
        result = improve_masked(data, initial, contract, provider, time_limit_s=10.,
                                mask_fraction=.3, guide_weight=1., iterative=iterative, seed=41)
        assert result['decode_calls'] > 0 and result['completed_candidates'] > 0, label
        assert result['hard_feasible_candidates'] > 0 and result['report']['hard'], label
        assert result['retained_edges_preserved'] and not result['timeout'], label
        assert evaluate(data, result['routes'], contract)['hard'], label
        reports[label] = {'provider_identity': provider.identity,
                          'wall_s': result['wall_s'], 'decode_calls': result['decode_calls'],
                          'completed_candidates': result['completed_candidates'],
                          'hard_feasible_candidates': result['hard_feasible_candidates'],
                          'timeout': result['timeout'], 'retained_edges_preserved': True}
    Path(out).write_text(json.dumps({'actual_runtime_activation_pass': True,
        'fixed_synthetic_engineering_only': True, 'development_data_not_used': True,
        'online_gain_established': False, 'arms': reports}, indent=2))
    print('PASS: actual/random compiled 51-node reconstruction and unchanged 10-second total budget')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['checkpoint', 'upstream', 'out']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    main(a.checkpoint, a.upstream, a.out)
