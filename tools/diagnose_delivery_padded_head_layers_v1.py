"""Layer-local arithmetic diagnosis on the one frozen step-13 head input.

No backbone, decoder, C0, training, old compiled-head replay, or selection.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import jax
import jax.numpy as jnp
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
ROUND = ROOT / 'research/review_3080_20261010/goal_round_20261013_r11_fault3_head_arithmetic'
INPUT = OLD / 'pretrained_initial_fault0_worker/request/13.npz'
sys.path.insert(0, str(OLD))
from delivery_actual_action_head_v1 import construct_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(head, latent, binding):
    h = jnp.asarray(latent[0], jnp.float32)
    a = jnp.asarray(binding['actions'], jnp.int32)
    parts = [h[a[:, 0]], h[a[:, 3]], h[a[:, 4]]]
    parts.extend(jnp.asarray(binding[k], jnp.float32) @ h for k in ('route_weights', 'prefix_weights', 'suffix_weights'))
    parts.append(jnp.asarray(binding['slot_position_length'], jnp.float32))
    x = jnp.concatenate(parts, axis=-1)
    linear = head.input(x)
    activation = jnp.tanh(linear)
    score = head.output(activation)[:, 0]
    return {k: np.asarray(v) for k, v in [('current_features', x), ('input_linear', linear), ('activation', activation), ('output_score', score)]}


def main():
    psutil.Process().cpu_affinity([7])
    reg = json.loads((ROUND / 'DELIVERY_LAYER_REGISTRATION.json').read_text(encoding='utf-8'))
    prior = json.loads((ROOT / 'research/review_3080_20261010/goal_round_20261013_r10_continuous_strong_delivery/DELIVERY_PADDED_HEAD_LOCAL_RESULT.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['program_sha256'] and sha(INPUT) == reg['input_sha256']
    assert sha(OLD / 'delivery_actual_action_head_v1.py') == reg['head_source_sha256']
    with np.load(INPUT, allow_pickle=False) as z:
        latent = z['latent'].copy()
        direct = {k: z[k].copy() for k in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length', 'context', 'timestep')}
    n = len(direct['actions'])
    padded = {k: v.copy() for k, v in direct.items()}
    for k in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length'):
        padded[k] = np.concatenate([direct[k], np.repeat(direct[k][:1], 750 - n, axis=0)], axis=0)
    head = construct_head(seed=20261012)
    left, right = stage(head, latent, direct), stage(head, latent, padded)
    result = {'scope': 'one new layer-local diagnosis of prior strict failure; no candidate selection', 'actual_jax': jax.__version__,
              'input_sha256': sha(INPUT), 'original_count': n, 'fixed_count': 750, 'stages': {},
              'strict_candidate_qualified': False, 'full_request_budget_qualified': False, 'stable_online_gain': False}
    for k in left:
        x, y = left[k], right[k][:n]
        assert x.shape == y.shape and x.dtype == y.dtype == np.float32
        mismatch = x.view(np.uint32) != y.view(np.uint32)
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        result['stages'][k] = {'shape': list(x.shape), 'different_elements': int(mismatch.sum()),
                               'different_rows': int(mismatch.reshape(n, -1).any(axis=1).sum()),
                               'max_abs_diff': float(diff.max()),
                               'direct_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                               'padded_sha256': hashlib.sha256(y.tobytes()).hexdigest()}
    final = result['stages']['output_score']
    assert n == 516 and final['different_elements'] == 3
    assert final['direct_sha256'] == prior['direct_sha256'] and final['padded_sha256'] == prior['fixed_sha256']
    with (ROUND / 'DELIVERY_LAYER_DIAGNOSIS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v['different_elements'] for k, v in result['stages'].items()}))


if __name__ == '__main__':
    main()
