"""Strict saved-input test of a new shape-split action head candidate.

The fixed 750-row first layer is sliced to the original action count before
the final output layer. No backbone, decoder, solver, training, or selection.
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
ROUND = ROOT / 'research/review_3080_20261010/goal_round_20261013_r12_learned_proposal_and_split_head'
sys.path.insert(0, str(OLD))
from delivery_actual_action_head_v1 import construct_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hidden(head, latent, binding):
    h = jnp.asarray(latent[0], jnp.float32)
    actions = jnp.asarray(binding['actions'], jnp.int32)
    parts = [h[actions[:, 0]], h[actions[:, 3]], h[actions[:, 4]]]
    parts.extend(jnp.asarray(binding[k], jnp.float32) @ h for k in ('route_weights', 'prefix_weights', 'suffix_weights'))
    parts.append(jnp.asarray(binding['slot_position_length'], jnp.float32))
    return jnp.tanh(head.input(jnp.concatenate(parts, axis=-1)))


def split_score(head, latent, binding):
    n = len(binding['actions'])
    if n > 750 or n < 1:
        raise ValueError('registered fixed action shape')
    padded = dict(binding)
    for key in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length'):
        value = binding[key]
        padded[key] = np.concatenate([value, np.repeat(value[:1], 750 - n, axis=0)], axis=0)
    prefinal = hidden(head, latent, padded)[:n]
    return np.asarray(head.output(prefinal)[:, 0])


def main():
    psutil.Process().cpu_affinity([7])
    reg = json.loads((ROUND / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['source_sha256']
    assert sha(OLD / 'delivery_actual_action_head_v1.py') == reg['head_source_sha256']
    assert reg['fixed_count'] == 750 and reg['head_seed'] == 20261012
    assert list(reg['inputs']) == [f'{arm}/request/{step:02d}.npz' for arm in ('pretrained_initial_fault0_worker', 'random888_initial_fault0_worker') for step in range(17)]
    head = construct_head(seed=20261012)
    rows = []
    for name, expected in reg['inputs'].items():
        path = OLD / name
        assert sha(path) == expected
        with np.load(path, allow_pickle=False) as z:
            latent = z['latent'].copy()
            binding = {k: z[k].copy() for k in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length', 'context', 'timestep')}
        direct = np.asarray(head(jnp.asarray(latent), binding))
        split = split_score(head, latent, binding)
        assert direct.shape == split.shape and direct.dtype == split.dtype == np.float32
        mismatch = direct.view(np.uint32) != split.view(np.uint32)
        rows.append({'input': name, 'input_sha256': expected, 'actions': len(direct), 'different_elements': int(mismatch.sum()),
                     'max_abs_diff': float(np.max(np.abs(direct.astype(np.float64) - split.astype(np.float64)))),
                     'direct_sha256': hashlib.sha256(direct.tobytes()).hexdigest(),
                     'split_sha256': hashlib.sha256(split.tobytes()).hexdigest()})
    result = {'scope': 'new split-before-final shape candidate on 34 fixed saved inputs only', 'source_sha256': sha(__file__),
              'registration_sha256': sha(ROUND / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json'), 'jax_version': jax.__version__,
              'rows': rows, 'strict_equal': all(r['different_elements'] == 0 for r in rows),
              'total_different_elements': sum(r['different_elements'] for r in rows),
              'backbone_executed': False, 'full_request_budget_qualified': False, 'stable_online_gain': False}
    with (ROUND / 'DELIVERY_SPLIT_HEAD_RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'inputs': len(rows), 'strict_equal': result['strict_equal'], 'different_elements': result['total_different_elements']}))


if __name__ == '__main__':
    main()
