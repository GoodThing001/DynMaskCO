"""Separate-process check of the new final-layer-only padding diagnosis."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import jax.numpy as jnp
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
ROUND = ROOT / 'research/review_3080_20261010/goal_round_20261013_r11_fault3_head_arithmetic'
sys.path.insert(0, str(OLD))
from delivery_actual_action_head_v1 import construct_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    psutil.Process().cpu_affinity([7])
    reg = json.loads((ROUND / 'DELIVERY_LAYER_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'))
    prod = json.loads((ROUND / 'DELIVERY_LAYER_DIAGNOSIS.json').read_text(encoding='utf-8'))
    source = OLD / 'pretrained_initial_fault0_worker/request/13.npz'
    checks = {}
    checks['registered_data'] = sha(source) == reg['input_sha256']
    checks['registered_producer'] = sha(ROUND / 'DELIVERY_LAYER_DIAGNOSIS.json') == reg['producer_sha256']
    checks['registered_head'] = sha(OLD / 'delivery_actual_action_head_v1.py') == reg['head_source_sha256']
    with np.load(source, allow_pickle=False) as z:
        embeddings = jnp.asarray(z['latent'][0], jnp.float32)
        actions = z['actions'].copy()
        weights = [z[k].copy() for k in ('route_weights', 'prefix_weights', 'suffix_weights')]
        scalar = z['slot_position_length'].copy()
    head = construct_head(seed=20261012)
    def make(m):
        a = jnp.asarray(actions)
        x = [embeddings[a[:, col]] for col in (0, 3, 4)]
        x += [jnp.asarray(w if m == len(a) else np.concatenate((w, np.repeat(w[:1], m-len(a), axis=0)), axis=0)) @ embeddings for w in weights]
        x.append(jnp.asarray(scalar if m == len(a) else np.concatenate((scalar, np.repeat(scalar[:1], m-len(a), axis=0)), axis=0)))
        # The copied action rows leave gather features identical; only the
        # matrix batch dimension and final dense batch dimension vary.
        if m != len(a):
            a2 = jnp.asarray(np.concatenate((actions, np.repeat(actions[:1], m-len(a), axis=0)), axis=0))
            x[:3] = [embeddings[a2[:, col]] for col in (0, 3, 4)]
        features = jnp.concatenate(x, axis=-1)
        pre = head.input(features)
        act = jnp.tanh(pre)
        out = head.output(act)[:, 0]
        return [np.asarray(z)[:len(actions)] for z in (features, pre, act, out)]
    direct, padded = make(len(actions)), make(750)
    names = ('current_features', 'input_linear', 'activation', 'output_score')
    for key, a, b in zip(names, direct, padded):
        recorded = prod['stages'][key]
        checks[key + '_direct_digest'] = hashlib.sha256(a.tobytes()).hexdigest() == recorded['direct_sha256']
        checks[key + '_padded_digest'] = hashlib.sha256(b.tobytes()).hexdigest() == recorded['padded_sha256']
        checks[key + '_bitwise_count'] = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))) == recorded['different_elements']
    checks['strict_gate_still_false'] = prod['strict_candidate_qualified'] is False and prod['stages']['output_score']['different_elements'] == 3
    result = {'checks': checks, 'failed': [k for k, ok in checks.items() if not ok],
              'final_layer_batch_shape_is_first_observed_divergence': all(prod['stages'][n]['different_elements'] == 0 for n in names[:-1]),
              'strict_candidate_qualified': False, 'stable_online_gain': False}
    with (ROUND / 'INDEPENDENT_DELIVERY_LAYER_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'checks': len(checks), 'failed': result['failed']}))
    return int(bool(result['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
