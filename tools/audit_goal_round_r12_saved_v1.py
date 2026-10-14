"""Fresh read-only audit of r12 pickup returns and delivery saved-head candidate."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'research/review_3080_20261010/goal_round_20261013_r12_learned_proposal_and_split_head'
PICKUP = ROOT / 'research/review_3080_20261010/pickup'
DELIVERY = ROOT / 'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
sys.path.insert(0, str(DELIVERY))
from delivery_actual_action_head_v1 import construct_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(value):
    if isinstance(value, dict) and '__mapping__' in value:
        return {unpack(k): unpack(v) for k, v in value['__mapping__']}
    if isinstance(value, dict) and '__sequence__' in value:
        return [unpack(v) for v in value['values']]
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def verify_pickup():
    v1 = unpack(json.loads((PICKUP / 'learned_after_strong_root_v1/RESULT.json').read_text(encoding='utf-8')))
    v2 = unpack(json.loads((PICKUP / 'learned_after_strong_recovery_execution_root_v2/RESULT.json').read_text(encoding='utf-8')))
    reg = json.loads((ROUND / 'PICKUP_RECOVERY_REGISTRATION.json').read_text(encoding='utf-8'))
    checks = 0
    for name, digest in reg['source_sha256'].items():
        assert sha(ROOT / name) == digest
        checks += 1
    assert [r['mode'] for r in v1['records']] == [r['mode'] for r in v2['records']] == ['pretrained', 'random']; checks += 1
    assert all(not r['parent_on_time'] and not r['actual']['qualified'] for r in v1['records']); checks += 1
    assert all('unexpected keyword argument \'anytime_vote\'' in ''.join(r['stderr']) for r in v1['records']); checks += 1
    candidates = []
    params = []
    for r in v2['records']:
        a = r['actual']; ready = r['ready']; cs = a['common_scores']; proposal = a['proposal']
        assert r['parent_on_time'] and r['worker_reaped'] and not r['stderr'] and 0 < r['wall_s'] < 10; checks += 1
        assert a['qualified'] and not a['timeout'] and a['mode'] == r['mode']; checks += 1
        assert a['source_end'] == ready['source_start'] and a['checkpoint_end_sha256'] == ready['checkpoint_sha256']; checks += 1
        assert proposal['encoder_calls'] == proposal['decoder_calls'] == 1 and proposal['complete']; checks += 1
        assert cs['sampling_once'] and cs['candidate_scores']['own_incumbent']['complete_scenarios'] == 10 and cs['candidate_scores']['actual_candidate']['complete_scenarios'] == 10; checks += 1
        assert a['selected_certificate']['qualified'] and a['accepted_on_common_SAA']; checks += 1
        assert cs['candidate_scores']['actual_candidate']['mean_shadow_utility'] > cs['candidate_scores']['own_incumbent']['mean_shadow_utility']; checks += 1
        assert a['raw_candidate'] == a['selected_plan']; checks += 1
        candidates.append(a['raw_candidate']); params.append(ready['provider_identity']['actual_param_hash'])
    assert candidates[0] == candidates[1] and params[0] != params[1]; checks += 1
    assert v2['records'][0]['actual']['common_scores']['scenario_sha256'] == v2['records'][1]['actual']['common_scores']['scenario_sha256']; checks += 1
    return {'checks': checks, 'failures': 0, 'same_raw_candidate': True, 'distinct_model_parameters': True,
            'model_learning_attribution': False, 'both_request_seconds': [r['wall_s'] for r in v2['records']]}


def verify_delivery():
    reg = json.loads((ROUND / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json').read_text(encoding='utf-8'))
    observed = json.loads((ROUND / 'DELIVERY_SPLIT_HEAD_RESULT.json').read_text(encoding='utf-8'))
    assert sha(ROUND / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json') == observed['registration_sha256']
    assert sha(ROOT / 'tools/qualify_delivery_split_final_head_v1.py') == reg['source_sha256']
    assert sha(DELIVERY / 'delivery_actual_action_head_v1.py') == reg['head_source_sha256']
    assert list(reg['inputs']) == [f'{arm}/request/{step:02d}.npz' for arm in ('pretrained_initial_fault0_worker', 'random888_initial_fault0_worker') for step in range(17)]
    head = construct_head(seed=reg['head_seed'])
    checks = 4
    total = 0
    for row in observed['rows']:
        path = DELIVERY / row['input']
        assert sha(path) == row['input_sha256'] == reg['inputs'][row['input']]; checks += 1
        with np.load(path, allow_pickle=False) as z:
            latent = z['latent'].copy()
            b = {k: z[k].copy() for k in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length', 'context', 'timestep')}
        n = len(b['actions'])
        direct = np.asarray(head(jnp.asarray(latent), b))
        h = jnp.asarray(latent[0], jnp.float32)
        fixed = dict(b)
        for key in ('actions', 'route_weights', 'prefix_weights', 'suffix_weights', 'slot_position_length'):
            fixed[key] = np.concatenate((b[key], np.broadcast_to(b[key][:1], (750-n,) + b[key].shape[1:])), axis=0)
        a = jnp.asarray(fixed['actions'], jnp.int32)
        terms = (h[a[:, 0]], h[a[:, 3]], h[a[:, 4]]) + tuple(jnp.asarray(fixed[k], jnp.float32) @ h for k in ('route_weights', 'prefix_weights', 'suffix_weights')) + (jnp.asarray(fixed['slot_position_length'], jnp.float32),)
        hidden = jnp.tanh(head.input(jnp.concatenate(terms, axis=1)))
        split = np.asarray(head.output(hidden[:n])[:, 0])
        assert direct.shape == split.shape and np.array_equal(direct.view(np.uint32), split.view(np.uint32)); checks += 1
        assert sha(path) == row['input_sha256'] and hashlib.sha256(direct.tobytes()).hexdigest() == row['direct_sha256'] and hashlib.sha256(split.tobytes()).hexdigest() == row['split_sha256']; checks += 1
        assert row['different_elements'] == 0 and row['actions'] == n; checks += 1
        total += n
    assert len(observed['rows']) == 34 and observed['strict_equal'] and observed['total_different_elements'] == 0; checks += 1
    return {'checks': checks, 'failures': 0, 'inputs': 34, 'actions': total, 'strict_equal': True,
            'full_request_budget_qualified': False}


def main():
    result = {'pickup': verify_pickup(), 'delivery': verify_delivery(), 'stable_online_gain': False}
    with (ROUND / 'INDEPENDENT_R12_SAVED_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'pickup_checks': result['pickup']['checks'], 'delivery_checks': result['delivery']['checks'], 'failures': 0}))


if __name__ == '__main__':
    main()
