"""Synthetic-only full MaskCO training engineering. No research training CLI.

Reconstruction follows the inspected CVRP customer-row CE implementation.
The additional pairwise route preference is a new, unvalidated surrogate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_routes(routes, demands, capacity):
    if not isinstance(routes, list) or not routes:
        raise ValueError('Full route list required')
    seen = []
    for route in routes:
        if not isinstance(route, list) or not route:
            raise ValueError('Empty or malformed vehicle route')
        if any(type(x) is not int or x < 1 or x >= len(demands) for x in route):
            raise ValueError('Customer ID must be an actual valid integer')
        if sum(demands[x] for x in route) > capacity:
            raise ValueError('Capacity violation')
        seen.extend(route)
    if sorted(seen) != list(range(1, len(demands))):
        raise ValueError('Missing or duplicated customer')


def adjacency(routes, nodes):
    # Matches sol2adj: an isolated customer's two depot edges have multiplicity 2.
    adj = np.zeros((nodes, nodes), np.float32)
    for route in routes:
        chain = [0] + route + [0]
        for a, b in zip(chain, chain[1:]):
            adj[a, b] += 1
            adj[b, a] += 1
    return adj


def retained_adjacency(routes, nodes, rng, keep):
    adj = np.zeros((nodes, nodes), np.float32)
    # One Bernoulli per directed route-vector edge; symmetric observation.
    for route in routes:
        chain = [0] + route + [0]
        for a, b in zip(chain, chain[1:]):
            if rng.random() < keep:
                adj[a, b] += 1
                adj[b, a] += 1
    return adj


def validate_splits(splits):
    required = {'teacher_train', 'teacher_diagnostic', 'development', 'confirmation'}
    if set(splits) != required:
        raise ValueError('All four independent split manifests required')
    seen_ids, seen_hashes = set(), set()
    for name, rows in splits.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError('Empty split: ' + name)
        for row in rows:
            key, content = row['instance_id'], row['data_sha256']
            if type(key) is not int or not isinstance(content, str) or len(content) != 64:
                raise ValueError('Invalid split identity')
            if key in seen_ids or content in seen_hashes:
                raise ValueError('Cross-split identity or data-content leakage')
            seen_ids.add(key)
            seen_hashes.add(content)


def flatten_params(params, nnx):
    return [(tuple(path), np.asarray(value.value)) for path, value in params.flat_state()]


def param_hash(params, nnx, group=None):
    h = hashlib.sha256()
    for path, value in flatten_params(params, nnx):
        if group is not None and path[0] != group:
            continue
        h.update(repr(path).encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def run_synthetic(upstream, out, registration):
    source_start = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ('models', 'modules', 'helpers') for p in sorted((Path(upstream) / folder).rglob('*.py'))}
    registered = json.loads(Path(registration).read_text(encoding='utf-8'))
    required = {'stage': 'synthetic_engineering_only', 'research_training_enabled': False,
                'research_data_generation_enabled': False, 'formal_gain_claim_enabled': False}
    for key, value in required.items():
        if type(registered.get(key)) is not type(value) or registered[key] != value:
            raise ValueError('Registration stage or scope mismatch: ' + key)
    if registered.get('optimizer') != {'name': 'adam', 'learning_rate': .001, 'fixed_steps': 3, 'selection': 'last_step_only'}:
        raise ValueError('Engineering optimizer changed')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    sys.path.insert(0, str(upstream))
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    from models.CVRPModel import CVRPModelConfig
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = CVRPModelConfig(num_layers=(1, 1), embed_dim=16, num_heads=2,
                          encoder_input_dim=10, dtype='float32', rngs=73)
    model = cfg.construct_model()
    graph, initial = nnx.split(model, nnx.Param)
    coords = np.asarray([[.5, .5], [.1, .1], [.2, .8], [.8, .1], [.9, .9]], np.float32)
    features = np.zeros((1, 5, 10), np.float32)
    features[0, :, :2] = coords
    features[0, 1:, 2] = .2
    features[0, :, 4] = 1.
    features[0, 1:, 6] = 1.
    preferred, alternative = [[1, 2], [3, 4]], [[1, 4], [2, 3]]
    demands = [0, 1, 1, 1, 1]
    for route in (preferred, alternative):
        validate_routes(route, demands, 3)
    target = jnp.asarray(adjacency(preferred, 5)[None])
    negative = jnp.asarray(adjacency(alternative, 5)[None])
    rng = np.random.default_rng(20261010)
    # The same masks, minibatches and order are replayed for each trained arm.
    schedule = []
    for step in range(1, 4):
        keep = float(rng.random())
        obs = retained_adjacency(preferred, 5, rng, keep)[None]
        schedule.append({'step': step, 'keep': keep, 'observed': obs.tolist(),
                         'training_key': 'synthetic:0', 'teacher_hash': canonical_hash(preferred)})
    schedule_hash = canonical_hash(schedule)
    tx = optax.adam(1e-3)

    def loss(params, obs, keep, preference_weight):
        current = nnx.merge(graph, params)
        logits = current.decode(current.encode(jnp.asarray(features)), jnp.asarray([keep]), obs)
        logp = jax.nn.log_softmax(logits, axis=-1)
        ce = -(logp[:, 1:] * target[:, 1:]).mean() * 2.5
        # Normalize by actual edge mass: depot row excluded as in upstream CE.
        good = (logp[:, 1:] * target[:, 1:]).sum() / target[:, 1:].sum()
        bad = (logp[:, 1:] * negative[:, 1:]).sum() / negative[:, 1:].sum()
        preference = jax.nn.softplus(-(good - bad))
        return ce + preference_weight * preference, (ce, preference)

    train_step = jax.jit(jax.value_and_grad(loss, has_aux=True))
    results = {}
    random_cfg = CVRPModelConfig(num_layers=(1, 1), embed_dim=16, num_heads=2,
                                 encoder_input_dim=10, dtype='float32', rngs=42)
    _, random_params = nnx.split(random_cfg.construct_model(), nnx.Param)
    for arm, weight, init in [('full_reconstruction_plus_preference', .2, initial),
                              ('reconstruction_only', 0., initial),
                              ('random_initial_same_objective', .2, random_params)]:
        params = jax.tree.map(lambda x: jnp.array(x), init)
        state = tx.init(params)
        records = []
        before = {g: param_hash(params, nnx, None if g == 'full' else g) for g in ('full', 'encoder', 'decoder')}
        for row in schedule:
            (total, (ce, preference)), grads = train_step(params, jnp.asarray(row['observed']), row['keep'], weight)
            updates, state = tx.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            values = [float(total), float(ce), float(preference)]
            if not all(math.isfinite(x) for x in values):
                raise ValueError('Nonfinite loss retained as engineering failure')
            records.append(dict(row, total=values[0], reconstruction=values[1], preference=values[2]))
        after = {g: param_hash(params, nnx, None if g == 'full' else g) for g in ('full', 'encoder', 'decoder')}
        save_path = out / (arm + '_fixed_final.pkl')
        with save_path.open('xb') as stream:
            pickle.dump({'params': params, 'config': cfg, 'step': 3}, stream)
        # Only this trusted, locally created engineering artifact is deserialized.
        with save_path.open('rb') as stream:
            saved = pickle.load(stream)
        reloaded = nnx.merge(graph, saved['params'])
        pred_before = nnx.merge(graph, params).decode(nnx.merge(graph, params).encode(jnp.asarray(features)), jnp.asarray([.5]), target)
        pred_after = reloaded.decode(reloaded.encode(jnp.asarray(features)), jnp.asarray([.5]), target)
        reload_ok = bool(np.array_equal(np.asarray(pred_before), np.asarray(pred_after)))
        results[arm] = {'before': before, 'after': after, 'schedule_sha256': schedule_hash,
                        'steps': records, 'fixed_final_step': saved['step'], 'save_reload_equal': reload_ok,
                        'actual_encoder_updated': before['encoder'] != after['encoder'],
                        'actual_decoder_updated': before['decoder'] != after['decoder'],
                        'saved_sha256': hashlib.sha256(save_path.read_bytes()).hexdigest()}
    full_model = nnx.merge(graph, initial)
    encoded = full_model.encode(jnp.asarray(features))
    empty_logits = np.asarray(full_model.decode(encoded, jnp.asarray([.5]), jnp.zeros_like(target)))
    retained_logits = np.asarray(full_model.decode(encoded, jnp.asarray([.5]), target))
    mask_context_effect = float(np.max(np.abs((empty_logits - retained_logits)[:, ~np.eye(5, dtype=bool)])))
    qa = {'stage': 'synthetic_engineering_only', 'research_training_enabled': False,
          'actual_checkpoint_loaded': False, 'full_scale_training_qualified': False,
          'coldchain_teacher_qualified': False, 'online_gain_established': False,
          'random_initial_sha256': param_hash(random_params, nnx), 'arms': results,
          'masked_adjacency_actual_logit_effect': mask_context_effect,
          'synthetic_optimizer': {'name': 'adam', 'learning_rate': .001, 'steps': 3},
          'synthetic_model': {'encoder_layers': 1, 'decoder_layers': 1, 'embedding': 16, 'heads': 2, 'input_dim': 10}}
    qa['registration_sha256'] = hashlib.sha256(Path(registration).read_bytes()).hexdigest()
    source_end = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ('models', 'modules', 'helpers') for p in sorted((Path(upstream) / folder).rglob('*.py'))}
    qa['source_start'] = source_start
    qa['source_end'] = source_end
    qa['source_unchanged'] = bool(source_start) and source_start == source_end
    qa['engineering_pass'] = all(r['actual_encoder_updated'] and r['actual_decoder_updated'] and r['save_reload_equal']
                                 and [s['step'] for s in r['steps']] == [1, 2, 3] for r in results.values())
    qa['engineering_pass'] &= results['full_reconstruction_plus_preference']['before'] == results['reconstruction_only']['before']
    qa['engineering_pass'] &= results['full_reconstruction_plus_preference']['after']['full'] != results['reconstruction_only']['after']['full']
    qa['engineering_pass'] &= mask_context_effect > 0 and param_hash(random_params, nnx) != param_hash(initial, nnx)
    qa['engineering_pass'] &= qa['source_unchanged']
    with (out / 'ENGINEERING_QA.json').open('x', encoding='utf-8') as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'engineering_pass': qa['engineering_pass'], 'stage': qa['stage']}))
    return qa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['synthetic_engineering_only'], required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    run_synthetic(args.upstream, args.out, args.registration)
