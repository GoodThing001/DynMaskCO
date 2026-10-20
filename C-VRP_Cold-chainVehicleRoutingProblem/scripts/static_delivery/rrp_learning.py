"""Actual masked successor/wait learning, grouping, and non-pickle checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rrp_data import load_instance, public_instances, source_protocol, PROTOCOL_PATH, PRIMARY_PROFILE
from rrp_model import make_model
from rrp_features import global_context, SCHEMA


def geometry_groups():
    groups = {}
    for path in public_instances():
        x = load_instance(path)
        h = hashlib.sha256(x.distances.tobytes() + x.demands.tobytes()).hexdigest()
        groups.setdefault(h, []).append(x.name)
    return groups


def validate_split(split):
    groups = geometry_groups()
    role_of = {}
    available = {name for values in groups.values() for name in values}
    for role, names in split.items():
        for name in names:
            if name not in available or name in role_of:
                raise ValueError('Unknown/duplicate split instance')
            role_of[name] = role
    for names in groups.values():
        if len({role_of[n] for n in names if n in role_of}) > 1:
            raise ValueError('Shared geometry cannot cross split roles')
    return split


def make_batch(instance, route, waits_s, kept_edges, *, objective='with_wage', profile=PRIMARY_PROFILE, time_domain='source_day_domain'):
    n = instance.nodes
    if len(route) != n or set(route) != set(range(n)) or len(waits_s) != n:
        raise ValueError('Invalid teacher route/waits')
    targets = np.empty(n, dtype=np.int32)
    wait_targets = np.zeros(n, dtype=np.float32)
    edges = list(zip(route, list(route[1:]) + [route[0]]))
    if not set(kept_edges) <= set(edges):
        raise ValueError('Kept edges must come from the current incumbent')
    for i, node in enumerate(route):
        targets[node] = route[(i + 1) % n]
        wait_targets[node] = waits_s[i]
    known = np.array([any(a == i for a, _ in kept_edges) for i in range(n)], np.float32)
    adjacency = np.zeros((n, n), np.float32)
    for a, b in kept_edges:
        adjacency[a, b] = 1
    horizon = (source_protocol()['max_tour_hours'] * 3600 if time_domain == 'operational_tour_9h'
               else 86400 - instance.raw['start_time'] * 3600)
    max_distance = max(float(instance.distances.max()), np.finfo(np.float32).tiny)
    features = np.column_stack((np.arange(n) == 0, instance.demands / instance.demands.sum(),
                                instance.distances[0] / max_distance,
                                known * wait_targets / horizon, known)).astype(np.float32)
    context = global_context(instance, objective, profile, time_domain)
    features = np.concatenate((features, np.broadcast_to(context, (n, len(context)))), axis=1)
    return {'features': features[None],
            'distance_bias': (-instance.distances / max_distance).astype(np.float32)[None],
            'adjacency': adjacency[None], 'timestep': np.array([len(kept_edges) / n], np.float32),
            'targets': targets[None], 'wait_targets': np.log1p(wait_targets)[None],
            'wait_limit_s': np.array([horizon], np.float32),
            'masked_rows': (1 - known)[None]}


def predict(model, batch):
    logits, wait_log = model(*(jnp.asarray(batch[k]) for k in ('features', 'distance_bias', 'adjacency', 'timestep')))
    scores = np.asarray(logits)[0]
    wait_limit = np.log1p(float(batch['wait_limit_s'][0]))
    waits = np.expm1(np.clip(np.asarray(wait_log)[0], 0, wait_limit))
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(waits)):
        raise ValueError('Nonfinite model output')
    waits[0] = 0
    return scores, waits


def _leaves(model):
    return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))


def parameter_digest(model):
    h = hashlib.sha256()
    for leaf in _leaves(model):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Trainer:
    def __init__(self, model, learning_rate, weight_decay):
        self.model = model
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        if not np.isfinite(learning_rate) or not np.isfinite(weight_decay) or learning_rate <= 0 or weight_decay < 0:
            raise ValueError('Invalid optimizer configuration')
        self.tx = optax.adamw(learning_rate, weight_decay=weight_decay)
        self.graphdef, self.state = nnx.split(model)
        self.opt_state = self.tx.init(self.state)
        self.steps = 0
        graphdef, tx = self.graphdef, self.tx

        @jax.jit
        def update(state, opt_state, batch):
            def loss(params):
                current = nnx.merge(graphdef, params)
                logits, wait_log = current(*(batch[k] for k in ('features', 'distance_bias', 'adjacency', 'timestep')))
                mask = batch['masked_rows']
                denominator = jnp.maximum(mask.sum(), 1)
                ce = (optax.softmax_cross_entropy_with_integer_labels(logits, batch['targets']) * mask).sum() / denominator
                mse = (((wait_log - batch['wait_targets']) ** 2) * mask).sum() / denominator
                return ce + mse, (ce, mse)
            (value, (ce, mse)), grads = jax.value_and_grad(loss, has_aux=True)(state)
            updates, opt_state = tx.update(grads, opt_state, state)
            return optax.apply_updates(state, updates), opt_state, (value, ce, mse, optax.global_norm(grads))
        self._update = update

    def step(self, batch):
        before = [np.asarray(a).copy() for a in jax.tree_util.tree_leaves(self.state)]
        self.state, self.opt_state, values = self._update(self.state, self.opt_state, {k: jnp.asarray(v) for k, v in batch.items()})
        metrics = [float(v) for v in values]  # Synchronize actual CPU/GPU computation.
        if not all(np.isfinite(metrics)):
            raise ValueError('Nonfinite training loss/gradient')
        nnx.update(self.model, self.state)
        self.steps += 1
        changed = sum(not np.array_equal(a, np.asarray(b)) for a, b in zip(before, jax.tree_util.tree_leaves(self.state)))
        return dict(zip(('loss', 'successor_ce', 'wait_log_mse', 'gradient_norm'), metrics), updated_parameter_leaves=changed)


def _save_tree(path, tree):
    leaves = jax.tree_util.tree_leaves(tree)
    np.savez(path, **{'leaf%05d' % i: np.asarray(v) for i, v in enumerate(leaves)})


def _load_tree(path, template):
    leaves, definition = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as values:
        if len(values.files) != len(leaves):
            raise ValueError('Checkpoint leaf count mismatch')
        loaded = []
        for i, original in enumerate(leaves):
            value = values['leaf%05d' % i]
            if value.shape != np.shape(original) or value.dtype != np.asarray(original).dtype:
                raise ValueError('Checkpoint shape/dtype mismatch')
            if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
                raise ValueError('Nonfinite checkpoint leaf: ' + path.name)
            loaded.append(jnp.asarray(value))
    return definition.unflatten(loaded)


def save_checkpoint(path, trainer, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    _save_tree(path / 'weights.npz', trainer.state)
    _save_tree(path / 'optimizer.npz', trainer.opt_state)
    data = {'model': trainer.model.config, 'feature_schema': SCHEMA, 'steps': trainer.steps,
            'source_protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            'learning_rate': trainer.learning_rate, 'weight_decay': trainer.weight_decay,
            'metadata': metadata, 'files': {}}
    for name in ('weights.npz', 'optimizer.npz'):
        data['files'][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / 'checkpoint.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def load_checkpoint(path):
    path = Path(path)
    data = json.loads((path / 'checkpoint.json').read_text(encoding='utf-8'))
    if data.get('feature_schema') != SCHEMA:
        raise ValueError('Checkpoint feature schema mismatch')
    if data.get('source_protocol_sha256') != hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest():
        raise ValueError('Checkpoint source protocol mismatch')
    for name in ('weights.npz', 'optimizer.npz'):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != data['files'][name]:
            raise ValueError('Checkpoint SHA256 mismatch')
    model = make_model(data['model']['profile'], data['model']['seed'])
    trainer = Trainer(model, data['learning_rate'], data['weight_decay'])
    trainer.state = _load_tree(path / 'weights.npz', trainer.state)
    trainer.opt_state = _load_tree(path / 'optimizer.npz', trainer.opt_state)
    trainer.steps = data['steps']
    nnx.update(model, trainer.state)
    return trainer, data['metadata']
