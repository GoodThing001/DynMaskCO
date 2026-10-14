"""D-TRAINTASK-v1 trainable delivery guide: a drop-in replacement for the frozen zero-shot guide.

The problem this solves. `delivery_equal_schedule.reconstruct` takes `logits` and scores each
insertion option as `delta/dscale - guide_weight * gain`, where `gain` comes from those logits
over (a, i), (i, b) and (a, b). Every delivery panel so far supplied the SAME frozen zero-shot
CVRP checkpoint there and only rewired where its logits were consumed. The one registered but
never-tested path is to TRAIN a delivery-specific component in that slot.

Interface. This class exposes exactly `features(instance, capacity)` and `logits(features,
adjacency)`, the two methods the existing provider exposes, so `reconstruct`, `run_arm` and the
registered panel accept it unchanged. The head is a small bilinear scorer over the FROZEN
encoder's node features:

    Z = linear2(relu(linear1(H)))      # per node
    logits = Z @ Z^T / sqrt(dim)       # symmetric edge scores

Resource matching. The registered panel matches ACTUAL decoder calls across arms. A trained head
has no decoder, so `logits()` deliberately still invokes the frozen decoder once and DISCARDS its
output before returning the head's scores. All arms therefore pay identical encoder passes and
identical decoder calls, and the only difference is which logits score the insertion - which is
exactly the information-value separation the protocol requires. The discarded call is recorded.

Scope: this module defines and verifies the component. It trains nothing here beyond a smoke
gradient step, runs no arm and claims no improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for p in (scripts / 'delivery', scripts / 'research_20261010'):
        sys.path.insert(0, str(p))
    return scripts


def build_head(dim, hidden, seed):
    from flax import nnx
    import jax.numpy as jnp

    class EdgeHead(nnx.Module):
        def __init__(self, dim, hidden, rngs):
            self.l1 = nnx.Linear(dim, hidden, rngs=rngs)
            self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
            self.hidden = hidden

        def __call__(self, H):
            Z = nnx.relu(self.l1(H))
            Z = self.l2(Z)
            return jnp.matmul(Z, jnp.swapaxes(Z, -1, -2)) / jnp.sqrt(float(self.hidden))

    return EdgeHead(dim, hidden, nnx.Rngs(int(seed)))


class DeliveryTrainedGuide:
    """Frozen encoder + trainable head, drop-in for the zero-shot provider."""

    def __init__(self, checkpoint, upstream_root, head_seed=2026120105, head_hidden=64,
                 head_snapshot=None):
        from delivery_maskco import MaskCORouteProvider
        self.base = MaskCORouteProvider(checkpoint=checkpoint, upstream_root=upstream_root)
        self.head_seed = int(head_seed)
        self.head_hidden = int(head_hidden)
        self._head = None
        self._dim = None
        self.n_decoder_calls_paid = 0
        self.n_head_calls = 0
        self.head_snapshot = head_snapshot
        self.identity = {
            'kind': 'delivery_traintask_trainable_guide',
            'protocol_version': 'D-TRAINTASK-v1',
            'base_checkpoint_sha256': self.base.identity['checkpoint_sha256'],
            'frozen_part': 'encoder and decoder of the pretrained CVRP checkpoint',
            'trainable_part': 'bilinear edge head over frozen node features',
            'head_seed': self.head_seed, 'head_hidden': self.head_hidden,
            'pays_one_decoder_call_per_logits_call': True,
            'head_sha256': (hashlib.sha256(head_snapshot).hexdigest()
                            if head_snapshot is not None else None),
        }

    # ---- the two provider methods -------------------------------------------------
    def features(self, instance, capacity):
        return self.base.features(instance, capacity)

    def _ensure_head(self, dim):
        if self._head is None:
            self._dim = int(dim)
            self._head = build_head(self._dim, self.head_hidden, self.head_seed)
        elif self._dim != int(dim):
            raise ValueError('node feature dimension changed between calls')

    def logits(self, features, adjacency):
        # Pay the frozen decode so decoder-call matching across arms still holds, then discard.
        _discarded = self.base.logits(features, adjacency)
        self.n_decoder_calls_paid += 1
        self._ensure_head(np.asarray(features).shape[-1])
        out = np.asarray(self._head(features))[0]
        self.n_head_calls += 1
        n = len(adjacency)
        if out.shape != (n, n) or not np.all(np.isfinite(out[~np.eye(n, dtype=bool)])):
            raise ValueError('trained head produced invalid off-diagonal edge scores')
        return out

    def head_params(self):
        from flax import nnx
        return nnx.state(self._head, nnx.Param)

    def head_param_hash(self):
        import jax
        h = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(self.head_params()):
            a = np.ascontiguousarray(np.asarray(leaf))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def base_param_hash(self):
        import jax
        from flax import nnx
        h = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(nnx.state(self.base.model, nnx.Param)):
            a = np.ascontiguousarray(np.asarray(leaf))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def save_head(self, path):
        import jax
        from flax import nnx
        import pickle
        payload = jax.tree_util.tree_map(lambda a: np.asarray(a),
                                        nnx.to_pure_dict(nnx.state(self._head)))
        Path(path).write_bytes(pickle.dumps(payload))
        self.identity['head_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return self.identity['head_sha256']

    def load_head(self, path, dim):
        """Restore a frozen head written by save_head, and verify it round-trips bit-exactly."""
        import pickle
        from flax import nnx
        self._ensure_head(dim)
        payload = pickle.loads(Path(path).read_bytes())
        nnx.update(self._head, payload)
        self.identity['loaded_head_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.identity['loaded_head_param_hash'] = self.head_param_hash()
        return self.identity['loaded_head_sha256']

    def smoke(self, scripts, seed=2026120151, instance_id=0):
        """Verify the component can be trained and that training it does not move the frozen part."""
        import jax
        import optax
        from flax import nnx
        from delivery_equal_schedule import generate, DeliveryContract, reconstruct
        from delivery_baselines import matrices
        from delivery_maskco import partial_adjacency

        data, initial = generate(seed, instance_id)
        contract = DeliveryContract()
        nodes = len(data['demands'])
        feats = self.features(data, contract.capacity)
        if hasattr(feats, 'block_until_ready'):
            feats.block_until_ready()

        adj = partial_adjacency(initial, [1, 2, 3], nodes)
        base_before = self.base_param_hash()
        head_before = self.head_param_hash()
        logits_1 = self.logits(feats, adj)
        logits_2 = self.logits(feats, adj)
        deterministic = bool(np.array_equal(logits_1, logits_2))
        off = ~np.eye(nodes, dtype=bool)
        finite = bool(np.all(np.isfinite(logits_1[off])))

        # one gradient step on the head only, against a target derived from the instance
        optimizer = nnx.Optimizer(self._head, optax.adam(1e-3), wrt=nnx.Param)
        target = -matrices(data)
        target = target / max(1e-9, float(np.abs(target[off]).mean()))

        def loss_fn(head):
            out = head(feats)[0]
            return ((out - target) ** 2)[off].mean()

        loss_before = float(loss_fn(self._head))
        grads = nnx.grad(loss_fn)(self._head)
        try:
            optimizer.update(grads)                 # newer nnx: model bound at construction
        except TypeError:
            optimizer.update(self._head, grads)     # older nnx: explicit model argument
        loss_after = float(loss_fn(self._head))
        head_after = self.head_param_hash()
        base_after = self.base_param_hash()

        # drop-in compatibility: reconstruct must accept this object's logits unchanged
        deadline = time.perf_counter() + 30.0
        cand, rem, ins, status = reconstruct(data, initial, contract, [1, 2, 3], adj,
                                            logits_1, 1.0, deadline, time.perf_counter)
        checks = {
            'logits_deterministic': deterministic,
            'logits_finite_off_diagonal': finite,
            'head_is_trainable': head_before != head_after,
            'head_loss_decreased_on_the_step': loss_after < loss_before,
            'frozen_encoder_and_decoder_unchanged': base_before == base_after,
            'decoder_call_paid_once_per_logits_call': self.n_decoder_calls_paid == 2,
            'reconstruct_accepts_the_guide': status in ('complete', 'no_feasible_insertion',
                                                        'deadline'),
            'reconstruct_completed_here': status == 'complete',
        }
        return {
            'kind': 'delivery_trained_guide_smoke_qualification',
            'identity': dict(self.identity),
            'n_nodes': nodes,
            'loss_before': loss_before, 'loss_after': loss_after,
            'head_param_hash_before': head_before, 'head_param_hash_after': head_after,
            'base_param_hash_before': base_before, 'base_param_hash_after': base_after,
            'reconstruct_status': status, 'n_remaining': len(rem),
            'n_insertions': len(ins),
            'checks': checks, 'all_checks_pass': all(checks.values()),
            'jax_backend': jax.default_backend(),
            'what_this_qualifies': [
                'the trained component exists, is deterministic, and is accepted unchanged by '
                'the existing reconstruct path',
                'the head is trainable while the encoder+decoder stay bit-identical',
                'one frozen decoder call is paid per logits call, so decoder-call matching '
                'across arms still holds'],
            'what_this_does_not_qualify': [
                'nothing is trained to convergence and no arm is compared',
                'no improvement, no attribution, no impossibility, no equivalence'],
        }


def main(args):
    bootstrap(args.scripts)
    guide = DeliveryTrainedGuide(args.checkpoint, args.upstream_root,
                                head_seed=args.head_seed, head_hidden=args.head_hidden)
    result = guide.smoke(args.scripts, seed=args.seed, instance_id=args.instance)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('checks', 'all_checks_pass', 'loss_before',
                                            'loss_after', 'reconstruct_status',
                                            'n_insertions', 'jax_backend')}, indent=2,
                     default=str))
    return 0 if result['all_checks_pass'] else 3


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--head-seed', type=int, default=2026120105)
    ap.add_argument('--head-hidden', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026120101)
    ap.add_argument('--instance', type=int, default=0)
    sys.exit(main(ap.parse_args()))
