"""Frozen real M-pre features and the ONLY trainable CA-UGMR mask-value head.

CPU only. No decoder fine-tuning, insertion head, real-day training or batch CLI.
The head starts at zero (KEEP); nonzero synthetic bias is a wiring fixture, not
a learned policy. The backbone is deliberately outside the head's nnx graph.
"""
from __future__ import annotations

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib
from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('models', 'experimental', 'simulation', 'decoding', 'coldchain'))
from mpre import load_cvrp_model
from dynmaskco_cc import ScoringMLP
from mpre_trained import edge_insertion_scores
from mpre_replanner import retained_edge_ratio
from dynmaskco_cc_graph import build_plan_adjacency, build_visible_index
from cvrptw_utils import coord_normalize_visible

EXPECTED_CHECKPOINT = '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
PUBLIC_NODE_BUCKET = 51  # design's public N50 + depot, never hidden cardinality


def state_digest(state):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenBackbone:
    def __init__(self, checkpoint):
        if any(d.platform != 'cpu' for d in jax.devices()):
            raise RuntimeError('STOP: CPU backend required')
        checkpoint = Path(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != EXPECTED_CHECKPOINT:
            raise ValueError('STOP: checkpoint byte identity mismatch')
        self.model, self.cfg, step = load_cvrp_model(str(checkpoint))
        self.d_encoder, self.d_decoder = map(int, self.cfg.embed_dim)
        if self.cfg.encoder_input_dim != 3 or (self.d_encoder, self.d_decoder) != (512, 512):
            raise ValueError('STOP: unexpected actual pretrained model configuration')
        self.manifest = {'checkpoint_sha256': digest, 'encoder_dim': self.d_encoder,
            'decoder_dim': self.d_decoder, 'layers': tuple(self.cfg.num_layers),
            'input_dim': self.cfg.encoder_input_dim, 'step': int(step), 'platform': 'cpu',
            'parameter_sha256': self.parameter_digest(), 'public_node_bucket': PUBLIC_NODE_BUCKET}
        self._encoding_key = None
        self._H = None

    def parameter_digest(self):
        return state_digest(nnx.state(self.model, nnx.Param))

    def encode_view(self, view):
        from p2_pos01_reference import validate_view
        validate_view(view)
        nodes = [o.oid for o in view.orders]
        mapping, inverse, n = build_visible_index([x for x in nodes if x != 0])
        if n > PUBLIC_NODE_BUCKET:
            raise ValueError('public bucket exceeded; do not shrink or silently truncate')
        orders = {o.oid: o for o in view.orders}
        coords = jnp.array([[[orders[c].x, orders[c].y] for c in inverse]], jnp.float32)
        normalized = np.asarray(coord_normalize_visible(coords, None))
        raw = np.zeros((1, PUBLIC_NODE_BUCKET, 3), np.float32)
        raw[0, :n, :2] = normalized[0]
        raw[0, :n, 2] = [orders[c].demand/view.capacity for c in inverse]
        valid = np.arange(PUBLIC_NODE_BUCKET) < n
        key = (tuple(inverse), raw.tobytes())
        if key != self._encoding_key:
            pair = valid[:, None] & valid[None, :]
            H = self.model.encode(jnp.asarray(raw),
                                  attn_options={'bias': jnp.where(pair, 0.0, -1e9)[None]})
            self._H = jax.lax.stop_gradient(H)
            if not np.isfinite(np.asarray(H)).all():
                raise ValueError('nonfinite actual encoder output')
            self._encoding_key = key
        return self._H, mapping, n, valid

    def context(self, view, base, partial):
        H, mapping, n, valid = self.encode_view(view)
        A0 = build_plan_adjacency(base, mapping, n)
        Am = build_plan_adjacency(partial, mapping, n)
        t = retained_edge_ratio(A0, Am)
        if not 0 <= t <= 1:
            raise ValueError('retained-edge timestep outside [0,1]')
        padded = np.zeros((PUBLIC_NODE_BUCKET, PUBLIC_NODE_BUCKET), np.float32)
        padded[:n, :n] = Am
        pair = valid[:, None] & valid[None, :]
        bias = jnp.where(pair, jnp.asarray(padded), -1e9)[None]
        Z = self.model.decode(H, jnp.array([t], jnp.float32), bias,
                              target='embed', attn_options={})
        Z = jax.lax.stop_gradient(Z)
        if not np.isfinite(np.asarray(Z)).all():
            raise ValueError('nonfinite actual decoder output')
        return H, Z, mapping, t

    def insertion_scores(self, view, base, partial, actions):
        _, Z, index, _ = self.context(view, base, partial)
        L = self.model.feature2logit(Z)
        cust = jnp.array([[index[c] for c, _, _ in actions]], jnp.int32)
        pred = jnp.array([[index[a.predecessor] for _, a, _ in actions]], jnp.int32)
        succ = jnp.array([[index[a.successor] for _, a, _ in actions]], jnp.int32)
        return np.asarray(edge_insertion_scores(L, cust, pred, succ)[0])


def vehicle_features(view, vehicle, plan):
    cc = vehicle.coldchain_state
    T = max(next(o.tw_end for o in view.orders if o.oid == 0), 1e-9)
    lots = () if cc is None else cc.cargo_manifest  # ALL lots, no MAX_LOTS truncation
    quality = [min((l.quality_remaining/l.initial_quality for l in lots if l.temp_class == z),
                   default=1.0) for z in range(3)]
    temperatures = (0.0,)*3 if cc is None else cc.compartment_temperature_c
    loads = (0.0,)*3 if cc is None else cc.zone_load
    return np.asarray([(plan.anchor_time-view.clock)/T,
        (view.capacity-1e-4-plan.anchor_load)/view.capacity,
        float(vehicle.status == 'committed'), float(bool(lots)),
        *[t/40.0 for t in temperatures], *[x/view.capacity for x in loads], *quality,
        0.0 if cc is None else cc.cumulative_energy_kwh/view.energy_budget,
        0.0 if vehicle.committed_finish is None else (vehicle.committed_finish-view.clock)/T], np.float32)


class MaskValueModel:
    def __init__(self, backbone, seed=0, synthetic_bias=0.0):
        self.backbone = backbone  # outside nnx train graph
        self.feature_dim = 2*(5*backbone.d_decoder+24)+48
        self.head = ScoringMLP(self.feature_dim, hidden=(128, 64), rngs=seed)
        self.head.l3.kernel.value = jnp.zeros_like(self.head.l3.kernel.value)
        self.head.l3.bias.value = jnp.full_like(self.head.l3.bias.value, synthetic_bias)

    def features(self, view, base, masks):
        from p2_pos01_reference import remove_mask
        orders = {o.oid: o for o in view.orders}
        vehicles = {v.vehicle_id: v for v in view.vehicles}
        T = max(orders[0].tw_end, 1e-9)
        vf = {vid: vehicle_features(view, vehicles[vid], p) for vid, p in base.items()}
        fleet = np.stack(list(vf.values()))
        used = sum(v.coldchain_state.cumulative_energy_kwh for v in view.vehicles
                   if v.coldchain_state is not None)
        global_features = np.concatenate([fleet.mean(0), fleet.min(0), fleet.max(0),
            [view.clock/T, view.booking_horizon/T, (view.energy_budget-used)/view.energy_budget]])
        rows = []
        for mask in masks:
            H, Z, ix, _ = self.backbone.context(view, base, remove_mask(base, mask))
            H, Z, tokens = np.asarray(H)[0], np.asarray(Z)[0], []
            for c in mask:
                vid, p = next((vid, p) for vid, p in base.items() if c in p.suffix)
                rank = p.suffix.index(c)
                pred = p.anchor_node if rank == 0 else p.suffix[rank-1]
                succ = 0 if rank+1 == len(p.suffix) else p.suffix[rank+1]
                o = orders[c]
                lots = () if vehicles[vid].coldchain_state is None else vehicles[vid].coldchain_state.cargo_manifest
                # Time distance to mutable frontier uses visible planned travel/service only.
                elapsed, prev = 0.0, p.anchor_node
                for n in p.suffix[:rank]:
                    no, po = orders[n], orders[prev]
                    elapsed += math_distance(po, no)/view.tw_speed+no.service
                    prev = n
                explicit = [o.demand/view.capacity, (o.tw_start-view.clock)/T,
                    (o.tw_end-view.clock)/T, o.service/T, o.temp_class/2,
                    o.initial_quality, float(any(l.temp_class == o.temp_class for l in lots)),
                    rank/max(1, len(p.suffix)-1)]
                tokens.append(np.concatenate([H[ix[c]], Z[ix[c]], Z[ix[p.anchor_node]],
                    Z[ix[pred]], Z[ix[succ]], explicit, vf[vid], [elapsed/T]]))
            stack = np.stack(tokens)
            rows.append(np.concatenate([stack.mean(0), stack.max(0), global_features]))
        out = np.asarray(rows, np.float32)
        if out.shape != (len(masks), self.feature_dim) or not np.isfinite(out).all():
            raise ValueError('invalid public mask feature tensor')
        return out

    def score_masks(self, view, base, masks):
        features = jax.lax.stop_gradient(jnp.asarray(self.features(view, base, masks)))
        return np.asarray(self.head(features))

    def one_synthetic_step(self, features, labels, scale=1.0):
        """One optimizer step on cached synthetic tensors. NEVER updates backbone.

        One state/day here; full day-balanced 200-step fitting is deliberately absent.
        """
        if scale < 1 or not np.isfinite(scale) or not np.isfinite(labels).all():
            raise ValueError('invalid synthetic target/scale')
        features = jax.lax.stop_gradient(jnp.asarray(features))
        labels = jnp.asarray(labels)
        gd, params = nnx.split(self.head, nnx.Param)
        def loss(state):
            pred = nnx.merge(gd, state)(features)/scale
            y = labels/scale
            err = pred-y
            ae = jnp.abs(err)
            huber = jnp.mean(jnp.where(ae <= 1, .5*err*err, ae-.5))
            diff = y[:, None]-y[None, :]
            eligible = jnp.triu(diff != 0, 1)
            pair = jax.nn.softplus(-jnp.sign(diff)*(pred[:, None]-pred[None, :]))
            ranking = jnp.sum(jnp.where(eligible, pair, 0))/jnp.maximum(1, eligible.sum())
            return huber+.1*ranking
        value, grad = jax.value_and_grad(loss)(params)
        import optax
        optimizer = optax.adam(1e-3)
        updates, _ = optimizer.update(grad, optimizer.init(params), params)
        self.head = nnx.merge(gd, optax.apply_updates(params, updates))
        return float(value)


def math_distance(a, b):
    return float(np.hypot(a.x-b.x, a.y-b.y))
