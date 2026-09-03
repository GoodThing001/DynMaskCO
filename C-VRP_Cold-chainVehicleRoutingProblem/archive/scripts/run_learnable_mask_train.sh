#!/bin/bash
# ============================================================
# Phase 3b v2: REINFORCE with real mask-reconstruct cost delta
#
# 修复 v1: beam samples from same logits → no mask signal
# v2: 1) beam → route_A  2) extract features → compute learned mask
#     3) apply mask → reconstruct with beam → route_B
#     4) reward = cost(route_A) - cost(route_B) → REINFORCE
# ============================================================
set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

POLICY_FILE="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/learnable_mask_weights.npy"

echo "============================================================"
echo "Phase 3b v2: Real mask-reconstruct REINFORCE"
echo "============================================================"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python3 -c "
import sys, os, numpy as np, time
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding')
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models')
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib')
sys.path.insert(0, 'models')

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModelConfig
from modules.functional import coord_normalize
from resource_beam import ResourceBeamSearcher
from learnable_mask import extract_edge_features, gumbel_topk, MaskPolicyTrainer

print('Loading model...')
params, _, _, model_config, _, _ = load_ckpt(
    'C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/mixed_edod/step50000.ckpt')
if model_config is None:
    model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
    model_config.encoder_input_dim = 7
model = model_config.construct_model()
if params is not None: model = nnx.merge(nnx.graphdef(model), params)
_MODEL_IN = int(model.init_proj.kernel.shape[0])

@jax.jit
def encode_fn(raw_features):
    raw_features = raw_features.at[..., :2].set(coord_normalize(raw_features[..., :2]))
    return model.encode(raw_features[..., :_MODEL_IN])

@jax.jit
def logit_fn(features, timestep):
    adjmat = jnp.zeros((features.shape[0], 51, 51), dtype=jnp.int8)
    return model.decode(features, timestep, adjmat.astype(jnp.float32))

print('Loading data...')
data = dict(np.load('C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz'))
N = min(32, data['coords'].shape[0])
cap, tw_max = 50, float(data['tw_end'].max())

f_list = [
    data['coords'][:N], (data['demands'][:N]/cap)[...,None],
    (data['tw_start'][:N]/tw_max)[...,None], (data['tw_end'][:N]/tw_max)[...,None],
    (data['temp_class'][:N].astype(np.float32)/2.0)[...,None],
]
if 'reveal_time' in data: f_list.append((data['reveal_time'][:N].astype(np.float32)/tw_max)[...,None])
raw_features = np.concatenate(f_list, axis=-1).astype(np.float32)
feats = np.array(encode_fn(jnp.array(raw_features)))
logits_all = np.array(logit_fn(jnp.array(feats), jnp.full(N, 0.3, dtype=jnp.float32)))
ql = data.get('quality_loss', np.zeros_like(data['demands']))[:N].astype(np.float32)
pad_len = data['routes'].shape[1]

trainer = MaskPolicyTrainer(lr=0.02)
rng = np.random.default_rng(42)

episodes, updates = 80, 0
for ep in range(episodes):
    ep_delta = 0.0
    for idx in range(N):
        logits_b = logits_all[idx].copy()
        logits_b[:, 0] = -1e9; np.fill_diagonal(logits_b, -1e9)

        searcher = ResourceBeamSearcher(
            data['coords'][idx], data['tw_start'][idx], data['tw_end'][idx],
            data['service_time'][idx], data['demands'][idx], cap, 1.0, K=12, tw_margin=0.05)

        # 1) beam → baseline route
        route_A, _ = searcher.generate(logits_b, max_steps=200)
        if not route_A or len(route_A) <= 2: continue
        prev, cost_A = 0, 0.0
        for nd in route_A:
            if nd <= 0 or nd >= 51: continue
            cost_A += searcher.dist[prev, nd]; prev = nd

        # 2) Convert route to padded array, extract edge features
        padded = np.pad(route_A, (0, max(0, pad_len - len(route_A))), constant_values=0)[:pad_len]
        edge_feats_all = extract_edge_features(
            padded[None,:], logits_b[None,:,:], data['tw_start'][idx:idx+1],
            data['tw_end'][idx:idx+1], ql[idx:idx+1], event_nodes=None,
            current_cycle=5, max_cycles=40)
        if not edge_feats_all or edge_feats_all[0].shape[0] == 0: continue
        edge_feats = edge_feats_all[0]  # (E, 6)

        # 3) Learned mask policy → soft mask
        logits_m = trainer.compute_logits(edge_feats)
        mask_soft = gumbel_topk(logits_m, k=max(1, int(0.3 * len(logits_m))), temperature=0.5, rng=rng)

        # 4) Hard threshold: keep top (1-keep_rate)*E edges (mask the rest)
        #    mask_soft[i] high → prefer to MASK this edge → remove from frozen set
        keep_count = max(1, int(0.3 * len(edge_feats)))
        # Edges to KEEP = those with lowest mask_soft (model thinks these should stay)
        keep_indices = np.argsort(mask_soft)[:keep_count]

        # Build frozen edges from kept positions
        frozen_edges = []
        E = edge_feats.shape[0]
        for e_idx in range(E):
            if e_idx in keep_indices:
                pos = e_idx % pad_len
                if pos < pad_len:
                    i = int(padded[pos])
                    j = int(padded[(pos+1) % pad_len])
                    if i > 0 or j > 0:
                        frozen_edges.append((i, j))

        # 5) Reconstruct with frozen edges
        route_B = searcher.reconstruct(frozen_edges[:max(1,int(0.3*len(route_A)))], logits_b, max_steps=200)
        if not route_B or len(route_B) <= 2: route_B = route_A
        prev, cost_B = 0, 0.0
        for nd in route_B:
            if nd <= 0 or nd >= 51: continue
            cost_B += searcher.dist[prev, nd]; prev = nd

        # 6) REINFORCE: positive reward if cost dropped
        trainer.update(edge_feats, mask_soft, cost_A, cost_B)
        if cost_B < cost_A - 0.01: updates += 1
        ep_delta += (cost_A - cost_B)

    if (ep+1) % 20 == 0:
        c = trainer.get_config()
        print(f'  ep {ep+1:3d}/{episodes} | delta={ep_delta:.2f} | updates={updates} | '
              f'evt={c[\"event_prox\"]:+.3f} conf={c[\"confidence\"]:+.3f} slack={c[\"tw_slack\"]:+.3f}')

print()
print(f'=== v2 Complete ===')
print(f'Total delta: {ep_delta:.1f}, successful updates: {updates}')
c = trainer.get_config()
for name, w in c.items():
    d = 'MORE mask' if w > 0.01 else ('LESS mask' if w < -0.01 else 'neutral')
    print(f'  {name:15s}: {w:+7.4f} → {d}')
np.save('${POLICY_FILE}', trainer.weights)
print(f'Saved to ${POLICY_FILE}')
"

echo ""
echo "=== DONE ==="
