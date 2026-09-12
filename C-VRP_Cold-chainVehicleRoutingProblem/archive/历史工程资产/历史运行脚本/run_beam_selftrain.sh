#!/bin/bash
# ============================================================
# Phase 3a Beam Full Self-Training: 全量标签 + 50000步微调
# 用法: bash run_beam_selftrain.sh
# ============================================================
set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/beam_full_st"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix"
DECODER="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
mkdir -p "$CKPT_DIR"

BASE_CKPT="$CKPT_DIR/../mixed_edod/step50000.ckpt"
TRAIN_DATA="${DATA_DIR}/dcc_50_mixed_edod_train.npz"
LABELS="${DATA_DIR}/beam_full_st_labels.npz"
BATCH_LABELS=128  # instances per batch for beam labeling

echo "============================================================"
echo "Phase 3a Beam Full Self-Training (11520 instances)"
echo "============================================================"

# ── Step 1: Beam 全量标签生成 ──
if [ -f "$LABELS" ]; then
    echo "  SKIP labels (exists: $LABELS)"
else
    echo "========== STEP 1: Full beam label generation =========="
    echo "  This will take ~2-3 hours for 11520 instances."
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

print('Loading model...')
params, _, _, model_config, _, _ = load_ckpt('$BASE_CKPT')
if model_config is None:
    model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
    model_config.encoder_input_dim = 7
model = model_config.construct_model()
if params is not None:
    model = nnx.merge(nnx.graphdef(model), params)
_MODEL_IN = int(model.init_proj.kernel.shape[0])

@jax.jit
def encode_fn(raw_features):
    raw_features = raw_features.at[..., :2].set(coord_normalize(raw_features[..., :2]))
    return model.encode(raw_features[..., :_MODEL_IN])

@jax.jit
def logit_fn(features, timestep):
    adjmat = jnp.zeros((1, 51, 51), dtype=jnp.int8)
    return model.decode(features, timestep, adjmat.astype(jnp.float32))

print('Loading training data...')
data = dict(np.load('${TRAIN_DATA}'))
N = data['coords'].shape[0]
cap = 50
tw_max = float(data['tw_end'].max())

new_routes = np.zeros_like(data['routes'])
new_costs = np.zeros(N, dtype=np.float32)
batch_size = $BATCH_LABELS
total_batches = (N + batch_size - 1) // batch_size

t0 = time.time()
for bi in range(total_batches):
    start = bi * batch_size
    end = min(start + batch_size, N)
    B = end - start

    features_list = [
        data['coords'][start:end],
        (data['demands'][start:end] / cap)[..., None],
        (data['tw_start'][start:end] / tw_max)[..., None],
        (data['tw_end'][start:end] / tw_max)[..., None],
        (data['temp_class'][start:end].astype(np.float32) / 2.0)[..., None],
    ]
    if 'reveal_time' in data:
        features_list.append(
            (data['reveal_time'][start:end].astype(np.float32) / tw_max)[..., None])
    raw_features = np.concatenate(features_list, axis=-1).astype(np.float32)

    feats = np.array(encode_fn(jnp.array(raw_features)))
    logits_all = np.array(logit_fn(jnp.array(feats), jnp.full(B, 0.5, dtype=jnp.float32)))

    for b in range(B):
        logits_b = logits_all[b]
        logits_b[:, 0] = -1e9
        np.fill_diagonal(logits_b, -1e9)

        searcher = ResourceBeamSearcher(
            data['coords'][start+b], data['tw_start'][start+b],
            data['tw_end'][start+b], data['service_time'][start+b],
            data['demands'][start+b], cap, 1.0, K=16, tw_margin=0.05)
        route, score = searcher.generate(logits_b, max_steps=200)

        if route is not None and len(route) > 2:
            pad = len(data['routes'][0])
            padded = np.pad(route, (0, max(0, pad - len(route))), constant_values=0)[:pad]
            new_routes[start+b, :len(padded)] = padded
            prev, cost = 0, 0.0
            for nd in route:
                if nd <= 0 or nd >= 51: continue
                cost += data['dist_mat'][start+b, prev, nd]
                prev = nd
            new_costs[start+b] = cost
        else:
            new_routes[start+b] = data['routes'][start+b]
            new_costs[start+b] = data['opt_costs'][start+b]

    if (bi+1) % 5 == 0 or bi == total_batches - 1:
        elapsed = time.time() - t0
        eta = elapsed / (bi+1) * (total_batches - bi - 1)
        print(f'  {end}/{N} ({(bi+1)/total_batches*100:.0f}%) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s | avg_cost={new_costs[start:end].mean():.2f}')

elapsed = time.time() - t0
print(f'Beam labels complete in {elapsed:.0f}s ({elapsed/3600:.1f}h). Avg cost: {new_costs[:N].mean():.2f}')
np.savez('${LABELS}', routes=new_routes, opt_costs=new_costs,
         coords=data['coords'], demands=data['demands'],
         tw_start=data['tw_start'], tw_end=data['tw_end'],
         service_time=data['service_time'], temp_class=data['temp_class'],
         reveal_time=data.get('reveal_time', np.zeros_like(data['demands'])),
         visible_mask=data.get('visible_mask', np.ones_like(data['demands'])),
         quality_loss=data.get('quality_loss', np.zeros_like(data['demands'], dtype=np.float32)),
         energy_mat=data.get('energy_mat', np.zeros((N, 51, 51), dtype=np.float32)),
         dist_mat=data.get('dist_mat', np.zeros((N, 51, 51), dtype=np.float32)))
print('Labels saved.')
"
fi

# ── Step 2: Fine-tune on full beam labels ──
echo ""
echo "========== STEP 2: Fine-tune on full beam labels =========="
if [ -f "${CKPT_DIR}/step50000.ckpt" ]; then
    echo "  SKIP (ckpt exists)"
else
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
        --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --encoder_input_dim 7 --peak_lr 5e-4 --batch_size 64 \
        --num_steps 50000 --save_interval 5000 \
        --data "$LABELS" --masking_mode spatio_temporal \
        --ckpt "$BASE_CKPT" \
        --logdir "${LOG_DIR}/beam_full_st" --savedir "$CKPT_DIR" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
fi

# ── Step 3: Evaluate on R1 EDoD=0.5 ──
echo ""
echo "========== STEP 3: Evaluate beam full ST model =========="
CKPT="${CKPT_DIR}/step50000.ckpt"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls "${CKPT_DIR}"/step*.ckpt 2>/dev/null | tail -1)
fi
echo "  Using checkpoint: $CKPT"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODER" \
    --capacity 50 --penalty 3. \
    --data "${DATA_DIR}/dcc_50_r1_edod05_test.npz" \
    --ckpt "$CKPT" \
    --keep_rate 0.3 --batch_size 8 --runs 1 --cycles 1 --sampling_steps 1 \
    --two_opt_steps 0 --seed 42 --gumbel_scale_factor 0. --threads_over_batches 1 \
    --enable_resource_decoder

echo ""
echo "=== DONE ==="
echo "Model: ${CKPT_DIR}"
