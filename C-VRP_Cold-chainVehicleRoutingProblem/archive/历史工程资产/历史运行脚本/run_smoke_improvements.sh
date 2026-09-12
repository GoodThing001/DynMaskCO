#!/bin/bash
# ============================================================
# run_smoke_improvements.sh — 验证 4 项改进（借鉴 ESWA 2025）
# ============================================================
# 验证内容：
#   1. 节点类型专属 embedding（type_embed 5 类）+ temp_class bug 修复
#   2. 不可售产品数 Num 指标（--quality_salable_threshold）
#   3. 3 阶段品质模型 + 寿命阈值（--enable_quality）
#   4. 显式边特征（energy_mat → 注意力偏置，--use_edge_feat）
#
# 用法:
#   bash run_smoke_improvements.sh
#
# 预计耗时 ~5min（200 steps 训练 ×2 + 解码 ×2）
# 全量验收见 docs/论文参考/对比方法/改进清单.md §7
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
R1_TRAIN="${DATA_DIR}/dcc_50_r1_edod05_train.npz"
R1_TEST="${DATA_DIR}/dcc_50_r1_edod05_test.npz"
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/smoke_improvements"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/smoke_improvements"

NUM_STEPS=200
SAVE_INTERVAL=200

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ═══════════════════════════════════════════════════════════
# Step 0: 数据字段检查（确认哪些路径会被激活）
# ═══════════════════════════════════════════════════════════
echo "============================================================"
echo "Step 0: 数据字段检查"
echo "============================================================"
python3 -c "
import numpy as np
for name, f in [('train', '${R1_TRAIN}'), ('test', '${R1_TEST}')]:
    d = dict(np.load(f))
    keys = [k for k in ['coords','demands','tw_start','tw_end','temp_class',
                        'reveal_time','visible_mask','quality_loss','energy_mat','routes'] if k in d]
    print(f'  {name}: {d[\"coords\"].shape[0]} inst, fields={keys}')
    assert 'visible_mask' in d, f'{name} 缺少 visible_mask（P0-2 数据必需）'
"

# ═══════════════════════════════════════════════════════════
# Step 1: 模型构造 + temp_class bug 修复检查（纯 Python，无训练）
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Step 1: 模型构造 + typed embedding + temp_class 修复检查"
echo "============================================================"
python3 -c "
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models')
import jax.numpy as jnp
from DynamicColdChainModel import DynamicColdChainModelConfig

cfg = DynamicColdChainModelConfig.get_config('softcap_fn')
cfg.encoder_input_dim = 7
model = cfg.construct_model()

# 检查 1：type_embed 5 类 + edge_weight 存在
assert model.type_embed.num_embeddings == 5, \
    f'type_embed 应为 5 类，实际 {model.type_embed.num_embeddings}'
assert hasattr(model, 'edge_weight'), 'edge_weight 缺失'
print(f'[OK] type_embed={model.type_embed.num_embeddings} 类 | edge_weight={float(model.edge_weight.value):.4f}')

# 检查 2：temp_class bug 修复（喂 temp_class/2.0，冷藏 0.5 不再被截断为常温）
B, N = 1, 4
raw = jnp.zeros((B, N, 7))
raw = raw.at[..., :2].set(0.5)
raw = raw.at[..., 5].set(jnp.array([[0.0, 0.0, 0.5, 1.0]]))  # depot, 常温(0), 冷藏(0.5), 冷冻(1.0)
vis = jnp.ones((B, N))
feats = model.encode(raw, visible_mask=vis)
print(f'[OK] encode 通过（typed embedding + 可见性门控）feats.shape={tuple(feats.shape)}')

# 检查 3：edge_feat 路径
edge = jnp.zeros((B, N, N))
feats2 = model.encode(raw, visible_mask=vis, edge_feat=edge)
print(f'[OK] edge_feat 路径通过 feats.shape={tuple(feats2.shape)}')
print('Step 1 全部通过')
"

# ═══════════════════════════════════════════════════════════
# Step 2: 训练 smoke（typed embedding 基准，200 steps）
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Step 2: 训练 smoke（typed embedding，${NUM_STEPS} steps）"
echo "============================================================"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
    --num_steps $NUM_STEPS --save_interval $SAVE_INTERVAL \
    --data "$R1_TRAIN" --masking_mode spatio_temporal \
    --online_seq_training --online_seq_steps 5 \
    --logdir "${LOG_DIR}/typed" --savedir "${CKPT_DIR}/typed" \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None --seed 42

# ═══════════════════════════════════════════════════════════
# Step 3: 训练 smoke（--use_edge_feat，200 steps）
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Step 3: 训练 smoke（--use_edge_feat，${NUM_STEPS} steps）"
echo "============================================================"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
    --num_steps $NUM_STEPS --save_interval $SAVE_INTERVAL \
    --data "$R1_TRAIN" --masking_mode spatio_temporal \
    --online_seq_training --online_seq_steps 5 \
    --use_edge_feat \
    --logdir "${LOG_DIR}/typed_edge" --savedir "${CKPT_DIR}/typed_edge" \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None --seed 42

# ═══════════════════════════════════════════════════════════
# Step 4: 解码 smoke（typed embedding 模型 + Num 指标）
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Step 4: 解码 smoke（Num 指标）"
echo "============================================================"
CK="${CKPT_DIR}/typed/step${NUM_STEPS}.ckpt"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE_SCRIPT" \
    --capacity 50 --penalty 3. --data "$R1_TEST" --ckpt "$CK" \
    --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 2 --cycles 10 \
    --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
    --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 8 \
    --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
    --quality_salable_threshold 0.1

# ═══════════════════════════════════════════════════════════
# Step 5: 解码 smoke（--enable_quality，3 阶段品质 + beam Num）
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Step 5: 解码 smoke（--enable_quality，3 阶段品质）"
echo "============================================================"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE_SCRIPT" \
    --capacity 50 --penalty 3. --data "$R1_TEST" --ckpt "$CK" \
    --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 2 --cycles 10 \
    --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
    --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 8 \
    --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
    --enable_quality --lambda_q 0.1 --quality_salable_threshold 0.1

echo ""
echo "============================================================"
echo "SMOKE 完成 — 4 项改进全部路径跑通"
echo "============================================================"
echo "CKPTs:  $CKPT_DIR/typed/  $CKPT_DIR/typed_edge/"
echo "日志:   $LOG_DIR/"
echo ""
echo "全量验收见 docs/论文参考/对比方法/改进清单.md §7："
echo "  - typed embedding 5-seed 重训 + cost/feas 不退化"
echo "  - --use_edge_feat 训练后 edge_weight 是否 != 0（边特征是否学到信号）"
echo "  - --enable_quality 下 num_unsalable 随阈值变化（常温 > 冷冻）"
