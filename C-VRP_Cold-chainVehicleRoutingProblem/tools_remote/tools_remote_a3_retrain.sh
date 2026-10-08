#!/bin/bash
# A3 正式 200 日重训启动器（2026-10-01 预写）。⚠️ 条件启用：仅当 rr3 信息门 formal 通过
# （A2 重立）后才可运行；否则本脚本只作准备件。
# 训练规则 = 已登记《A3正式训练规则预声明_2026-10-01》：200 训练历史日、s*=200（前向
# 200 步窗口、登记后不回调）、pretrained seed42 dim128 m_max200 batch8 lr1e-3 AdamW；
# 同架构 random_cvrp 同 s*；C 臂 partial 谱系同 B（同 s*、同 200 天）。
# 谱系封存：model.bin + config.json（含源码 hash/步数/池口径）由训练器自动写入。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_a3
mkdir -p $B
STEPS=200

run_train() {
  NAME=$1; ARM=$2; EXTRA=$3
  if [ -d $B/$NAME ]; then
    echo "REJECT: 已存在 $B/$NAME（禁止覆盖，谱系唯一）"; return 1
  fi
  $PY scripts/training/train_maskco_scenario.py \
    --arm $ARM --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --train-instances 200 --n-orders 200 --dim 128 --m-max 200 \
    --steps $STEPS --batch-size 8 --seed 42 --save-every 50 $EXTRA \
    --out $B/$NAME > $B/${NAME}.log 2>&1
  echo "$NAME EXIT=$?"
}

# B 臂：clock 全掩码 pretrained（主模型）
run_train a3_train200_pretrained_s42 pretrained "" || exit 2
# C 臂：partial 训练谱系（迭代模式用）
run_train a3_train200_partial_pretrained_s42 pretrained "--partial-mask --p-max 0.8" || exit 2
# D 臂：同架构随机对照
run_train a3_train200_random_cvrp_s42 random_cvrp "" || exit 2
echo "A3_RETRAIN_DONE"
