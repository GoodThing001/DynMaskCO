#!/bin/bash
# A3 在线比较启动器（v2，2026-10-02 按用户口径修正）。⚠️ 条件启用：A2 基线策略已锁定
# （results/A2_LOCK.json 存在，含 baseline_policy=reveal|density 与三种子 gate.json 路径），
# 且 200 日 s*=200 三模型已训练。启动前按《A3在线比较预声明》锁定臂表：
# A=cond_hist 基线（从步骤 2 gate.json 复用逐日数据）；B=MaskCO 单步；C=MaskCO 迭代；
# D=同架构随机。主比较 = maskco−cond_hist、δ₃=20。
# v2 修复：①基线策略显式参数（--baseline-policy reveal|density，无默认值）；
# ②无 A2_LOCK.json 即拒绝（不得在 A2 未锁定时跑 A3）；③驱动显式传 --future-policy
# 与基线策略一致（旧版默认 reveal 会与 density 基线错策略比较）。
# 用法：bash tools_remote/tools_remote_a3_online.sh --baseline-policy <reveal|density>
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0

POLICY=""
if [ "${1:-}" = "--baseline-policy" ]; then POLICY="${2:-}"; fi
case "$POLICY" in reveal|density) ;; *) echo "REJECT: 用法 $0 --baseline-policy reveal|density"; exit 2;; esac

LOCK=results/A2_LOCK.json
test -f $LOCK || { echo "REJECT: A2 基线未锁定（缺 $LOCK，先由复验裁决写入）"; exit 2; }
LOCKED=$($PY -c "import json;print(json.load(open('$LOCK',encoding='utf-8')).get('baseline_policy'))")
[ "$LOCKED" = "$POLICY" ] || { echo "REJECT: 请求策略 $POLICY 与 A2 锁定策略 $LOCKED 不一致"; exit 2; }

B=results/a1_a3
MB_B=$B/a3_train200_pretrained_s42/model.bin
MB_C=$B/a3_train200_partial_pretrained_s42/model.bin
MB_D=$B/a3_train200_random_cvrp_s42/model.bin
for f in $MB_B $MB_C $MB_D; do
  test -f $f || { echo "REJECT: 缺模型 $f（先跑 tools_remote_a3_retrain.sh）"; exit 2; }
done

TS=$(date +%m%d_%H%M%S)
run_one() {
  SEED=$1; ARM_LABEL=$2; MODEL=$3; EXTRA=$4
  BASE_JSON=$5   # 该 seed 的步骤 2 gate.json（含 cond_hist 等基线逐日数据）
  ARM=$( [ "$ARM_LABEL" = "D" ] && echo random_cvrp || echo pretrained )
  GPU=$( [ "$ARM_LABEL" = "C" ] && echo 1 || echo 0 )   # B/D→GPU0，C→GPU1（避免同卡争抢）
  OUT=results/a1_a3_online_${POLICY}_${ARM_LABEL}_${SEED}_40_$TS
  SESS=a3_${POLICY}_${ARM_LABEL}_${SEED}_$TS
  test -f $BASE_JSON || { echo "REJECT: 基线缺失 $BASE_JSON"; exit 2; }
  mkdir $OUT || { echo "REJECT: out dir exists $OUT"; exit 2; }
  tmux has-session -t $SESS 2>/dev/null && { echo "REJECT: session $SESS exists"; exit 2; }
  {
    echo "launch: $(date '+%F %T')  policy=$POLICY  arm=$ARM_LABEL($ARM)  seed=$SEED  gpu=$GPU  model=$MODEL"
    echo "baseline_from=$BASE_JSON   A2_LOCK=$LOCK"
    uptime
    echo "nproc: $(nproc)  workers: 2  time_limit: 10s"
  } > $OUT/resource.txt
  tmux new-session -d -s "$SESS" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
     CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_FLAGS=--xla_gpu_autotune_level=0 \
     $PY scripts/evaluation/run_a1_step3_compare.py \
       --model-bin $MODEL --arm $ARM --only maskco --baseline-from $BASE_JSON \
       --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
       --gate-seed $SEED --energy-pricing marginal --future-policy $POLICY \
       $EXTRA --workers 2 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
     echo \"$(date '+%F %T') $SESS done run=\$rc\""
}

# 基线路径全部读自 A2_LOCK.json（由复验裁决写入，人工不填占位符）
get_base() {
  SEED=$1
  $PY -c "import json;d=json.load(open('$LOCK',encoding='utf-8'));print(d['gate_json_by_seed'].get('$SEED',''))"
}

for SEED in 20260926 20260930 20261001; do
  BASE=$(get_base $SEED)
  [ -n "$BASE" ] || { echo "REJECT: A2_LOCK 缺 seed $SEED 的 gate.json 路径"; exit 2; }
  run_one $SEED B $MB_B "" "$BASE" || exit 2
  sleep 30   # 错峰：避免多批同时构建模型挤爆显存
  run_one $SEED C $MB_C "--iterative --remask-frac 0.5 --rounds 1" "$BASE" || exit 2
  sleep 30
  run_one $SEED D $MB_D "" "$BASE" || exit 2
  sleep 30
done
sleep 3
tmux ls
echo "launched: A3 online policy=$POLICY 3 seeds x B/C/D (ts=$TS)"
