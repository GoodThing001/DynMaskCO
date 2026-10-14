#!/bin/bash
# ============================================================================
# 1003 GPU 腿**回落脚本**（手动执行，仅当 3080 上某方法起不来时使用）
#
# 背景：2026-10-13 用户指示把 GPU 腿整体搬到 3080，`_formal_v2_gpu.sh` 已改为只写 ALLDONE
# （幂等占位）。若 3080 侧因 rl4co/torchrl 依赖装不上而无法运行某方法，用本脚本在 1003
# 的（被他人占 ~89% 的）GPU 上补跑该方法的臂 —— 命令与迁移前的原始 GPU 腿完全一致。
#
# 用法：bash CC_Compare/_gpu_leg_1003_fallback.sh [method ...]
#   不传参数 = 跑全部 4 个方法；也可只跑例如： bash ... routefinder cada
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
MARK=/tmp/xformal_v2/gpu_1003_fallback.done
mkdir -p "$OUT"; : > "$MARK"

if [ ! -f "$BASE" ]; then echo "FATAL: 缺基线 $BASE" | tee -a "$MARK"; exit 1; fi

RFP=../CC_Compare/RouteFinder/checkpoints/100/rf-transformer.ckpt
CADA=../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt
MVM=../CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt
RRC=../CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt

gpu_seq() {  # dev tag ckpt [solver]
  local dev=$1 tag=$2 ckpt=$3 kind=${4:-ordering}
  for tier in "p_c=0:0" "p_c=(5,10,15):51015"; do
    local p="${tier%%:*}" sfx="${tier##*:}"
    if [ -f "$OUT/${tag}_${sfx}/gate.json" ]; then
      echo "SKIP_done ${tag}_${sfx}" >> "$MARK"; continue
    fi
    if [ "$kind" = "rrnco" ]; then
      CUDA_VISIBLE_DEVICES=$dev $PY scripts/evaluation/run_a1_external_accept.py \
        --solver rrnco --rrnco-ckpt "$ckpt" --rrnco-device cuda \
        --baseline-from "$BASE" --dev-instances 40 --workers 4 \
        --penalty "$p" --out "$OUT/${tag}_${sfx}"
    else
      CUDA_VISIBLE_DEVICES=$dev $PY scripts/evaluation/run_a1_external_accept.py \
        --solver ordering --provider "$tag" --provider-ckpt "$ckpt" \
        --provider-device cuda --baseline-from "$BASE" --dev-instances 40 \
        --workers 4 --penalty "$p" --out "$OUT/${tag}_${sfx}"
    fi
    echo "DONE_${tag}_${sfx} $?" >> "$MARK"
  done
}

WANT=("$@")
if [ ${#WANT[@]} -eq 0 ]; then WANT=(routefinder cada mvmoe rrnco); fi
has() { for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

# GPU1：routefinder → cada → mvmoe（顺序）
(
  has routefinder && gpu_seq 1 routefinder "$RFP"
  has cada        && gpu_seq 1 cada        "$CADA"
  has mvmoe       && gpu_seq 1 mvmoe       "$MVM"
  echo ALLDONE_G1 >> "$MARK"
) &
# GPU0：rrnco
( has rrnco && gpu_seq 0 rrnco "$RRC" rrnco; echo ALLDONE_G0 >> "$MARK" ) &
wait
echo ALLDONE >> "$MARK"
