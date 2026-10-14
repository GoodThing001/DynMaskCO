#!/bin/bash
# 正式批次 v2 · 追加腿：CO-enriched-ML（2 模式 × 2 档）
#
# 前置（必须全部满足）：
#   1) results/a1_c1fresh_reveal_40/gate.json 已产出（身份合格基线）；
#   2) F1 四条腿全部 ALLDONE（否则改 ordering_providers.py 会让在跑臂 source_stable=False）；
#   3) ordering_providers.py 已加注册表行：
#        'coenrich': ('CC_Compare/CO-enriched-ML/a1_accept_v1',
#                     'coenrich_provider', 'COEnrichedMLProvider'),
# 模式：nopchgs = 纯学习排序（0.02s/决策）；pchgs = +官方 PC-HGS 优先级生成器（≈1–2s/决策）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
COE_ABS=/home/hzeng/project/MASKCO-Main/CC_Compare/CO-enriched-ML/a1_accept_v1
CKPT=../CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1
PCHGS=$COE_ABS/build_pchgs/PCHGS
MARK=/tmp/xformal_v2/coenrich.done
mkdir -p "$OUT"; : > "$MARK"

if [ ! -f "$BASE" ]; then echo "FATAL: 缺基线 $BASE" | tee -a "$MARK"; exit 1; fi
if [ ! -d "$CKPT" ]; then echo "FATAL: 缺重训权重 $CKPT" | tee -a "$MARK"; exit 1; fi

run_one() {  # mode tier sfx
  local mode=$1 p=$2 sfx=$3
  local dest="$OUT/coenrich_${mode}_${sfx}"
  if [ -f "$dest/gate.json" ]; then
    echo "SKIP_done coenrich_${mode}_${sfx}" >> "$MARK"; return 0
  fi
  local envprefix=""
  if [ "$mode" = "pchgs" ]; then
    if [ ! -x "$PCHGS" ]; then
      echo "SKIP_pchgs_missing_exe $PCHGS" >> "$MARK"; return 0
    fi
    envprefix="COENRICH_PCHGS_EXE=$PCHGS"
  fi
  env $envprefix $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider coenrich --provider-ckpt "$CKPT" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 \
    --penalty "$p" --out "$dest"
  echo "DONE_coenrich_${mode}_${sfx} $?" >> "$MARK"
}

# 先跑纯排序（快），再跑 PC-HGS 模式（慢）。
# 2026-10-13 调整：**同模式的 2 档并行**（各 workers=6 ⇒ 12 核；机器 ~70 空闲核）。
# 依据：1 天探针实测 +PC-HGS 单决策 ≈4.2 s（13.6 min/天 ÷ 194 决策），远低于 10 s 预算，
# 且探针与 9-worker 基线同时跑仍 timeouts=0 ⇒ 并行把 +PC-HGS 段从 ~18 h 压到 ~9–10 h。
# 若某档出现 `mean_timeouts>0`，按 README §12 R3 处置（该模式记「超预算」，只留纯排序）。
run_pair() {  # mode
  local mode=$1
  run_one "$mode" "p_c=0" 0 &
  local pid0=$!
  run_one "$mode" "p_c=(5,10,15)" 51015 &
  local pid1=$!
  wait "$pid0" "$pid1"
}

run_pair nopchgs
run_pair pchgs
echo ALLDONE >> "$MARK"
