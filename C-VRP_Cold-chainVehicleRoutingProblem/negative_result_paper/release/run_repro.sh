#!/bin/bash
# B 论文复现包：从零执行入口（Linux；在仓库扩展根目录运行）。
# 前置：把 SHA256SUMS.txt 列出的 9 个大文件放到对应路径并核对；本 release/ 位于
# negative_result_paper/release/，脚本/模块/结果 JSON 均为本包内快照。
#
# 用法（建议先建环境）：
#   conda env create -f negative_result_paper/release/environment.yml
#   conda activate maskco-b-repro
#   bash negative_result_paper/release/run_repro.sh <EXT_ROOT>
set -e
EXT=${1:-$(pwd)}
REL="$EXT/negative_result_paper/release"
WORK="$EXT/repro_work"
mkdir -p "$WORK"
echo "== stage v1 modules =="
cp "$REL/v1_modules/coldchain_state.py"      "$EXT/scripts/coldchain/coldchain_state.py.staged"
cp "$REL/v1_modules/coldchain_contract.py"   "$EXT/scripts/coldchain/coldchain_contract.py.staged"
cp "$REL/v1_modules/strict_online_env.py"    "$EXT/scripts/simulation/strict_online_env.py.staged"
cp "$REL/v1_modules/dynmaskco_cc_context.py" "$EXT/scripts/simulation/dynmaskco_cc_context.py.staged"
cp "$REL/v1_modules/cc_lns_replanner.py"     "$EXT/scripts/simulation/cc_lns_replanner.py.staged"
cp "$REL/v1_modules/mtrained_replanner.py"   "$EXT/scripts/simulation/mtrained_replanner.py.staged"
echo "(staged v1 modules as *.staged — 替换 .py 后运行，结束后还原)"
echo "== smoke: fixed-state 1 instance (cal, s42) =="
PYTHONPATH="$EXT/scripts" python "$REL/scripts/run_fixed_state_quality.py" \
  --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --split cal --max-instances 1 --n-samples 1 \
  --out "$WORK/fsq_smoke"
echo "== smoke: D1/D2 headroom census (train, 1 instance) =="
PYTHONPATH="$EXT/scripts" python "$REL/scripts/run_headroom_census.py" \
  --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
  --objective-profile results/o0cc/scale_v2/objective_profile.json \
  --split train --max-instances 1 \
  --out "$WORK/hc_smoke"
echo "== figures (machine-readable unified tables) =="
cd "$EXT/negative_result_paper"
python scripts/make_figures.py
echo "REPRO_SMOKE_DONE"
