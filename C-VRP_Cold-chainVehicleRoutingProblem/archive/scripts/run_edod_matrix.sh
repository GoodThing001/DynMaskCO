#!/bin/bash
# EDoD 全矩阵评估 (R1/C1/RC1 × EDoD 0.2/0.5/0.8)
# 用法: bash analysis/run_edod_matrix.sh

cd /home/hzeng/project/MASKCO-Main/
source /home/hzeng/envs/MASKCO_env/bin/activate

RESULTS_FILE="C-VRP_Cold-chainVehicleRoutingProblem/analysis/edod_results.txt"
> $RESULTS_FILE

declare -A CKPTS
CKPTS["r1_02"]="phaseb_r1_edod02/step50000.ckpt"
CKPTS["r1_05"]="st_round2/step5000.ckpt"
CKPTS["r1_08"]="phaseb_r1_edod08/step50000.ckpt"
CKPTS["c1_02"]="phaseb_c1_edod02/step50000.ckpt"
CKPTS["c1_05"]="phaseb_c1_edod05/step50000.ckpt"
CKPTS["c1_08"]="phaseb_c1_edod08/step50000.ckpt"
CKPTS["rc1_02"]="phaseb_rc1_edod02/step50000.ckpt"
CKPTS["rc1_05"]="phaseb_rc1_edod05/step50000.ckpt"
CKPTS["rc1_08"]="phaseb_rc1_edod08/step50000.ckpt"

for type in r1 c1 rc1; do
  for edod in 02 05 08; do
    key="${type}_${edod}"
    ckpt="${CKPTS[$key]}"
    data="C-VRP_Cold-chainVehicleRoutingProblem/data/dcc_50_${type}_edod${edod}_test.npz"

    echo "===== ${type^^} EDoD=0.${edod} =====" | tee -a $RESULTS_FILE

    output=$(CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u "C-VRP_Cold-chainVehicleRoutingProblem/decoding/cvrptw.py" \
        --capacity 50 --penalty 3. \
        --data "$data" \
        --ckpt "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/$ckpt" \
        --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
        --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
        --threads_over_batches 1 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1)

    # 提取指标
    cost=$(echo "$output" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    feas=$(echo "$output" | grep "TW feas rate:" | tail -1 | awk '{print $NF}' | tr -d '%')
    viol=$(echo "$output" | grep "Avg TW viol/inst:" | tail -1 | awk '{print $NF}')
    gap=$(echo "$output" | grep "Gap:" | tail -1 | awk '{print $NF}' | tr -d '%')

    echo "${type^^}|0.${edod}|${feas}|${cost}|${viol}|${gap}" >> $RESULTS_FILE
    echo "  feas=${feas}% cost=${cost} viol=${viol} gap=${gap}%" | tee -a $RESULTS_FILE
  done
done

# 汇总表
echo ""
echo "============================================================================"
echo "  EDoD FULL MATRIX SUMMARY"
echo "============================================================================"
printf "%-6s | %18s | %18s | %18s\n" "Type" "EDoD=0.2" "EDoD=0.5" "EDoD=0.8"
echo "-------|--------------------|--------------------|--------------------"

for type in r1 c1 rc1; do
  printf "%-6s |" "${type^^}"
  for edod in 02 05 08; do
    line=$(grep "^${type^^}|0.${edod}|" $RESULTS_FILE)
    feas=$(echo "$line" | cut -d'|' -f3)
    gap=$(echo "$line" | cut -d'|' -f6)
    printf " %5s%% g=%+6s%% |" "$feas" "$gap"
  done
  echo ""
done
echo "============================================================================"
