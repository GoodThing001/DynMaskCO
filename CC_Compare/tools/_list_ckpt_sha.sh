#!/bin/bash
# 只读：列出 A-v1 各 learned provider 的 checkpoint 大小与 sha256（证据台账用）
set -u
cd /home/hzeng/project/MASKCO-Main/CC_Compare || exit 1
paths="AttentionModel/pretrained/cvrp_100/epoch-99.pt
DeepACO/pretrained/cvrp/cvrp100.pt
Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt
Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt
SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt
POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt
RouteFinder/checkpoints/100/rf-transformer.ckpt
CaDA/100/result/2024-1121-1355/checkpoint-300.pt
MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt
RRNCO/checkpoints/rcvrptw/epoch_199.ckpt
CO-enriched-ML/training/models/NeuralNetwork_samples-50_instances-15_runtime-3600_featureset-dynamicstatic_iteration-29"
echo "$paths" | while read -r p; do
  if [ -f "$p" ]; then
    sz=$(stat -c%s "$p")
    h=$(sha256sum "$p" | cut -c1-16)
    printf '%-95s %12s  %s\n' "$p" "$sz" "$h"
  else
    printf '%-95s %12s  %s\n' "$p" "MISSING" "-"
  fi
done
