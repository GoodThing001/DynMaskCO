#!/bin/bash
# F1 v2 四条腿的起飞前检查（只读）：
#   1) bash -n 语法检查
#   2) 各 provider 的 checkpoint 路径是否存在（脚本里是从扩展根出发的相对路径）
#   3) provider 名是否都在 ordering_providers._REGISTRY 里
#   4) 输出根/基线路径是否就绪
# 目的：避免基线跑完 7h 后某条腿因路径/语法错误立刻失败。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python

echo "===== 1) 语法检查 ====="
for f in ../CC_Compare/_formal_v2_launch.sh ../CC_Compare/_formal_v2_gpu.sh \
         ../CC_Compare/_formal_v2_cpuA.sh ../CC_Compare/_formal_v2_cpuB.sh \
         ../CC_Compare/_formal_v2_cpuC.sh ../CC_Compare/_formal_v2_after.sh \
         ../CC_Compare/_formal_v2_coenrich.sh ../CC_Compare/_formal_v2_mapt.sh; do
  if bash -n "$f" 2>/dev/null; then echo "  OK   $f"; else echo "  FAIL $f"; bash -n "$f"; fi
done

echo "===== 2) checkpoint 路径 ====="
for c in ../CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt \
         ../CC_Compare/DeepACO/pretrained/cvrp/cvrp100.pt \
         ../CC_Compare/Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt \
         ../CC_Compare/POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt \
         ../CC_Compare/Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt \
         ../CC_Compare/SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt \
         ../CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt \
         ../CC_Compare/RouteFinder/checkpoints/100/rf-transformer.ckpt \
         ../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt \
         ../CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt \
         ../CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt; do
  if [ -f "$c" ]; then echo "  OK   $(stat -c%s "$c") B  $c"; else echo "  MISS $c"; fi
done

echo "===== 3) provider 注册表 ====="
$PY - <<'PYEOF'
import sys
sys.path.insert(0, 'scripts/evaluation')
import ordering_providers as op
need = ['routefinder', 'cada', 'mvmoe', 'pomo', 'attention', 'symnco',
        'omnivrp', 'sgbs', 'deepaco', 'lih']
have = set(op._REGISTRY)
print('  registry:', sorted(have))
missing = [n for n in need if n not in have]
print('  missing :', missing if missing else '(none)')
extra = sorted(have - set(need))
print('  extra   :', extra)
PYEOF

echo "===== 4) 路径与基线 ====="
[ -d results/a1_ext_formal2 ] && echo "  OK   results/a1_ext_formal2 已存在" || echo "  NOTE results/a1_ext_formal2 尚未创建（首臂起飞时创建）"
[ -f results/a1_c1fresh_reveal_40/gate.json ] && echo "  OK   基线已落盘" || echo "  WAIT 基线仍在跑（四条腿尚未起飞，正常）"
echo "===== 预检结束 ====="
