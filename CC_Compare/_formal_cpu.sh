#!/bin/bash
# 正式批次 CPU 腿（2026-09-30 修复版代码 sha 2c4dee57 + fixed driver）：
# 轻量方法按官方空间定档（各 RESULTS.md 探针依据），顺序跑，workers 6。
# 前置：xformal_gpu + xformal_gpu2 完成后再启动（避免拖慢 GPU 臂协调器）。
# 启动：tmux new -d -s xformal_cpu 'bash CC_Compare/_formal_cpu.sh'
# 标记：/tmp/xformal_cpu.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xformal_cpu.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"
run() {  # name ckpt extra_args...
  local name=$1 ckpt=$2; shift 2
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider "$name" --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 "$@" \
    --penalty "p_c=0" --out "results/a1_ext_extra/${name}_0_f"
  echo "DONE_${name}_0 $?" >> "$MARK"
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider "$name" --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 "$@" \
    --penalty "p_c=(5,10,15)" --out "results/a1_ext_extra/${name}_51015_f"
  echo "DONE_${name}_51015 $?" >> "$MARK"
}
run attention ../CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt
run deepaco   ../CC_Compare/DeepACO/pretrained/cvrp/cvrp100.pt
run omnivrp   ../CC_Compare/Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt
run lih       ../CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt
run sgbs      ../CC_Compare/SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
run symnco    ../CC_Compare/Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt
run pomo      ../CC_Compare/POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt
echo ALLDONE >> "$MARK"
