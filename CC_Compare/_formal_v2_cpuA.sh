#!/bin/bash
# A-v1 正式批次 v2 · CPU 腿 A（轻量方法，workers=6，顺序）
# attention → deepaco → omnivrp → pomo → symnco → sgbs
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
MARK=/tmp/xformal_v2/cpuA.done
mkdir -p "$OUT"; : > "$MARK"

run() {  # name ckpt
  local name=$1 ckpt=$2
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider "$name" --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 \
    --penalty "p_c=0" --out "$OUT/${name}_0"
  echo "DONE_${name}_0 $?" >> "$MARK"
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider "$name" --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 \
    --penalty "p_c=(5,10,15)" --out "$OUT/${name}_51015"
  echo "DONE_${name}_51015 $?" >> "$MARK"
}

run attention ../CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt
run deepaco   ../CC_Compare/DeepACO/pretrained/cvrp/cvrp100.pt
run omnivrp   ../CC_Compare/Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt
run pomo      ../CC_Compare/POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt
run symnco    ../CC_Compare/Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt
run sgbs      ../CC_Compare/SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
echo ALLDONE >> "$MARK"
