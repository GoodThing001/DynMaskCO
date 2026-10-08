#!/bin/bash
# A1 冻结源码副本（烟测通过后、正式 40 天前执行）：10 个封存文件 + a1_strong_controls.py +
# V 模型 npz + 运行器 a1_run_frozen.py。已存在则拒绝，绝不覆盖。
# 注意：必须在 A3 全部批次落盘后执行（工作目录源码 end-seal 复查约束）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
VCKPT="${1:-}"
[ -f "$VCKPT" ] || { echo "REJECT: 用法 $0 <v_model.npz>"; exit 2; }
DST=results/a1_src_frozen
if [ -e "$DST" ]; then
  echo "REJECT: $DST already exists（冻结树不可覆盖；请先人工处置或换名）"
  exit 2
fi
mkdir -p $DST/scripts/evaluation $DST/scripts/simulation $DST/scripts/coldchain $DST/models
cp scripts/evaluation/run_a1_step2_gate.py $DST/scripts/evaluation/
cp scripts/evaluation/run_identity.py $DST/scripts/evaluation/
cp scripts/evaluation/scenario_saa.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_energy_c0.py $DST/scripts/evaluation/
cp scripts/evaluation/coldchain_evaluator_a1.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_reserve.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_encoder_v3.py $DST/scripts/evaluation/
cp scripts/simulation/strict_online_env.py $DST/scripts/simulation/
cp scripts/simulation/a1_strong_controls.py $DST/scripts/simulation/
cp scripts/coldchain/coldchain_state.py $DST/scripts/coldchain/
cp scripts/coldchain/coldchain_contract.py $DST/scripts/coldchain/
cp "$VCKPT" $DST/models/a1_v_model.npz
cp tools_remote/tools_remote_a1_run_frozen.py $DST/a1_run_frozen.py
{
  echo "# a1 frozen source snapshot $(date '+%F %T')"
  for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
           scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
           scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
           scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
           scripts/simulation/a1_strong_controls.py \
           scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
    echo "$(sha256sum $DST/$f | cut -d' ' -f1)  $f"
  done
  echo "$(sha256sum $DST/models/a1_v_model.npz | cut -d' ' -f1)  models/a1_v_model.npz"
  echo "$(sha256sum $DST/a1_run_frozen.py | cut -d' ' -f1)  a1_run_frozen.py"
} > $DST/SOURCE_MANIFEST.sha256
chmod -R a-w $DST
echo "frozen tree: $DST"
cat $DST/SOURCE_MANIFEST.sha256
