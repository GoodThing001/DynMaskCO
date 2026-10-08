#!/bin/bash
# S3-4-D3 冻结源码副本（第一步：创建不可变隔离源码树；烟测通过后、正式 40 天前执行）
# - 把 10 个封存文件按相对结构复制到 results/s3_4_d3_src_frozen（已存在则拒绝，绝不覆盖）；
# - 写入 SOURCE_MANIFEST.sha256（启动快照）与运行器 s3_4_d3_run_frozen.py；
# - 冻结树此后不再被任何同步/修改触碰。
# 注意：必须在 A3 全部批次落盘后执行（工作目录 scenario_saa.py 的 end-seal 复查会读同一路径，
# 中途替换会破坏在跑批次的 source_stable）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
DST=results/s3_4_d3_src_frozen
if [ -e "$DST" ]; then
  echo "REJECT: $DST already exists（冻结树不可覆盖；请先人工处置或换名）"
  exit 2
fi
mkdir -p $DST/scripts/evaluation $DST/scripts/simulation $DST/scripts/coldchain
cp scripts/evaluation/run_a1_step2_gate.py $DST/scripts/evaluation/
cp scripts/evaluation/run_identity.py $DST/scripts/evaluation/
cp scripts/evaluation/scenario_saa.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_energy_c0.py $DST/scripts/evaluation/
cp scripts/evaluation/coldchain_evaluator_a1.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_reserve.py $DST/scripts/evaluation/
cp scripts/evaluation/run_exp_encoder_v3.py $DST/scripts/evaluation/
cp scripts/simulation/strict_online_env.py $DST/scripts/simulation/
cp scripts/coldchain/coldchain_state.py $DST/scripts/coldchain/
cp scripts/coldchain/coldchain_contract.py $DST/scripts/coldchain/
cp tools_remote/tools_remote_s3_4_d3_run_frozen.py $DST/s3_4_d3_run_frozen.py
{
  echo "# s3_4_d3 frozen source snapshot $(date '+%F %T')"
  for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
           scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
           scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
           scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
           scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
    echo "$(sha256sum $DST/$f | cut -d' ' -f1)  $f"
  done
  echo "$(sha256sum $DST/s3_4_d3_run_frozen.py | cut -d' ' -f1)  s3_4_d3_run_frozen.py"
} > $DST/SOURCE_MANIFEST.sha256
chmod -R a-w $DST
echo "frozen tree: $DST"
cat $DST/SOURCE_MANIFEST.sha256
