#!/bin/bash
# S3-4 廉价影子搜索——2 天烟测（2026-10-02，规格《S3-4廉价影子搜索_正式规格预声明》§3）。
# 内容：同 seed 20260926、同 2 天、p_c=0、reveal、marginal 下两批——
#   A = --shadow-mode ls --incr-eval（新）；B = --shadow-mode ls（旧）。
# 判据（烟测阶段，非证据）：0 崩溃；A/B 逐日效用逐位一致（等价性端到端验证）；超时观察
# （S3-4 目标是 0 超时，正式判据在 40 天批次）。烟测从工作目录跑并记录双批源码 hash；
# 通过后按冻结链纪律新建 S3-4 冻结源码树，正式 1×40 天三档只从冻结树启动。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=$(date +%m%d_%H%M%S)
OUT_A=results/a1_s3_4_smoke_incr_$TS
OUT_B=results/a1_s3_4_smoke_old_$TS
mkdir $OUT_A || exit 2
mkdir $OUT_B || exit 2
{
  echo "launch: $(date '+%F %T')  S3-4 smoke"
  echo "A=$OUT_A (ls+incr)  B=$OUT_B (ls old)"
  sha256sum scripts/evaluation/scenario_saa.py scripts/evaluation/run_a1_step2_gate.py
  uptime
} | tee $OUT_A/resource.txt > $OUT_B/resource.txt

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --shadow-mode ls --incr-eval --workers 3 --out $OUT_A > $OUT_A/run.log 2>&1
echo "SMOKE_INCR_EXIT=$?" > $OUT_A/run.exit

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --shadow-mode ls --workers 3 --out $OUT_B > $OUT_B/run.log 2>&1
echo "SMOKE_OLD_EXIT=$?" > $OUT_B/run.exit

$PY -c "
import json
a = json.load(open('$OUT_A/gate.json', encoding='utf-8'))
b = json.load(open('$OUT_B/gate.json', encoding='utf-8'))
pa = a['gate']['p_c=0']['per_day']; pb = b['gate']['p_c=0']['per_day']
same = True
for arm in ('uncond_hist', 'cond_hist', 'explicit_feat'):
    ua = [r['utility'] for r in sorted(pa[arm], key=lambda x: x['i'])]
    ub = [r['utility'] for r in sorted(pb[arm], key=lambda x: x['i'])]
    ok = all(abs(x - y) < 1e-6 for x, y in zip(ua, ub))
    same = same and ok
    ta = [r['timeouts'] for r in pa[arm]]; tb = [r['timeouts'] for r in pb[arm]]
    print(arm, 'utilities_equal=%s' % ok, 'to_incr=%s to_old=%s' % (ta, tb))
print('SMOKE_EQUALITY=%s' % same)
" > $OUT_A/smoke_compare.txt
cat $OUT_A/smoke_compare.txt
echo "smoke done: A=$OUT_A B=$OUT_B"
