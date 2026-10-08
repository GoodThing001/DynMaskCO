#!/bin/bash
# S3-4-D3 anytime 部分投票——2 天烟测（2026-10-03，注册件 S3-4-D3_anytime部分投票_正式规格预声明 §3）。
# 内容：同 seed 20260926、同 2 天、p_c=0、reveal、marginal、ls+incr、10s 下两批——
#   ON  = --anytime-vote（新语义：部分投票提交 + 先验早停）；
#   OFF = 同修订版旧语义（保计划+拒单+计次）。
# 判据（烟测阶段，非证据）：0 崩溃；OFF 与既有 v2_low OFF 批一致（旧路径无回归交叉确认）；
# ON 的 partial_commits/early_stops/timeouts 计数；ON vs OFF 逐臂逐日效用差（部分投票的
# 净方向只作线索）；T33 本地全绿为前置（launcher 不代跑本地测试）。
# 通过后按冻结链纪律新建 S3-4-D3 冻结源码树，正式 1×40 天三档只从冻结树启动。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=$(date +%m%d_%H%M%S)
OUT_ON=results/a1_s3_4_d3_smoke_on_$TS
OUT_OFF=results/a1_s3_4_d3_smoke_off_$TS
mkdir $OUT_ON || exit 2
mkdir $OUT_OFF || exit 2
{
  echo "launch: $(date '+%F %T')  S3-4-D3 smoke"
  echo "ON=$OUT_ON (anytime-vote)  OFF=$OUT_OFF (旧语义同修订版)"
  echo "前置：本地 T1-T33 ALL PASS（T33a-e 为 D3 新回归）；本脚本不代跑本地测试"
  sha256sum scripts/evaluation/scenario_saa.py scripts/evaluation/run_a1_step2_gate.py \
            scripts/tests/test_a1_saa_correctness.py
  uptime
} | tee $OUT_ON/resource.txt > $OUT_OFF/resource.txt

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --shadow-mode ls --incr-eval --anytime-vote --workers 3 --out $OUT_ON \
  > $OUT_ON/run.log 2>&1
echo "D3_SMOKE_ON_EXIT=$?" > $OUT_ON/run.exit

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --shadow-mode ls --incr-eval --workers 3 --out $OUT_OFF \
  > $OUT_OFF/run.log 2>&1
echo "D3_SMOKE_OFF_EXIT=$?" > $OUT_OFF/run.exit

$PY -c "
import json
on = json.load(open('$OUT_ON/gate.json', encoding='utf-8'))
off = json.load(open('$OUT_OFF/gate.json', encoding='utf-8'))
pon = on['gate']['p_c=0']['per_day']; poff = off['gate']['p_c=0']['per_day']
for arm in ('uncond_hist', 'cond_hist', 'explicit_feat'):
    ra = sorted(pon[arm], key=lambda x: x['i']); rb = sorted(poff[arm], key=lambda x: x['i'])
    uon = [r['utility'] for r in ra]; uoff = [r['utility'] for r in rb]
    diffs = [a - b for a, b in zip(uon, uoff)]
    print(arm, 'u_on=%s u_off=%s diff=%s' % (['%.2f' % x for x in uon],
                                             ['%.2f' % x for x in uoff],
                                             ['%.2f' % x for x in diffs]))
    print('  ON  to=%s partial=%s early=%s' % ([r['timeouts'] for r in ra],
                                               [r['partial_commits'] for r in ra],
                                               [r['early_stops'] for r in ra]))
    print('  OFF to=%s partial=%s early=%s' % ([r['timeouts'] for r in rb],
                                               [r['partial_commits'] for r in rb],
                                               [r['early_stops'] for r in rb]))
print('on identity source_stable=%s' % on['identity']['source_stable'])
print('off identity source_stable=%s' % off['identity']['source_stable'])
" | tee $OUT_ON/smoke_compare.txt
echo "D3 smoke done: ON=$OUT_ON OFF=$OUT_OFF"
