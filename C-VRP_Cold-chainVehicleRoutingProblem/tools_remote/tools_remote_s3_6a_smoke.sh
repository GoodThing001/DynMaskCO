#!/bin/bash
# S3-6a 软 k-NN 2 天烟测（2026-10-03 注册件 §3）。⚠️ 启用条件：S3-4 线（含 D3）已按序裁决。
# 前置：① A3 全部落盘且新代码已上传；② 服务器 T37 绿。
# 内容：同 seed 20260926、2 天、p_c=0、density、marginal、greedy、10s，四臂同批
# （uncond/cond 硬 k-NN/explicit/soft_knn），soft_knn 带宽 = 训练日 h_med（mult=1 主值，
# 诊断另附 0.5/2.0 只报权重分布不跑门）。判据（烟测）：0 崩溃 + n_eff/权重分布合理 + 超时观察。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=$(date +%m%d_%H%M%S)
OUT=results/a1_s3_6a_smoke_$TS
mkdir $OUT || exit 2
{
  echo "launch: $(date '+%F %T')  S3-6a smoke"
  echo "arms: uncond_hist,cond_hist,explicit_feat,soft_knn(mult=1.0)"
  sha256sum scripts/evaluation/scenario_saa.py scripts/evaluation/run_a1_step2_gate.py
  uptime
} > $OUT/resource.txt

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --future-policy density --shadow-mode greedy \
  --arms uncond_hist,cond_hist,explicit_feat,soft_knn \
  --softknn-mult 1.0 --workers 4 --out $OUT \
  > $OUT/run.log 2>&1
echo "S3_6A_SMOKE_EXIT=$?" > $OUT/run.exit

$PY -c "
import json
d = json.load(open('$OUT/gate.json', encoding='utf-8'))
blk = d['gate']['p_c=0']
for arm in ('uncond_hist', 'cond_hist', 'explicit_feat', 'soft_knn'):
    s = blk['arms'][arm]
    rows = blk['per_day'][arm]
    print(arm, 'u=%s to=%s hard=%.2f' % (
        ['%.1f' % r['utility'] for r in rows],
        [r['timeouts'] for r in rows], s['hard_feasible_rate']))
print('soft_knn_minus_cond_hist:', blk.get('soft_knn_minus_cond_hist', {}).get('mean'))
print('cond_minus_uncond:', blk['main_comparison']['cond_minus_uncond']['mean'])
print('identity.source_stable:', d['identity']['source_stable'])
" | tee $OUT/smoke_summary.txt
echo "S3-6a smoke done: $OUT"
