#!/bin/bash
# A1 两臂 2 天烟测（2026-10-03）。前置：① A3 全部落盘后新代码已上传；② 服务器 T 全套绿；
# ③ V 正式训练完成（--a1-v-ckpt 指向 v_model.npz）。内容：同 seed 20260926、2 天、p_c=0、
# density、marginal、greedy、10s，五臂同批（uncond/cond/explicit + a1_consensus/a1_rollout）。
# 判据（烟测阶段，非证据）：0 崩溃；hard=1.0；A1 臂 SAA 完成率/超时报告；逐臂效用只作线索。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
VCKPT="${1:-}"
[ -f "$VCKPT" ] || { echo "REJECT: 用法 $0 <v_model.npz 绝对或相对路径>"; exit 2; }
TS=$(date +%m%d_%H%M%S)
OUT=results/a1_a1_smoke_$TS
mkdir $OUT || exit 2
{
  echo "launch: $(date '+%F %T')  A1 smoke"
  echo "arms: uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout"
  echo "config: density marginal greedy 10s K=10 v_ckpt=$VCKPT"
  sha256sum $VCKPT scripts/simulation/a1_strong_controls.py \
            scripts/evaluation/scenario_saa.py scripts/evaluation/run_a1_step2_gate.py
  uptime
} > $OUT/resource.txt

$PY scripts/evaluation/run_a1_step2_gate.py \
  --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
  --future-policy density --shadow-mode greedy \
  --arms uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout \
  --a1-v-ckpt $VCKPT --a1-h 2.0 --workers 5 --out $OUT \
  > $OUT/run.log 2>&1
echo "A1_SMOKE_EXIT=$?" > $OUT/run.exit

$PY -c "
import json
d = json.load(open('$OUT/gate.json', encoding='utf-8'))
blk = d['gate']['p_c=0']
for arm in ('uncond_hist', 'cond_hist', 'explicit_feat', 'a1_consensus', 'a1_rollout'):
    s = blk['arms'][arm]
    rows = blk['per_day'][arm]
    print(arm, 'u=%s to=%s hard=%.2f scr=%s' % (
        ['%.1f' % r['utility'] for r in rows],
        [r['timeouts'] for r in rows], s['hard_feasible_rate'],
        ('%.2f' % s['mean_saa_completion_rate'])
        if s.get('mean_saa_completion_rate') is not None else 'None'))
print('a1_consensus_minus_cond_hist:', blk.get('a1_consensus_minus_cond_hist', {}).get('mean'))
print('a1_rollout_minus_cond_hist:', blk.get('a1_rollout_minus_cond_hist', {}).get('mean'))
print('identity.source_stable:', d['identity']['source_stable'])
" | tee $OUT/smoke_summary.txt
echo "A1 smoke done: $OUT"
