#!/bin/bash
# ============================================================================
# A-v1 对比方法「正式批次 v2」（2026-10-13）
#
# 目的：产出一份**同一版源码 + 身份可裁决**的正式总表。
# 背景（见 CC_Compare/results/a1_compare_table/README.md）：
#   - 09-30 正式臂的配对基线取自 results/a1_step2_gate_c1_20260926（无 identity 块）→
#     落盘 _pairing_identity.identity_verified=false；
#   - 之后 scenario_saa.py / run_a1_external_accept.py / rrnco_accept_replanner.py 漂移，
#     且臂运行时加载的旧版本未留存 → 无法事后补配对；
#   - 已复验：当前代码重算 contract_hash / data_sha256 / data_meta 与旧臂记录逐字节相同
#     （实例与 C0 物理未变），因此重跑 = 同协议、只是把身份链补全。
#
# 结构：
#   0) 新 C1 基线（reveal 策略，3 臂 × 3 档，40 天 seed 20260926）→ 提供 identity 合格的配对基线
#   1) GPU 腿：RouteFinder / CaDA / MVMoE（GPU1 顺序）+ RRNCO（GPU0）
#   2) CPU 腿 A：attention / deepaco / omnivrp / pomo / symnco / sgbs
#   3) CPU 腿 B：lih（单跑，最慢）
#   4) CPU 腿 C：OR-Tools / PyVRP + 规则臂 myopic / valuee-λ2 / valuee-λ3
#
# 启动（服务器仓库根）：bash CC_Compare/_formal_v2_launch.sh
# 进度：/tmp/xformal_v2/{base,gpu,cpuA,cpuB,cpuC}.done
# ============================================================================
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1

PY=/home/hzeng/envs/cc_compare/bin/python
OUTROOT=results/a1_ext_formal2
BASE=results/a1_c1fresh_reveal_40/gate.json
RUNDIR=/tmp/xformal_v2
mkdir -p "$RUNDIR" "$OUTROOT"

echo "== A-v1 正式批次 v2 启动 ==" 
date -u

# ---------------------------------------------------------------- 0) C1 基线
echo "[base] 新 C1 基线（reveal，3 臂 × 3 档，workers=9）..."
$PY scripts/evaluation/run_a1_step2_gate.py \
    --train-instances 200 --gate-instances 40 --n-orders 200 \
    --capacity 50.0 --num-vehicles 15 --rho 0.60 --K 10 \
    --time-limit 10.0 --k-neighbors 10 \
    --gate-seed 20260926 --shadow-mode greedy --energy-pricing marginal \
    --future-policy reveal --workers 9 \
    --out results/a1_c1fresh_reveal_40
echo "DONE_base $?" >> "$RUNDIR/base.done"
if [ ! -f "$BASE" ]; then
  echo "FATAL: 基线未产出 $BASE" | tee -a "$RUNDIR/base.done"
  exit 1
fi
python - "$BASE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
ident = d.get('identity') or {}
print('[base] identity present:', bool(ident),
      'source_stable:', ident.get('source_stable'),
      'contract:', str(ident.get('contract_sha256'))[:16],
      'data:', str(ident.get('data_sha256'))[:16])
for pname, blk in d['gate'].items():
    for an, ab in blk['arms'].items():
        u = ab.get('utility_all_days') or {}
        print('   %-16s %-16s util=%s to/day=%.2f' % (
            pname, an,
            ('%.2f' % u['mean']) if u.get('mean') is not None else 'n/a',
            ab.get('mean_timeouts', 0.0)))
PYEOF

# ---------------------------------------------------------------- 启动 4 条腿
# ⚠️ 必须用**绝对路径**：本脚本第 25 行已 cd 到扩展根，而扩展根下**没有** CC_Compare 目录
# （CC_Compare 在仓库根）。2026-10-13 实际踩到：写成相对路径 `bash CC_Compare/_formal_v2_*.sh`
# 会让 4 个 tmux 会话立刻 "No such file or directory" 退出 —— 会话消失、无任何日志、
# 而 `base.done` 已写、launch.log 还打印了「已启动 tmux」，看起来像已启动。现同时加逐腿日志。
LEGS=/home/hzeng/project/MASKCO-Main/CC_Compare
LOGD=/tmp/xformal_v2
tmux new -d -s xf2_gpu  "bash $LEGS/_formal_v2_gpu.sh  > $LOGD/gpu.log  2>&1"
tmux new -d -s xf2_cpuA "bash $LEGS/_formal_v2_cpuA.sh > $LOGD/cpuA.log 2>&1"
tmux new -d -s xf2_cpuB "bash $LEGS/_formal_v2_cpuB.sh > $LOGD/cpuB.log 2>&1"
tmux new -d -s xf2_cpuC "bash $LEGS/_formal_v2_cpuC.sh > $LOGD/cpuC.log 2>&1"
echo "已启动 tmux: xf2_gpu / xf2_cpuA / xf2_cpuB / xf2_cpuC（日志 $LOGD/{gpu,cpuA,cpuB,cpuC}.log）"
tmux ls
