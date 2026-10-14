#!/bin/bash
# 1 天小样探针：CO-enriched 的 **+PC-HGS 模式** 是否能在 10 s/决策预算内跑
# （纯排序模式已实测 0.02 s/次；PC-HGS 因缓存键含 per-vehicle 几何 ⇒ 每车都可能 miss）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
COE_ABS=/home/hzeng/project/MASKCO-Main/CC_Compare/CO-enriched-ML/a1_accept_v1
OUT=results/a1_ext_formal2_wiretest
LOG=/tmp/xformal_v2/coenrich_pchgs_probe.log
mkdir -p "$OUT"
{
  echo "== probe start $(date -u) =="
  COENRICH_PCHGS_EXE="$COE_ABS/build_pchgs/PCHGS" \
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
      --provider coenrich --provider-ckpt ../CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1 \
      --provider-device cpu --baseline-from results/a1_c1rr2_reveal_40/gate.json \
      --dev-instances 1 --workers 6 --penalty "p_c=0" --out "$OUT/coenrich_pchgs_1d"
  echo "rc=$?"
  $PY - <<'PYEOF'
import json, os
p='results/a1_ext_formal2_wiretest/coenrich_pchgs_1d/gate.json'
if not os.path.exists(p):
    print('NO GATE'); raise SystemExit
d=json.load(open(p,encoding='utf-8'))
for yr,yb in d['results'].items():
    m=yb.get('solver_meta') or {}
    print('solver_meta: n_solves=%s n_fail=%s max_decide_s=%s mean_solve_s=%s' % (
        m.get('n_solves'), m.get('n_fail'), m.get('max_decide_s'), m.get('mean_solve_time_s')))
    print('reject_reasons:', m.get('reject_reasons'))
    for pname,blk in yb.items():
        if pname.startswith('_'): continue
        for an,ab in blk.get('arms',{}).items():
            if an!='ordering_accept': continue
            print('  %s util=%s timeouts=%s hfr=%s' % (pname,
                (ab.get('utility_all_days') or {}).get('mean'), ab.get('mean_timeouts'),
                ab.get('hard_feasible_rate')))
PYEOF
  echo "== probe done $(date -u) =="
} > "$LOG" 2>&1
