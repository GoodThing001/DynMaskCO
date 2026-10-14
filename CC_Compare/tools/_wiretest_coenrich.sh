#!/bin/bash
# 追加 provider 的「真驱动」集成测试（1 天小样），在 F1 四条腿起飞**之前**排除接线故障。
#
# 为什么现在可以改 ordering_providers.py：
#   - 正在跑的 F1 基线封存的是 SOURCE_FILES_STEP2，**不含** ordering_providers.py；
#   - F1 四条腿尚未起飞，它们起飞时才封存该文件（此后不再改 → source_stable 仍为 True）；
#   - 该文件也不在 run_identity.SHARED_SOURCE_FILES 里，故不影响与基线的配对。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1rr2_reveal_40/gate.json
OUT=results/a1_ext_formal2_wiretest

echo "===== 1) 接线（幂等）====="
$PY ../CC_Compare/tools/wire_extra_providers.py
$PY - <<'PYEOF'
import sys
sys.path.insert(0, 'scripts/evaluation')
import ordering_providers as op
print('registry:', sorted(op._REGISTRY))
PYEOF

echo "===== 2) CO-enriched 纯排序 1 天 ====="
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider coenrich --provider-ckpt ../CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1 \
    --provider-device cpu --baseline-from "$BASE" --dev-instances 1 --workers 1 \
    --penalty "p_c=0" --out "$OUT/coenrich_1d" 2>&1 | tail -6
echo "coenrich rc=$?"
echo "----- 结果摘要 -----"
$PY - <<'PYEOF'
import json, os
p = 'results/a1_ext_formal2_wiretest/coenrich_1d/gate.json'
if not os.path.exists(p):
    print('NO GATE'); raise SystemExit
d = json.load(open(p, encoding='utf-8'))
for yr, yb in d['results'].items():
    for pname, blk in yb.items():
        if pname.startswith('_'):
            continue
        for an, ab in blk['arms'].items():
            u = ab.get('utility_all_days') or {}
            print('  %-18s %-18s util=%s to=%s hfr=%s' % (
                pname, an, u.get('mean'), ab.get('mean_timeouts'), ab.get('hard_feasible_rate')))
    print('  identity:', bool(yb.get('_identity')))
PYEOF
echo "===== 集成测试结束（MAPT 单独测）====="
