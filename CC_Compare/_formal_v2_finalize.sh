#!/bin/bash
# 正式批次 v2 收口：等四条腿全部 ALLDONE → 重新采集总表数据集 → 打印身份与裁决状态。
# 用法（服务器仓库根）：bash CC_Compare/_formal_v2_finalize.sh
set -u
ROOT=/home/hzeng/project/MASKCO-Main
EXT=$ROOT/C-VRP_Cold-chainVehicleRoutingProblem
PY=/home/hzeng/envs/cc_compare/bin/python
RUN=/tmp/xformal_v2

echo "== 等待 /tmp/xformal_v2/*.done 全部 ALLDONE =="
for i in $(seq 1 480); do
  n=$(grep -l ALLDONE $RUN/gpu.done $RUN/cpuA.done $RUN/cpuB.done $RUN/cpuC.done 2>/dev/null | wc -l)
  echo "  [$i] 已完成腿数 = $n / 4"
  [ "$n" -ge 4 ] && break
  sleep 60
done

echo "== 完成标记 =="
for f in $RUN/*.done; do echo "--- $f"; cat "$f"; done

echo "== 源码冻结对账（启动 vs 现在）=="
cd "$EXT" || exit 1
$PY ../CC_Compare/tools/record_freeze_v2.py --json ../CC_Compare/results/a1_compare_table/freeze_v2_end.json
$PY - <<'PYEOF'
import json
a = json.load(open('../CC_Compare/results/a1_compare_table/freeze_v2_start.json', encoding='utf-8'))
b = json.load(open('../CC_Compare/results/a1_compare_table/freeze_v2_end.json', encoding='utf-8'))
diff = [k for k in a if a[k] != b.get(k)]
print('冻结文件数:', len(a), '| 启动后发生变化的文件数:', len(diff))
for k in diff:
    print('  CHANGED:', k)
PYEOF

echo "== 各臂 identity / 配对裁决状态 =="
$PY - <<'PYEOF'
import glob, json, os
root = 'results/a1_ext_formal2'
rows = []
for p in sorted(glob.glob(os.path.join(root, '*', 'gate.json'))):
    name = os.path.basename(os.path.dirname(p))
    d = json.load(open(p, encoding='utf-8'))
    for yr, yb in (d.get('results') or {}).items():
        ident = yb.get('_identity') or {}
        pid = yb.get('_pairing_identity') or {}
        for pname, blk in yb.items():
            if pname.startswith('_') or not isinstance(blk, dict):
                continue
            for an, ab in (blk.get('arms') or {}).items():
                if an in ('cond_hist', 'uncond_hist', 'explicit_feat'):
                    continue
                u = ab.get('utility_all_days') or {}
                pd = blk.get('%s_minus_cond_hist' % an) or {}
                rows.append((name, pname, an, u.get('mean'), ab.get('mean_timeouts'),
                             ab.get('hard_feasible_rate'),
                             ident.get('source_stable'),
                             pid.get('identity_verified'), pid.get('problems'),
                             pd.get('mean'), pd.get('formal_adjudication')))
print('%-22s %-16s %-16s %9s %7s %5s %6s %6s %9s %6s' % (
    'arm_dir', 'p_c', 'arm', 'util', 'to/day', 'hfr', 'stable', 'paired', 'vs_cond', 'formal'))
for r in rows:
    print('%-22s %-16s %-16s %9s %7s %5s %6s %6s %9s %6s' % (
        r[0], r[1], r[2],
        ('%.1f' % r[3]) if isinstance(r[3], float) else r[3],
        ('%.2f' % r[4]) if isinstance(r[4], float) else r[4],
        r[5], r[6], r[7], str(r[8]),
        ('%.1f' % r[9]) if isinstance(r[9], float) else r[9], r[10]))
PYEOF

echo "== 重新采集总表数据集 =="
$PY ../CC_Compare/tools/collect_compare_table.py \
    --json ../CC_Compare/results/a1_compare_table/dataset.json
echo "== 完成（本地执行 render_compare_table.py 生成表格）=="
