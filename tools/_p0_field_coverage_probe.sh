#!/bin/bash
# P0 audit task 2: READ-ONLY field-coverage probe over I3 / teacher / online artifacts.
# Prints existence, size and the ACTUAL schema of each artifact so field coverage is
# evidence-based.  No writes, no training, no new runs.
set -u
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
cd "$EXT" || exit 1
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

echo "== ENV =="; echo "host=$(hostname) date=$(date -Is) pwd=$(pwd)"

echo
echo "== [T2-1] candidate artifacts: existence + size =="
for f in \
  results/a1_c1rr2_reveal_40/gate.json \
  results/a1_c1fresh_reveal_40/gate.json \
  results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json \
  results/correction_validation_1009_v3/fixed_state/per_state.json \
  results/correction_validation_1009_v3/fixed_state/summary.json \
  data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
  data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
  data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz ; do
  if [ -f "$f" ]; then echo "OK   $(stat -c '%s' "$f")  $f"; else echo "MISS $f"; fi
done

echo
echo "== [T2-2] any per-DECISION online log for I3 / a1 arms? =="
find results -maxdepth 4 \( -path '*/_bverify_pinned*' \) -prune -o \
  -type f \( -name '*DECISION*' -o -name '*decision_log*' -o -name '*per_decision*' \
             -o -name '*trajectory*' -o -name '*trace*' \) -print 2>/dev/null | head -25

echo
echo "== [T2-3] teacher dataset sidecars (contexts/candidates jsonl) =="
find . -maxdepth 6 -type f \( -name 'contexts.jsonl' -o -name 'candidates.jsonl' \) -print 2>/dev/null \
  | head -12 | while read -r f; do echo "$(stat -c '%s' "$f")  $(wc -l < "$f") lines  $f"; done

echo
echo "== [T2-4] gate.json schema (the recorded cond_hist control) =="
G=results/a1_c1rr2_reveal_40/gate.json
if [ -f "$G" ]; then
  $PY - "$G" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print('TOP keys:', sorted(d.keys())[:24])
for k in ('days', 'per_day', 'rows', 'instances', 'records', 'results'):
    v = d.get(k)
    if isinstance(v, list) and v:
        print('  %s: list(%d) elem keys = %s' % (k, len(v), sorted(v[0].keys())[:18]))
    elif isinstance(v, dict):
        print('  %s: dict keys = %s' % (k, sorted(v.keys())[:14]))
EOF
else echo "MISS $G"; fi

echo
echo "== [T2-5] field probe across artifacts (which of the named fields exist?) =="
FIELDS='visible_state|scenario|margin|P_before|P_after|install|receipt|commit|timeout|reason|certif|accepted|rejected|utility|revenue|energy_kwh|hard'
for f in results/a1_c1rr2_reveal_40/gate.json \
         results/correction_validation_1009_v3/fixed_state/per_state.json \
         results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json ; do
  [ -f "$f" ] || continue
  echo "-- $f"
  for pat in visible_state scenario_id margin P_before install_receipt commit_diff timeout_reason cert_ok plan_hash incremental_distance; do
    n=$(grep -o -i "$pat" "$f" 2>/dev/null | wc -l)
    printf '     %-22s %s\n' "$pat" "$n"
  done
done

echo
echo "== [T2-6] teacher npz keys (what per-state supervision exists) =="
$PY - <<'EOF'
import numpy as np, os
for p in ('data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz',
          'data/m0_scale/dcc_50_r1_edod05_train_teacher.npz'):
    if not os.path.isfile(p):
        print('MISS', p); continue
    with np.load(p, allow_pickle=True) as z:
        print('--', p, 'keys:', sorted(z.keys()))
        for k in sorted(z.keys()):
            a = z[k]
            print('     %-24s shape=%s dtype=%s' % (k, getattr(a, 'shape', None), a.dtype))
EOF
echo
echo "== DONE =="
exit 0
