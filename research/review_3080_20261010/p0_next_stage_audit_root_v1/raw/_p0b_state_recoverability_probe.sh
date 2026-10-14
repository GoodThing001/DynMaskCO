#!/bin/bash
# Task B: bounded READ-ONLY probe of the 32-state asset and its input identities.
set -u
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
cd "$EXT" || exit 1
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

echo "== ENV =="; echo "host=$(hostname) date=$(date -Is)"

echo
echo "== [B1] the 32-state asset directory (ALL files) =="
ls -la results/m0_scale/sgbs_fixed_cal 2>/dev/null || echo "MISSING results/m0_scale/sgbs_fixed_cal"

echo
echo "== [B2] per_state.json key inventory (actual schema) =="
P=results/m0_scale/sgbs_fixed_cal/per_state.json
if [ -f "$P" ]; then
  $PY - "$P" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print('records:', len(d))
keys = sorted(d[0].keys()) if d else []
print('per-record keys:', keys)
for probe in ('snapshot','state_hash','schema_version','vehicles','mask_set','visible_ids',
              'P0','partial','protected','vis','J0','inst','event','clock'):
    print('  %-16s %s' % (probe, probe in keys))
print('sample record:', json.dumps(d[0], ensure_ascii=False)[:260] if d else None)
EOF
else echo "MISSING $P"; fi

echo
echo "== [B3] summary.json identity block =="
S=results/m0_scale/sgbs_fixed_cal/summary.json
[ -f "$S" ] && $PY -c "
import json,sys
d=json.load(open('$S',encoding='utf-8'))
print('top keys:', sorted(d.keys()))
for k in ('split','n_states','n_instances','seed','identity','source','config','data_sha256','manifest'):
    if k in d: print(' ',k,'=',json.dumps(d[k],ensure_ascii=False)[:200])
" || echo "MISSING $S"

echo
echo "== [B4] input data identity: cal npz + any manifest =="
for f in data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
         data/m0_scale/M0_TEACHER_MANIFEST.json data/m0_scale/manifest.json \
         data/m0_scale/DATA_MANIFEST.json ; do
  if [ -f "$f" ]; then echo "OK   $(stat -c '%s' "$f")  $(sha256sum "$f" | cut -c1-32)  $f"; else echo "MISS $f"; fi
done
echo "-- all files in data/m0_scale (depth 1) --"
ls -la data/m0_scale 2>/dev/null | head -20

echo
echo "== [B5] do m0_scale result manifests record the cal npz sha? (bounded depth 2) =="
grep -rl --include='*.json' --include='*.sha256' 'dcc_50_r1_edod05_cal_teacher' \
     results/m0_scale --exclude-dir=_bverify_pinned 2>/dev/null | head -12
echo "(end)"

echo
echo "== [B6] is a snapshot module present + its registered identity status =="
for f in scripts/simulation/recourse_snapshot.py scripts/evaluation/run_sgbs_fixed_state.py \
         scripts/evaluation/run_fixed_state_quality.py ; do
  [ -f "$f" ] && echo "OK   $(sha256sum "$f" | cut -c1-32)  $f" || echo "MISS $f"
done
echo "-- snapshot schema versions --"
grep -n 'SNAPSHOT_SCHEMA_VERSION\s*=\|CC_SNAPSHOT_SCHEMA_VERSION\s*=' scripts/simulation/recourse_snapshot.py | head -4

echo
echo "== DONE =="
exit 0
