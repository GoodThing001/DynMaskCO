#!/bin/bash
# Read the ACTUAL schema of sealed fixed-state assets (read-only, bounded output).
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

for f in \
  results/m0_scale/sgbs_fixed_cal/summary.json \
  results/m0_scale/sgbs_fixed_cal/per_state.json \
  results/m0_scale/mpre_reinforce_s42_clean/summary.json \
  results/m0_scale/mpre_reinforce_s42_clean/training_identity.json \
  results/correction_validation_1009_v3/fixed_state/per_state.json \
  results/correction_validation_1009_v3/fixed_state/summary.json ; do
  if [ ! -f "$f" ]; then echo "MISSING $f"; continue; fi
  echo "===== $f  ($(stat -c %s "$f") bytes) ====="
  $PY - "$f" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
def keys(o, depth=0, maxd=3):
    pad = '  ' * depth
    if isinstance(o, dict):
        for k, v in list(o.items())[:14]:
            t = type(v).__name__
            if isinstance(v, (dict, list)) and depth < maxd:
                n = len(v)
                print('%s%s: %s(%d)' % (pad, k, t, n))
                if isinstance(v, dict):
                    keys(v, depth + 1, maxd)
                elif v and isinstance(v[0], dict):
                    keys(v[0], depth + 1, maxd)
            else:
                s = str(v)
                print('%s%s: %s = %s' % (pad, k, t, s[:70]))
    elif isinstance(o, list) and o:
        keys(o[0], depth, maxd)
print('TOP TYPE:', type(d).__name__, 'len', len(d))
keys(d)
EOF
done
exit 0
