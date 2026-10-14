#!/bin/bash
# Locate READ-ONLY copies of the training-era sources by SHA.  No writes to any frozen tree.
# Targets (from F's training_identity.json):
#   scripts/simulation/dynmaskco_cc_context.py  9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54
#   scripts/simulation/cc_lns_replanner.py      7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3
set -u
BASE=/home/hzeng/project/MASKCO-Main
cd "$BASE" || exit 1

echo "== ENV =="; echo "host=$(hostname) date=$(date -Is) pwd=$(pwd)"

echo
echo "== [A] every copy of the two filenames under BASE (depth<=9), with sha =="
find . -maxdepth 9 -type f \( -name 'dynmaskco_cc_context.py' -o -name 'cc_lns_replanner.py' \) \
     -print 2>/dev/null | while read -r f; do
  echo "$(sha256sum "$f" | cut -c1-16)  $(stat -c '%s' "$f")  $f"
done

echo
echo "== [B] exact-SHA hunt for the training-era context module =="
find . -maxdepth 9 -type f -name 'dynmaskco_cc_context.py' -print 2>/dev/null | while read -r f; do
  s=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$s" = "9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54" ]; then
    echo "MATCH_TRAIN_CONTEXT  $f"
  fi
done
echo "== [B2] exact-SHA hunt for the training-era cc_lns_replanner =="
find . -maxdepth 9 -type f -name 'cc_lns_replanner.py' -print 2>/dev/null | while read -r f; do
  s=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$s" = "7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3" ]; then
    echo "MATCH_TRAIN_D  $f"
  fi
done

echo
echo "== [C] is there a B_REPRO_CLEAN mirror tree? =="
if [ -d B_REPRO_CLEAN ]; then
  ls -d B_REPRO_CLEAN/* 2>/dev/null | head -12
  for p in B_REPRO_CLEAN/scripts/simulation/dynmaskco_cc_context.py \
           B_REPRO_CLEAN/scripts/simulation/cc_lns_replanner.py; do
    [ -f "$p" ] && sha256sum "$p" || echo "MISS $p"
  done
else
  echo "NO B_REPRO_CLEAN at $BASE"
fi

echo
echo "== [D] recovered workspace snapshots holding a full extension tree =="
find . -maxdepth 6 -type d -name 'workspace' -print 2>/dev/null | head -10

echo
echo "== [E] state recoverability: training state manifest =="
M=results/m0_scale/mpre_reinforce_s42_clean/state_manifest.json
if [ -f "$M" ]; then
  /home/hzeng/miniconda3/envs/MASKCO_env/bin/python - "$M" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
ie = d.get('inst_event', [])
print('n_states =', len(ie))
print('first 6  =', ie[:6])
print('dist keys=', len(d.get('inst_dist', {})))
EOF
else
  echo "MISS $M"
fi

echo
echo "== [F] do the fixed-state sealed assets name the source identity they used? =="
for f in results/correction_validation_1009_v3/fixed_state/summary.json \
         results/correction_validation_1009_v3/SOURCE_START.json \
         results/i3_registered_1008_v3/engineering_actual_teacher_training/config.json; do
  if [ -f "$f" ]; then
    echo "-- $f"
    /home/hzeng/miniconda3/envs/MASKCO_env/bin/python - "$f" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception as e:
    print('  unreadable:', e); raise SystemExit
def hunt(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v):
                if 'dynmaskco_cc_context' in k or 'cc_lns_replanner' in k:
                    print('  %s = %s' % (k, v))
            elif isinstance(v, (dict, list)):
                hunt(v, path + '/' + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            if isinstance(v, (dict, list)):
                hunt(v, path)
hunt(d)
EOF
  else
    echo "MISS $f"
  fi
done

echo
echo "== DONE =="
exit 0
