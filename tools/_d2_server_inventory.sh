#!/bin/bash
# D2 unblock, server side: bounded, READ-ONLY asset inventory.  No writes, no training.
# Everything is printed with explicit markers so "not found in the searched scope" is never
# reported as "absent from the server".
set -u
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
cd "$EXT" || { echo "FATAL: cannot cd $EXT"; exit 1; }

echo "== ENV =="
echo "host=$(hostname) date=$(date -Is) pwd=$(pwd)"
echo "python=$(/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -V 2>&1)"

echo
echo "== [1] F checkpoint: recorded path + SHA =="
P=results/m0_scale/mpre_reinforce_s42_clean/model.ckpt
if [ -f "$P" ]; then sha256sum "$P"; ls -l "$P"; else echo "MISSING: $P"; fi

echo
echo "== [1b] any *clean* / mpre / mtrained training dirs (depth<=4, _bverify_pinned pruned) =="
find results -maxdepth 4 \
  \( -path '*/_bverify_pinned' -o -path '*/_bverify_pinned/*' \) -prune -o \
  -type d \( -name '*mpre*' -o -name '*mtrained*' -o -name '*reinforce*' -o -name '*clean*' \) -print 2>/dev/null | head -40

echo
echo "== [1c] every model.ckpt / *.ckpt under results (depth<=6, pinned pruned), with size =="
find results -maxdepth 6 \
  \( -path '*/_bverify_pinned' -o -path '*/_bverify_pinned/*' \) -prune -o \
  -type f \( -name 'model.ckpt' -o -name '*.ckpt' -o -name '*.msgpack' -o -name '*.orbax' \) -print 2>/dev/null \
  | head -60 | while read -r f; do echo "$(stat -c '%s' "$f" 2>/dev/null) $f"; done

echo
echo "== [1d] sha256 of any ckpt whose dir mentions mpre/mtrained/reinforce/clean =="
find results -maxdepth 6 \
  \( -path '*/_bverify_pinned' -o -path '*/_bverify_pinned/*' \) -prune -o \
  -type f \( -name 'model.ckpt' -o -name '*.ckpt' \) -print 2>/dev/null \
  | grep -Ei 'mpre|mtrained|reinforce|clean' | head -20 | while read -r f; do sha256sum "$f"; done

echo
echo "== [2] D geometric baseline =="
for f in scripts/simulation/cc_lns_replanner.py scripts/simulation/cc_lns_swap.py \
         scripts/evaluation/run_cc_lns_swap_fixed_state.py scripts/evaluation/run_cc_lns_swap_gate.py \
         scripts/simulation/dynmaskco_cc_context.py scripts/evaluation/run_fixed_state_quality.py; do
  if [ -f "$f" ]; then echo "OK   $(sha256sum "$f")"; else echo "MISS $f"; fi
done

echo
echo "== [3] mask pool call sites (definition + callers) =="
grep -rn "mask_candidate_pool\|validate_mask_scope" scripts --include='*.py' 2>/dev/null | head -30

echo
echo "== [4] E2: same-task random-init control artifacts =="
find results -maxdepth 5 \
  \( -path '*/_bverify_pinned' -o -path '*/_bverify_pinned/*' \) -prune -o \
  -type f -name 'manifest.json' -print 2>/dev/null | grep -Ei 'mpre|mtrained|reinforce|random|rand' | head -20
echo "-- training dirs under results/m0_scale (depth 1) --"
ls -1 results/m0_scale 2>/dev/null | head -40

echo
echo "== [5] fixed-state sealed outputs (json, <=512KB), with the two field probes =="
find results -maxdepth 6 \
  \( -path '*/_bverify_pinned' -o -path '*/_bverify_pinned/*' \) -prune -o \
  -type f -name '*.json' -size -512k -print 2>/dev/null \
  | grep -Ei 'fixed|mpre|mtrained|step5|sgbs' | head -30 | while read -r f; do
      a=$(grep -c -i 'executable_action\|legal_actions\|enumerate_legal_actions' "$f" 2>/dev/null || echo 0)
      c=$(grep -c -i 'commit_diff\|committed_difference\|true_commit\|installed' "$f" 2>/dev/null || echo 0)
      echo "actions=$a commit=$c  $f"
    done

echo
echo "== [5b] does the pinned/frozen identity record name the F checkpoint? =="
grep -rn "mpre_reinforce\|f5acf83f" --include='*.json' --include='*.md' . 2>/dev/null \
  | grep -v _bverify_pinned | head -15

echo
echo "== DONE =="
exit 0
