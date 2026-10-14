#!/bin/bash
# P0 task 2 follow-up: bounded READ-ONLY provenance trace for
#   results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json
# Question: can a copy with the same identity be recovered from another trusted archive?
# Local SHA of the found copy: 10e60da10a80d74ae59cc10333f012f1f09e9541497db2d69c84f48ed776564a
# Embedded identities: audit_sha256=ac747189... , merged_teacher_sha256=9028f757...
set -u
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
cd "$EXT" || exit 1
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

echo "== ENV =="; echo "host=$(hostname) date=$(date -Is) pwd=$(pwd)"

echo
echo "== [P1] does the v5 dir exist on the server, and what is in it? =="
if [ -d results/i3_registered_1008_v5 ]; then
  echo "DIR PRESENT: results/i3_registered_1008_v5"
  ls -l results/i3_registered_1008_v5 2>/dev/null | head -30
  echo "-- top-level file count: $(find results/i3_registered_1008_v5 -maxdepth 1 -type f | wc -l)"
else
  echo "DIR ABSENT: results/i3_registered_1008_v5"
fi

echo
echo "== [P2] server-wide search for the exact filename (depth<=9) =="
find . -maxdepth 9 -type f -name 'LOCAL_128_STAGE_ACCEPTANCE_20261009.json' -print 2>/dev/null | while read -r f; do
  echo "$(sha256sum "$f" | cut -c1-16)  $(stat -c '%s' "$f")  $f"
done
echo "(end of filename search)"

echo
echo "== [P3] any *STAGE_ACCEPTANCE* on the server =="
find . -maxdepth 9 -type f -name '*STAGE_ACCEPTANCE*' -print 2>/dev/null | head -20
echo "(end)"

echo
echo "== [P4] corroborating artifacts named by the local trace: present on the server? =="
for f in \
  results/i3_registered_1008_v5/INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json \
  results/i3_registered_1008_v5/TEACHER_AUDIT_ACCEPTED.json \
  results/i3_registered_1008_v5/SOURCE_MANIFEST.sha256 \
  results/i3_registered_1008_v5/teacher_merged/report.json \
  results/i3_registered_1008_v5/maskco/train_report.json \
  results/i3_registered_1008_v5/reconstruction_control/train_report.json \
  results/i3_online_recovery_1009_v1/INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json \
  results/i3_online_recovery_1009_v1/SOURCE_MANIFEST.sha256 ; do
  if [ -f "$f" ]; then echo "OK   $(stat -c '%s' "$f")  $f"; else echo "MISS $f"; fi
done

echo
echo "== [P5] does any server file carry the two embedded identities? (bounded) =="
for H in ac747189c1a2f02e5268ee934c8ad0cecd1780a3c76ec9a6a51dac80a3c22a72 \
         9028f75770d2ac8d7b115101cf58af297839e8ef1f6cdf48d8c717b8e5081495 ; do
  echo "-- ${H:0:16}..."
  grep -rl --include='*.json' --include='*.sha256' --include='*.txt' "$H" \
       results/i3_registered_1008_v5 results/i3_online_recovery_1009_v1 2>/dev/null | head -8
  echo "   (end)"
done

echo
echo "== [P6] offline archives on the server that might hold it =="
for d in B_REPRO_CLEAN/results B_REPRO_CLEAN/repro_work \
         results/correction_validation_1009_v2/recovered results/correction_validation_1009_v3/recovered ; do
  if [ -d "/home/hzeng/project/MASKCO-Main/$d" ]; then
    n=$(find "/home/hzeng/project/MASKCO-Main/$d" -type f -name '*128*ACCEPT*' 2>/dev/null | head -5)
    echo "DIR $d : ${n:-no *128*ACCEPT* match}"
  else
    echo "DIR $d : absent"
  fi
done

echo
echo "== DONE =="
exit 0
