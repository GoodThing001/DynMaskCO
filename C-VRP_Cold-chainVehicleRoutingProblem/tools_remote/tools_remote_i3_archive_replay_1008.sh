#!/usr/bin/env bash
# Independently replay every branch from the fixed 200-order engineering archive.
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
INPUT="$ROOT/results/i3_teacher_200_archive_1008_1055"
OUT="$ROOT/results/i3_teacher_200_archive_replay_1008"
SESSION=i3_teacher_200_archive_replay_1008
test ! -e "$OUT"
! tmux has-session -t "$SESSION" 2>/dev/null
grep -qx RUN_EXIT=0 "$INPUT/run.exit"
grep -qx SEAL_EXIT=0 "$INPUT/run.exit"
mkdir -p "$OUT"
cp -a "$INPUT/src" "$OUT/src"
chmod -R u+w "$OUT/src"
cp "$ROOT/scripts/evaluation/diag_i3_archive_replay.py" "$OUT/src/scripts/evaluation/"
(cd "$OUT/src" && sha256sum scripts/evaluation/diag_i3_archive_replay.py > REPLAY_MANIFEST.sha256)
chmod -R a-w "$OUT/src"
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
INPUT="$ROOT/results/i3_teacher_200_archive_1008_1055"
OUT="$ROOT/results/i3_teacher_200_archive_replay_1008"
SRC="$OUT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
date > "$OUT/started_at.txt"
(cd "$SRC" && sha256sum -c SOURCE_MANIFEST.sha256 && sha256sum -c I3_PROBE_MANIFEST.sha256 && sha256sum -c REPLAY_MANIFEST.sha256) > "$OUT/verify_start.txt" 2>&1
START_RC=$?
if [ "$START_RC" -eq 0 ]; then
  timeout 1800s /home/hzeng/miniconda3/envs/MASKCO_env/bin/python "$SRC/scripts/evaluation/diag_i3_archive_replay.py" --report "$INPUT/probe.json" --archive "$INPUT/private.json" --out "$OUT/independent_replay.json" > "$OUT/run.log" 2>&1
  RUN_RC=$?
else
  RUN_RC=99
fi
(cd "$SRC" && sha256sum -c SOURCE_MANIFEST.sha256 && sha256sum -c I3_PROBE_MANIFEST.sha256 && sha256sum -c REPLAY_MANIFEST.sha256) > "$OUT/verify_end.txt" 2>&1
SEAL_RC=$?
printf 'START_EXIT=%s\nRUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$START_RC" "$RUN_RC" "$SEAL_RC" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
tmux new-session -d -s "$SESSION" "bash '$OUT/run.sh'"
tmux has-session -t "$SESSION"
printf 'STARTED=%s\nOUTPUT=%s\n' "$SESSION" "$OUT"
