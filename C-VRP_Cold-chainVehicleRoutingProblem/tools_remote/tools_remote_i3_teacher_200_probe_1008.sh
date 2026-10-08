#!/usr/bin/env bash
# One finite engineering probe. No development/final-test data or model training.
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
BASE="$ROOT/results/a3_baseline_condonly_v2_src_frozen_20261007"
OUT="$ROOT/results/i3_teacher_200_probe_1008_0908"
SRC="$OUT/src"
SESSION=i3_teacher_200_probe_1008_0908
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
cd "$ROOT"
test ! -e "$OUT"
! tmux has-session -t "$SESSION" 2>/dev/null
(cd "$BASE" && sha256sum -c SOURCE_MANIFEST.sha256)
mkdir -p "$OUT"
cp -a "$BASE" "$SRC"
chmod -R u+w "$SRC"
mkdir -p "$SRC/scripts/training"
cp scripts/training/i3_counterfactual_teacher.py "$SRC/scripts/training/"
cp scripts/evaluation/diag_i3_teacher_probe.py "$SRC/scripts/evaluation/"
cp scripts/simulation/recourse_snapshot.py "$SRC/scripts/simulation/"
(cd "$SRC" && sha256sum scripts/training/i3_counterfactual_teacher.py \
    scripts/evaluation/diag_i3_teacher_probe.py scripts/simulation/recourse_snapshot.py > I3_PROBE_MANIFEST.sha256)
(cd "$SRC" && sha256sum -c SOURCE_MANIFEST.sha256 && sha256sum -c I3_PROBE_MANIFEST.sha256) > "$OUT/verify_start.txt"
chmod -R a-w "$SRC"
"$PY" "$SRC/scripts/evaluation/diag_i3_teacher_probe.py" --help > "$OUT/import_check.txt"
cp results/a3_repair_chain_1007_2250/server_resources_1008.json "$OUT/"
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
OUT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/i3_teacher_200_probe_1008_0908
SRC="$OUT/src"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
date > "$OUT/started_at.txt"
timeout 1800s "$PY" "$SRC/scripts/evaluation/diag_i3_teacher_probe.py" \
    --scale200 --out "$OUT/probe.json" > "$OUT/run.log" 2>&1
RUN_RC=$?
(cd "$SRC" && sha256sum -c SOURCE_MANIFEST.sha256 && sha256sum -c I3_PROBE_MANIFEST.sha256) > "$OUT/verify_end.txt" 2>&1
SEAL_RC=$?
printf 'RUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$RUN_RC" "$SEAL_RC" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
chmod u+x "$OUT/run.sh"
tmux new-session -d -s "$SESSION" "bash '$OUT/run.sh'"
tmux has-session -t "$SESSION"
printf 'STARTED=%s\nOUTPUT=%s\n' "$SESSION" "$OUT"
