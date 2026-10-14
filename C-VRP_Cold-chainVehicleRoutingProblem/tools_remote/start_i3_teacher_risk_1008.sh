#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_teacher_risk_1008_v1"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
test ! -e "$OUT"
! tmux has-session -t i3_teacher_risk_1008_v1 2>/dev/null
mkdir -p "$OUT"
cp "$ROOT/tools_remote/i3_teacher_risk_diagnostic_1008.py" "$OUT/diagnostic.py"
cp "$ROOT/tools_remote/i3_teacher_risk_chain_1008.py" "$OUT/chain.py"
cp "$ROOT/configs/i3_20261008_v1/teacher_risk_addendum.json" "$OUT/addendum.json"
sha256sum "$OUT/diagnostic.py" "$OUT/chain.py" "$OUT/addendum.json" > "$OUT/SOURCE_MANIFEST.sha256"
chmod a-w "$OUT/diagnostic.py" "$OUT/chain.py" "$OUT/addendum.json" "$OUT/SOURCE_MANIFEST.sha256"
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_teacher_risk_1008_v1"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date > "$OUT/started_at.txt"
timeout 216000s "$PY" "$OUT/chain.py" "$OUT" "$ROOT" > "$OUT/chain.log" 2>&1
rc=$?
sha256sum -c "$OUT/SOURCE_MANIFEST.sha256" > "$OUT/verify_end.txt" 2>&1
seal=$?
printf 'RUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$rc" "$seal" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
tmux new-session -d -s i3_teacher_risk_1008_v1 "bash '$OUT/run.sh'"
tmux has-session -t i3_teacher_risk_1008_v1
printf 'STARTED_FINITE_TEACHER_DIAGNOSTIC=1\n'
