#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_teacher_data_audit_1008_v2"
test ! -e "$OUT"
! tmux has-session -t i3_teacher_data_audit_1008_v2 2>/dev/null
mkdir -p "$OUT"
cp "$ROOT/results/i3_registered_1008_v4/teacher_data_audit.py" "$OUT/audit.py"
cp "$ROOT/results/i3_registered_1008_v4/teacher_data_audit_chain.py" "$OUT/chain.py"
sha256sum "$OUT/audit.py" "$OUT/chain.py" > "$OUT/SOURCE_MANIFEST.sha256"
chmod a-w "$OUT/audit.py" "$OUT/chain.py" "$OUT/SOURCE_MANIFEST.sha256"
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_teacher_data_audit_1008_v2"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date > "$OUT/started_at.txt"
timeout 111600s "$PY" "$OUT/chain.py" "$OUT" "$ROOT" > "$OUT/chain.log" 2>&1
rc=$?
sha256sum -c "$OUT/SOURCE_MANIFEST.sha256" > "$OUT/verify_end.txt" 2>&1
seal=$?
printf 'RUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$rc" "$seal" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
tmux new-session -d -s i3_teacher_data_audit_1008_v2 "bash '$OUT/run.sh'"
tmux has-session -t i3_teacher_data_audit_1008_v2
printf 'STARTED_FINITE_TEACHER_DATA_AUDIT=1\n'
