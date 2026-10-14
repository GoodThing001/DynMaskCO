#!/usr/bin/env bash
# Finite engineering validation, no online gate or formal I3 training.
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_training_runner_probe_1008_1110"
SESSION=i3_training_runner_probe_1008_1110
test ! -e "$OUT"
! tmux has-session -t "$SESSION" 2>/dev/null
mkdir -p "$OUT"
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_training_runner_probe_1008_1110"
cd "$ROOT"
date > "$OUT/started_at.txt"
timeout 900s /home/hzeng/miniconda3/envs/MASKCO_env/bin/python tools_remote/validate_i3_training_runner_1008.py > "$OUT/run.log" 2>&1
rc=$?
printf 'RUN_EXIT=%s\nDONE=1\n' "$rc" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
tmux new-session -d -s "$SESSION" "bash '$OUT/run.sh'"
tmux has-session -t "$SESSION"
printf 'STARTED=%s\nOUTPUT=%s\n' "$SESSION" "$OUT"
