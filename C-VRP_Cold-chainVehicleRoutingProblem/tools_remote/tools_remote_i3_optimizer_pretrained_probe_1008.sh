#!/usr/bin/env bash
# Synthetic gradient/save test with actual frozen CVRP weights, CPU only.
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
UPSTREAM=/home/hzeng/project/MASKCO-Main/MASKCO_code
OUT="$ROOT/results/i3_optimizer_pretrained_probe_1008_0915"
EXT="$OUT/src/C-VRP_Cold-chainVehicleRoutingProblem"
SESSION=i3_optimizer_pretrained_probe_1008_0915
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
test ! -e "$OUT"
! tmux has-session -t "$SESSION" 2>/dev/null
mkdir -p "$EXT"
cp -a "$ROOT/scripts" "$EXT/"
# Preserve project_paths layout; this link only reads the upstream repository.
ln -s "$UPSTREAM" "$OUT/src/MASKCO_code"
find "$EXT/scripts" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_MANIFEST.sha256"
find "$UPSTREAM" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$OUT/UPSTREAM_PY_MANIFEST.sha256"
sha256sum "$UPSTREAM/ckpts/cvrp100.ckpt" > "$OUT/WEIGHTS_MANIFEST.sha256"
chmod -R a-w "$EXT"
cat > "$OUT/run.py" <<'PY'
import os
import runpy
import sys
from pathlib import Path
out = Path(__file__).resolve().parent
# JAX and BLAS thread pools inherit one allowed CPU before initialization.
allowed = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(allowed)})
os.environ.update(JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
script = out / 'src/C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_i3_optimizer.py'
sys.argv = [str(script), '--cvrp-ckpt',
            '/home/hzeng/project/MASKCO-Main/MASKCO_code/ckpts/cvrp100.ckpt']
print('ENGINEERING_ONLY: actual CVRP checkpoint, synthetic losses, 1 CPU, no GPU', flush=True)
runpy.run_path(str(script), run_name='__main__')
PY
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
OUT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/i3_optimizer_pretrained_probe_1008_0915
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date > "$OUT/started_at.txt"
timeout 900s "$PY" "$OUT/run.py" > "$OUT/run.log" 2>&1
RUN_RC=$?
{ sha256sum -c "$OUT/SOURCE_MANIFEST.sha256"; sha256sum -c "$OUT/UPSTREAM_PY_MANIFEST.sha256"; sha256sum -c "$OUT/WEIGHTS_MANIFEST.sha256"; } > "$OUT/verify_end.txt" 2>&1
SEAL_RC=$?
printf 'RUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$RUN_RC" "$SEAL_RC" > "$OUT/run.exit"
date > "$OUT/ended_at.txt"
RUN
{ sha256sum -c "$OUT/SOURCE_MANIFEST.sha256"; sha256sum -c "$OUT/UPSTREAM_PY_MANIFEST.sha256"; sha256sum -c "$OUT/WEIGHTS_MANIFEST.sha256"; } > "$OUT/verify_start.txt"
tmux new-session -d -s "$SESSION" "bash '$OUT/run.sh'"
tmux has-session -t "$SESSION"
printf 'STARTED=%s\nOUTPUT=%s\n' "$SESSION" "$OUT"
