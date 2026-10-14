#!/usr/bin/env bash
# Fresh frozen release, finite QA and conditional registered experiment.
set -euo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_registered_1008_v3"
EXT="$OUT/src/C-VRP_Cold-chainVehicleRoutingProblem"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
test ! -e "$OUT"
! tmux has-session -t i3_registered_1008_v3 2>/dev/null
printf '36e08e06ce7b8e44eee2e6ff778904c1f26bd3145eb7d2072fb7360f33732318  %s\n' "$ROOT/scripts/evaluation/scenario_saa.py" | sha256sum -c -
printf '056a2d9a6f175d355b4e51f4bb587964a949d542553f45b764215e9dcbb55077  %s\n' "$ROOT/results/a1_a3/a3_train200_pretrained_s42/model.bin" | sha256sum -c -
mkdir -p "$EXT" "$OUT/warmstart"
cp -a "$ROOT/scripts" "$EXT/"
"$PY" -m zipfile -e "$ROOT/tools_remote/i3_release_1008_v2.zip" "$OUT/overlay"
cp -a "$OUT/overlay/scripts/." "$EXT/scripts/"
ln -s /home/hzeng/project/MASKCO-Main/MASKCO_code "$OUT/src/MASKCO_code"
cp "$ROOT/configs/i3_20261008_v1/protocol.json" "$OUT/protocol.json"
cp "$ROOT/tools_remote/i3_registered_pipeline_1008.py" "$OUT/pipeline.py"
cp "$ROOT/results/a1_a3/a3_train200_pretrained_s42/model.bin" "$ROOT/results/a1_a3/a3_train200_pretrained_s42/config.json" "$OUT/warmstart/"
find "$EXT/scripts" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_MANIFEST.sha256"
sha256sum "$OUT/pipeline.py" "$OUT/protocol.json" "$OUT/warmstart/model.bin" "$OUT/warmstart/config.json" /home/hzeng/project/MASKCO-Main/MASKCO_code/ckpts/cvrp100.ckpt >> "$OUT/SOURCE_MANIFEST.sha256"
chmod -R a-w "$EXT" "$OUT/warmstart"
chmod a-w "$OUT/pipeline.py" "$OUT/protocol.json" "$OUT/SOURCE_MANIFEST.sha256"
cp "$ROOT/results/a3_repair_chain_1007_2250/server_resources_i3_1008.json" "$OUT/resources.json"
cat > "$OUT/qa.sh" <<'QA'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_registered_1008_v3"
EXT="$OUT/src/C-VRP_Cold-chainVehicleRoutingProblem"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date > "$OUT/qa_started_at.txt"
timeout 1800s "$PY" "$EXT/scripts/tests/validate_i3_registered_release.py" --out "$OUT" --project "$ROOT" > "$OUT/qa.log" 2>&1
RUN_RC=$?
sha256sum -c "$OUT/SOURCE_MANIFEST.sha256" > "$OUT/qa_verify_end.txt" 2>&1
SEAL_RC=$?
printf 'RUN_EXIT=%s\nSEAL_EXIT=%s\nDONE=1\n' "$RUN_RC" "$SEAL_RC" > "$OUT/qa.exit"
date > "$OUT/qa_ended_at.txt"
QA
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
OUT="$ROOT/results/i3_registered_1008_v3"
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date > "$OUT/started_at.txt"
timeout 604800s "$PY" "$OUT/pipeline.py" "$OUT" "$ROOT" "$OUT/src/C-VRP_Cold-chainVehicleRoutingProblem" > "$OUT/pipeline.log" 2>&1
rc=$?
printf 'RUN_EXIT=%s\nDONE=1\n' "$rc" > "$OUT/pipeline.exit"
date > "$OUT/ended_at.txt"
RUN
sha256sum -c "$OUT/SOURCE_MANIFEST.sha256" > "$OUT/verify_start.txt"
tmux new-session -d -s i3_release_qa_1008_v3 "bash '$OUT/qa.sh'"
tmux new-session -d -s i3_registered_1008_v3 "bash '$OUT/run.sh'"
tmux has-session -t i3_release_qa_1008_v3
tmux has-session -t i3_registered_1008_v3
printf 'STARTED_QA=1\nSTARTED_CONDITIONAL_PIPELINE=1\nOUTPUT=%s\n' "$OUT"
