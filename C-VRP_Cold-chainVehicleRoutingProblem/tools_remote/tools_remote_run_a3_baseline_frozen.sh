#!/usr/bin/env bash
# A3 same-source cond_hist baseline. Never touches the original A2/A3 outputs.
set -euo pipefail
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
MODE=${1:-}
RUN_KIND=$MODE
if test "$MODE" = worker; then RUN_KIND=$2; fi
case "$RUN_KIND" in
  smoke-cond|full-cond) FROZEN=results/a3_baseline_condonly_v2_src_frozen_20261007 ;;
  *) FROZEN=results/a3_baseline_src_frozen_20261007 ;;
esac

if test "$MODE" = worker; then
  SEED=$3
  OUT=$4
  export RR3_FROZEN_SRC=$FROZEN
  export RR3_EVIDENCE_OUT=$OUT/import_evidence.txt
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  ARGS=(--gate-seed "$SEED" --future-policy density --energy-pricing marginal
        --shadow-mode greedy --K 10 --time-limit 10 --out "$OUT")
  if test "$RUN_KIND" = smoke; then
    ARGS+=(--gate-instances 1 --penalty p_c=0 --workers 3)
  elif test "$RUN_KIND" = smoke-cond; then
    ARGS+=(--gate-instances 1 --penalty p_c=0 --workers 1 --only-cond)
  elif test "$RUN_KIND" = full-cond; then
    ARGS+=(--gate-instances 40 --workers 3 --only-cond)
  else
    ARGS+=(--gate-instances 40 --workers 9)
  fi
  set +e
  "$PY" "$FROZEN/rr3_run_frozen.py" "${ARGS[@]}" > "$OUT/run.log" 2>&1
  RC=$?
  printf 'RUN_EXIT=%s\n' "$RC" > "$OUT/run.exit"
  (cd "$FROZEN" && sha256sum -c SOURCE_MANIFEST.sha256) > "$OUT/verify_end.txt" 2>&1
  VRC=$?
  printf 'SEAL_EXIT=%s\n' "$VRC" >> "$OUT/run.exit"
  if test "$RC" -eq 0 && test "$VRC" -eq 0 && test -f "$OUT/gate.json"; then
    printf 'DONE=1\n' >> "$OUT/run.exit"
  else
    printf 'DONE=0\n' >> "$OUT/run.exit"
  fi
  exit "$RC"
fi

case "$MODE" in
  smoke|smoke-cond) SEED=20260926 ;;
  full|full-cond)
    SEED=${2:-}
    case "$SEED" in 20260926|20260930|20261001) ;; *)
      echo 'REJECT: full requires one of 20260926/20260930/20261001'; exit 2;; esac ;;
  *) echo 'Usage: bash tools_remote_run_a3_baseline_frozen.sh smoke|smoke-cond|full <seed>|full-cond <seed>'; exit 2 ;;
esac

test -f "$FROZEN/PROVENANCE.json"
(cd "$FROZEN" && sha256sum -c SOURCE_MANIFEST.sha256) >/dev/null
TS=$(date +%m%d_%H%M%S)
OUT=results/a3_baseline_${MODE}_${SEED}_${TS}
SESS=a3base_${MODE}_${SEED}_${TS}
mkdir "$OUT"
{
  date '+launch=%F %T'
  echo "mode=$MODE seed=$SEED frozen=$FROZEN"
  uptime
  case "$MODE" in smoke) WORKERS=3;; smoke-cond) WORKERS=1;;
    full-cond) WORKERS=3;; *) WORKERS=9;; esac
  echo "nproc=$(nproc) workers=$WORKERS"
  sha256sum "$FROZEN/SOURCE_MANIFEST.sha256"
} > "$OUT/resource.txt"
(cd "$FROZEN" && sha256sum -c SOURCE_MANIFEST.sha256) > "$OUT/verify_start.txt"
tmux has-session -t "$SESS" 2>/dev/null && { echo "REJECT: session $SESS exists"; exit 2; }
tmux new-session -d -s "$SESS" "bash tools_remote/tools_remote_run_a3_baseline_frozen.sh worker $MODE $SEED $OUT"
echo "STARTED: $SESS $OUT"
