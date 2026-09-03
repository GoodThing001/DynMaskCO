#!/bin/bash
# ============================================================
# P0-2: Baseline Solver Audit (OR-Tools + PyVRP)
# ============================================================
# Validates that OR solvers are correctly adapted before claiming
# "0% feasible" on narrow-TW DCC instances.
#
# Prerequisites:
#   pip install ortools==9.8.3296 pyvrp==0.11.3
#
# Usage:
#   bash run_baseline_audit.sh          # full audit
#   bash run_baseline_audit.sh smoke    # quick smoke test (skip DCC 60s)
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate

SCRIPTS="C-VRP_Cold-chainVehicleRoutingProblem/scripts"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz"

echo "============================================================"
echo "P0-2: Baseline Solver Audit"
echo "============================================================"

if [ "${1:-}" = "smoke" ]; then
    echo "Mode: SMOKE (skip DCC 60s test)"
    python -u "$SCRIPTS/baselines/solomon_audit.py"
else
    echo "Mode: FULL"
    python -u "$SCRIPTS/baselines/solomon_audit.py" --data "$DATA"
fi
