#!/bin/bash
# ============================================================
# P0 Verification Suite — run all protocol audit tests
# ============================================================
# Runs:
#   P0-2: Baseline solver audit (Solomon sanity check)
#   P0-3a: Future cardinality leakage test
#   P0-3b: Target-label leakage test
#
# Prerequisites:
#   pip install ortools==9.8.3296 pyvrp==0.11.3
#
# Usage:
#   bash run_p0_verification.sh          # full suite
#   bash run_p0_verification.sh smoke    # quick (skip DCC 60s test)
#   bash run_p0_verification.sh p02      # P0-2 only
#   bash run_p0_verification.sh p03      # P0-3 only
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate

SCRIPTS="C-VRP_Cold-chainVehicleRoutingProblem/scripts"
DATA_P03A="$SCRIPTS/../data/p0_fix/dcc_50_r1_edod05_test.npz"
CKPT="$SCRIPTS/../ckpts/p0_fix/phase3c/step50000.ckpt"
DATA_P03B="$SCRIPTS/../data/p0_fix/dcc_50_mixed_edod_train.npz"

MODE="${1:-full}"

echo "============================================================"
echo "P0 Verification Suite — Protocol Audit"
echo "============================================================"
echo "Mode: $MODE"
echo ""

PASS=0
FAIL=0

# ── P0-2: Baseline Solver Audit ──
if [ "$MODE" = "full" ] || [ "$MODE" = "smoke" ] || [ "$MODE" = "p02" ]; then
    echo "============================================================"
    echo "P0-2: Baseline Solver Audit"
    echo "============================================================"
    if [ "$MODE" = "smoke" ]; then
        python -u "$SCRIPTS/baselines/solomon_audit.py" && ((PASS++)) || ((FAIL++))
    else
        python -u "$SCRIPTS/baselines/solomon_audit.py" --data "$DATA_P03A" && ((PASS++)) || ((FAIL++))
    fi
    echo ""
fi

# ── P0-3a: Future Cardinality Leakage ──
if [ "$MODE" = "full" ] || [ "$MODE" = "smoke" ] || [ "$MODE" = "p03" ]; then
    echo "============================================================"
    echo "P0-3a: Future Cardinality Leakage Test"
    echo "============================================================"
    if [ -f "$CKPT" ] && [ -f "$DATA_P03A" ]; then
        python -u "$SCRIPTS/tests/test_future_cardinality.py" \
            --data "$DATA_P03A" --ckpt "$CKPT" && ((PASS++)) || ((FAIL++))
    else
        echo "  SKIP: checkpoint or data not found"
        echo "    ckpt: $CKPT"
        echo "    data: $DATA_P03A"
    fi
    echo ""
fi

# ── P0-3b: Target-Label Leakage ──
if [ "$MODE" = "full" ] || [ "$MODE" = "smoke" ] || [ "$MODE" = "p03" ]; then
    echo "============================================================"
    echo "P0-3b: Target-Label Leakage Test"
    echo "============================================================"
    if [ -f "$CKPT" ] && [ -f "$DATA_P03B" ]; then
        python -u "$SCRIPTS/tests/test_target_label_leakage.py" \
            --data "$DATA_P03B" --ckpt "$CKPT" && ((PASS++)) || ((FAIL++))
    else
        echo "  SKIP: checkpoint or data not found"
        echo "    ckpt: $CKPT"
        echo "    data: $DATA_P03B"
    fi
    echo ""
fi

# ── Summary ──
echo "============================================================"
echo "P0 Verification Summary"
echo "============================================================"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: ALL P0 TESTS PASSED"
else
    echo "  RESULT: $FAIL test(s) need attention"
fi
echo ""
echo "Next steps:"
echo "  - Review P0-3a Test 2 (attention cardinality — known limitation)"
echo "  - Review P0-3b vis-only loss delta (expected ≈ 0)"
echo "  - If P0-2 Test 1 passes: adapter correct, narrow-TW = genuine hardness"
echo "  - Proceed to P0-5 Method Freeze"
