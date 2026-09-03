#!/bin/bash
# ============================================================
# P0-2: Baseline Solver Audit (redirect)
# ============================================================
# Superseded by run_baseline_audit.sh — kept for reference.
# ============================================================
echo "P0-2 audit has moved. Please use:"
echo "  bash run_baseline_audit.sh        # full audit"
echo "  bash run_baseline_audit.sh smoke  # quick smoke test"
echo ""
echo "Running new audit now..."
exec bash "$(dirname "$0")/run_baseline_audit.sh" "$@"
