"""Run one archived development implementation on the registered original input."""
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ARMS = ("MASK", "SLOT", "ORDER_SPLIT_D", "ORDER_SPLIT")

def check(row, method, block, affinity, initial=None):
    if (row["case"], row["seed"], row["method"], row["role"]) != (block["case"], block["seed"], method, "validation"):
        raise ValueError("Development cell identity mismatch")
    if len(affinity) != 1 or row["cpu"] != affinity or row["status"] != "AUDITED" or not row["endpoint"]["independently_audited"]:
        raise ValueError("Single-core or full-fleet audit mismatch")
    if row["timing"]["budget_s"] != 60 or not 0 <= row["selected_s"] <= 60 or row["U"] < row["initial_U"]-1e-6:
        raise ValueError("Time limit or incumbent mismatch")
    if not math.isclose(row["U"], row["endpoint"]["net_benefit_U"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Economic objective mismatch")
    if initial is not None and (row["initial_routes"] != initial[0] or abs(row["initial_U"]-initial[1]) > 1e-6):
        raise ValueError("Paired initial fleet differs")

def main():
    idx, method = sys.argv[1:]
    cfg = json.loads((HERE/"phase.json").read_text())
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or set(block["methods"]) != set(ARMS) or cfg["seconds"] != 60:
        raise ValueError("Unregistered method or wall time")
    case, seed = block["case"], block["seed"]
    origin = PROJECT/"_maskco_cross_dataset_20261119"
    frozen = PROJECT/"_maskco_perishable_external_20261119"/"orderdev01"
    sys.path.insert(0, str(origin))
    import compare
    if method in ("MASK", "SLOT"):
        row = compare.execute(method, case, seed, False)
        row["role"] = "validation"
    else:
        sys.path.insert(0, str(frozen))
        import block_search
        start = time.perf_counter()
        lib = block_search.library()
        access, core, source, locked = compare.setup("SLOT")
        policy, scope = compare.load("SLOT", access, core, source, locked, case)
        setup = time.perf_counter() - start
        with access.restricted_scope(scope):
            row = block_search.solve(lib, core, locked, policy, case, "validation", seed, method, 60)
        row["cpu"] = sorted(os.sched_getaffinity(0))
        row["setup_s"] = setup
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    with (HERE/"out"/f"{idx}_{method}.json").open("x") as file:
        json.dump(row, file, allow_nan=False)

if __name__ == "__main__":
    main()

