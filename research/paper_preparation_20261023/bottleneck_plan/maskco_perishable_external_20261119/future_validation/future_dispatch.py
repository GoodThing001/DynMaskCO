"""Run the frozen 18-method validation one case/seed block per physical core."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FLEET = {"FLEET_PRICE", "FLEET_PRICE_D"}
RECENT = {"RouteFinder", "CaDA", "MVMoE", "RRNCO", "PyVRP"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def execute(method, block, cfg):
    if method in FLEET:
        from fleet_timed_worker import endpoint
        return endpoint(method, block, cfg)
    if method in RECENT:
        import recent_worker
        return recent_worker.execute(method, block["case"], block["seed"], cfg["seconds"])
    import external_common
    return external_common.endpoint(method, block["case"], block["seed"], cfg)


def check(row, method, block, affinity, initial=None):
    if (row["method"], row["case"], row["seed"], row["role"]) != (
            method, block["case"], block["seed"], "external_test"):
        raise ValueError(f"Wrong endpoint identity for {method}")
    if (row["cpu"] != affinity or len(affinity) != 1 or row["status"] != "AUDITED"
            or not row["endpoint"]["independently_audited"] or not 0 <= row["selected_s"] <= 60):
        raise ValueError(f"Core, cutoff or endpoint audit mismatch for {method}")
    u, first = row["U"], row["initial_U"]
    if (not math.isfinite(u) or not math.isfinite(first) or u < first - 1e-6
            or not math.isclose(u, row["endpoint"]["net_benefit_U"], rel_tol=0, abs_tol=1e-6)):
        raise ValueError(f"Endpoint net benefit mismatch for {method}")
    if initial is not None and (row["initial_routes"] != initial[0] or abs(initial[1] - first) > 1e-6):
        raise ValueError(f"Paired initializer differs for {method}")


def main():
    index = int(sys.argv[1])
    cfg = read(HERE / "phase.json")
    lock = read(HERE / "future_lock.json")
    if lock["decision"] != "LOCK_FOR_VALIDATION" or lock["arms"] != sorted(FLEET) or cfg["seconds"] != 60:
        raise ValueError("Locked 60-second validation required")
    block = cfg["blocks"][index]
    methods = block["methods"]
    if len(methods) != 18 or set(methods) != set(cfg["interpreters"]):
        raise ValueError("Incomplete 18-method block")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise ValueError("An assigned physical core is required")
    out = HERE / "out"
    if len(sys.argv) == 3:
        method = sys.argv[2]
        if method not in methods:
            raise ValueError(method)
        os.environ["RECENT5_ASSETS"] = str(HERE.parents[1] / "_maskco_recent5_20261120" / "assets")
        row = execute(method, block, cfg)
        check(row, method, block, affinity)
        row.setdefault("budget_s", cfg["seconds"])
        with (out / f"{index}_{method}.cell.json").open("x", encoding="utf-8") as stream:
            json.dump(row, stream, allow_nan=False)
        return
    rows, initial = {}, None
    for method in methods:
        subprocess.run([cfg["interpreters"][method], "-B", str(HERE / "future_dispatch.py"),
                        str(index), method], check=True)
        row = read(out / f"{index}_{method}.cell.json")
        check(row, method, block, affinity, initial)
        if initial is None:
            initial = row["initial_routes"], row["initial_U"]
        rows[method] = row
    with (out / f"{index}.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(case=block["case"], seed=block["seed"], arms=rows), stream, allow_nan=False)


if __name__ == "__main__":
    main()
