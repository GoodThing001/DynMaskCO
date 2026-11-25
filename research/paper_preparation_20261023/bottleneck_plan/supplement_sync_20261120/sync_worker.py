"""Run one frozen method for one paired case and seed on a pinned physical core."""
import json
import math
import os
from pathlib import Path
import sys
import time

import external_common as common

HERE = Path(__file__).resolve().parent
METHODS = {"ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS", "PyVRP"}


def execute(method, block, cfg):
    if method not in METHODS or method not in block["methods"]:
        raise ValueError(method)
    if cfg["seconds"] != 60 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("The registered task requires one pinned core and 60 online seconds")
    started = time.perf_counter()
    if method == "PyVRP":
        from recent_worker import execute as recent_execute
        row = recent_execute(method, block["case"], block["seed"], cfg["seconds"])
    else:
        row = common.endpoint(method, block["case"], block["seed"], cfg)
    if ((row["case"], row["seed"], row["method"]) != (block["case"], block["seed"], method)
            or row["status"] != "AUDITED" or not row["endpoint"]["independently_audited"]
            or not math.isfinite(row["U"]) or not 0 <= row["selected_s"] <= cfg["seconds"]
            or abs(row["U"] - row["endpoint"]["net_benefit_U"]) > 1e-6):
        raise ValueError("The selected incumbent failed identity, time, or strict objective audit")
    row.update(budget_s=cfg["seconds"], worker_execute_s=time.perf_counter() - started,
               batch="sync01", setup_timing_scope="existing adapter preparation; process total is recorded by controller")
    return row


def main():
    index, method = sys.argv[1:]
    cfg = common.read(HERE / "phase.json")
    block = cfg["blocks"][int(index)]
    row = execute(method, block, cfg)
    with (HERE / "out" / f"{index}_{method}.json").open("x", encoding="utf-8") as stream:
        json.dump(row, stream, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
