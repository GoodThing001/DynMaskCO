# -*- coding: utf-8 -*-
"""P4 轨迹工作器（作为 unified_worker.py 部署）。

直接安装搜索时钟记录补丁（覆盖 MASK/SLOT 与 external_common 两条链）；
构造式方法若无搜索时钟事件，轨迹记 None 并标记 no_search_clock=True（如实报告）。
"""
import time

ENTRY_S = time.perf_counter()

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPEAT = PROJECT / "C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace"
RUN10 = PROJECT / "C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_stage3_20261022/run10_six_methods_dev72/recovery_preparation/workspace"
sys.path.insert(0, str(REPEAT / "C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"))
sys.path.append(str(RUN10 / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))

import future_dispatch
from future_dispatch import check

CUTOFFS = (5, 15, 30, 60)


def snapshots(events, cutoffs=CUTOFFS):
    out = {}
    for t in cutoffs:
        eligible = [e for e in events if e["elapsed_s"] <= t]
        selected = max(eligible, key=lambda e: e["score"]["net_benefit_U"], default=None)
        out[str(t)] = None if selected is None else dict(
            U=selected["score"]["net_benefit_U"], accepted_s=selected["elapsed_s"], routes=selected["routes"])
    return out


def main(index, method):
    cfg = json.loads((HERE / "phase.json").read_text())
    block = cfg["blocks"][int(index)]
    os.environ["RECENT5_ASSETS"] = str(PROJECT / "_maskco_recent5_20261120" / "assets")
    import run_dev72_clock
    make_clock = run_dev72_clock.clock_type
    clocks = []

    def recording_clock(base):
        class HeldClock(make_clock(base)):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                clocks.append(self)
        return HeldClock

    with ExitStack() as stack:
        stack.enter_context(patch.object(run_dev72_clock, "clock_type", recording_clock))
        row = future_dispatch.execute(method, block, cfg)
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    if len(clocks) == 1:
        clock = clocks[0]
        history = snapshots(clock.events)
        if not history["60"] or abs(history["60"]["U"] - row["U"]) > 1e-6:
            raise ValueError("60-second history does not reproduce final incumbent")
        preparation_s = clock.start - ENTRY_S
        remaining_s = 60 - preparation_s
        cold = snapshots(clock.events, [remaining_s])[str(remaining_s)]
        row.update(trajectory=history, budget_s=cfg["seconds"],
                   entry_to_search_s=preparation_s, cold_start_60=cold)
    else:
        row.update(trajectory=None, no_search_clock=True, budget_s=cfg["seconds"],
                   note="构造式方法：无逐时点接受事件的连续搜索时钟，轨迹不适用")
    with (HERE / "out" / f"{index}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
