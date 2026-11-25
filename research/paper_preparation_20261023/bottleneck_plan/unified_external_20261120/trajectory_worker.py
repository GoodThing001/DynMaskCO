"""Read accepted incumbent histories after the unchanged 60-second search finishes."""
import time

ENTRY_S = time.perf_counter()

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys

import external_common
from future_dispatch import execute, check


HERE = Path(__file__).resolve().parent
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
    cfg = json.loads((HERE/"phase.json").read_text())
    block = cfg["blocks"][int(index)]
    os.environ["RECENT5_ASSETS"] = str(HERE.parents[1]/"_maskco_recent5_20261120"/"assets")
    original = external_common.install
    clocks = []
    with ExitStack() as stack:
        def install(*args, **kwargs):
            result = original(*args, **kwargs)
            import run_dev72_clock
            make_clock = run_dev72_clock.clock_type
            def recording_clock(base):
                class HeldClock(make_clock(base)):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        clocks.append(self)
                return HeldClock
            stack.enter_context(patch.object(run_dev72_clock, "clock_type", recording_clock))
            return result
        stack.enter_context(patch.object(external_common, "install", install))
        row = execute(method, block, cfg)
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    if len(clocks) != 1:
        raise ValueError(f"Expected one active search clock, got {len(clocks)}")
    clock = clocks[0]
    history = snapshots(clock.events)
    if not history["60"] or abs(history["60"]["U"]-row["U"]) > 1e-6:
        raise ValueError("60-second history does not reproduce final incumbent")
    preparation_s = clock.start - ENTRY_S
    remaining_s = 60 - preparation_s
    cold = snapshots(clock.events, [remaining_s])[str(remaining_s)]
    row.update(trajectory=history, budget_s=cfg["seconds"],
               entry_to_search_s=preparation_s, cold_start_60=cold)
    with (HERE/"out"/f"{index}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
