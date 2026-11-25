# -*- coding: utf-8 -*-
"""P4 统一调度器（作为 future_dispatch.py 部署）。

- MASK/SLOT：冻结 compare 链 + P4 派生实例注册表（角色 validation）；
- 其余 16 法：冻结 external_common 链（角色 external_test，locked_method 锁版要求）。
两链共用 P4 heldout_sources.json（9 个 Solomon 冷链投影，SHA 锁定）。
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(row, method, block, affinity, initial=None):
    role_ok = row["role"] in ("validation", "external_test")
    if (row["case"], row["seed"], row["method"]) != (block["case"], block["seed"], method) or not role_ok:
        raise ValueError("P4 cell identity mismatch")
    if len(affinity) != 1 or row["cpu"] != affinity or row["status"] != "AUDITED" or not row["endpoint"]["independently_audited"]:
        raise ValueError("Single-core or full-fleet audit mismatch")
    timing = row.get("timing", {})
    budget = timing.get("budget_s", row.get("budget_s", 60))
    if budget != 60 or not 0 <= row["selected_s"] <= 60 or row["U"] < row["initial_U"] - 1e-6:
        raise ValueError("Time limit or incumbent mismatch")
    if not math.isclose(row["U"], row["endpoint"]["net_benefit_U"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Economic objective mismatch")
    if initial is not None and (row["initial_routes"] != initial[0] or abs(row["initial_U"] - initial[1]) > 1e-6):
        raise ValueError("Paired initial fleet differs")


def execute_mask_slot(method, case, seed):
    import numpy as np
    import random
    import datetime as dt
    import time
    from contextlib import ExitStack
    from unittest.mock import patch
    origin = PROJECT / "_maskco_cross_dataset_20261119"
    sys.path.insert(0, str(origin))
    import compare
    registry = read(HERE / "heldout_sources.json")
    if case not in registry["derived_instances"]:
        raise ValueError("Unregistered P4 case: " + case)
    compare.CFG.update(derived_instances=registry["derived_instances"], paths=dict(workspace=str(HERE)))
    access, core, source, locked = compare.setup(method)
    import timed_search
    from run_dev72_clock import clock_type
    timed_search.single_cpu_guard()
    started = time.perf_counter()
    policy, scope = compare.load(method, access, core, source, locked, case)
    loading = time.perf_counter() - started
    random.seed(seed)
    np.random.seed(seed)
    role = "validation"
    with access.restricted_scope(scope):
        access.require_case(case, 0, role)
        start = time.perf_counter()
        utc = dt.datetime.now(dt.timezone.utc)
        cpu = time.process_time()
        task = core.ProfitTask(case, fold=0, role=role)
        clock = clock_type(timed_search.SearchClock)(task, start=start, utc=utc, seconds=60)
        clock.candidate_source = method
        provider = locked.RepairProvider(method, policy, False)
        with ExitStack() as stack:
            if method == "SLOT":
                locked.setup_operators(stack, provider, "SLOT")
            stack.enter_context(patch.object(timed_search, "SearchClock", lambda *a, **kw: clock))
            result = timed_search.guided_search(task, provider, "maskco_base", seconds=60,
                                                iterations=1000000, seed=seed, remove_count=10, topk=8)
        forwards = provider.forward_calls
        if result["failure"] or not result["selected_incumbent_in_budget"]:
            raise ValueError(str(result["failure"]))
        final = clock.events[-1]
        fresh = core.ProfitTask(case, fold=0, role=role)
        audit = fresh.evaluate(final["routes"], audit=True)
        if (not audit["independently_audited"] or final["elapsed_s"] > 60
                or abs(audit["net_benefit_U"] - result["final"]["net_benefit_U"]) > 1e-6):
            raise ValueError("Unverified 60-second endpoint")
        cache = task.route.cache_info()
        return dict(case=case, seed=seed, method=method, role=role, status="AUDITED", U=audit["net_benefit_U"],
                    endpoint=audit, selected_s=final["elapsed_s"], source=final["source"],
                    initial_U=clock.events[0]["score"]["net_benefit_U"], initial_routes=clock.events[0]["routes"],
                    routes=final["routes"], iterations=result["iterations"], forwards=forwards,
                    setup_s=loading, requests=cache.hits + cache.misses, cpu_s=time.process_time() - cpu,
                    timing=result["timing"], cpu=list(os.sched_getaffinity(0)))


def execute(method, block, cfg):
    if method in ("MASK", "SLOT"):
        return execute_mask_slot(method, block["case"], block["seed"])
    if method in ("RouteFinder", "CaDA", "MVMoE", "RRNCO", "PyVRP"):
        import recent_worker
        return recent_worker.execute(method, block["case"], block["seed"], cfg["seconds"])
    import external_common
    return external_common.endpoint(method, block["case"], block["seed"], cfg)


def main():
    idx, method = sys.argv[1:]
    cfg = read(HERE / "phase.json")
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or len(block["methods"]) != 18 or cfg["seconds"] != 60:
        raise ValueError("Unregistered P4 method or wall time")
    row = execute(method, block, cfg)
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    with (HERE / "out" / f"{idx}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main()
