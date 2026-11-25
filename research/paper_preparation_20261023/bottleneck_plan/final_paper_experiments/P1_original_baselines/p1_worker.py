"""P1 MASK/SLOT 统一实验工作器（作为 future_dispatch.py 部署）。

在冻结 `_maskco_cross_dataset_20261119/compare.py` 的 MASK/SLOT 求解链上
仅扩展案例注册表（derived_instances），不修改任何冻结源码与搜索机制：
  - 冻结 f0_r0 权重（locked.scope_for -> repeat02 workspace panel/training/f0_r0）；
  - 原掩码数量、候选筛选与重插规则（guided_search remove_count=10, topk=8, maskco_base）；
  - SLOT 的最短可行插位保护由 locked.setup_operators 安装；
  - 60 秒单物理核在线预算，模型准备不计入，最终解经独立审计后才接受。
"""
import datetime as dt
import json
import math
import os
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(row, method, block, affinity, initial=None):
    if (row["case"], row["seed"], row["method"], row["role"]) != (block["case"], block["seed"], method, "validation"):
        raise ValueError("P1 cell identity mismatch")
    if len(affinity) != 1 or row["cpu"] != affinity or row["status"] != "AUDITED" or not row["endpoint"]["independently_audited"]:
        raise ValueError("Single-core or full-fleet audit mismatch")
    if row["timing"]["budget_s"] != 60 or not 0 <= row["selected_s"] <= 60 or row["U"] < row["initial_U"] - 1e-6:
        raise ValueError("Time limit or incumbent mismatch")
    if not math.isclose(row["U"], row["endpoint"]["net_benefit_U"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Economic objective mismatch")
    if initial is not None and (row["initial_routes"] != initial[0] or abs(row["initial_U"] - initial[1]) > 1e-6):
        raise ValueError("Paired initial fleet differs")


def execute(method, case, seed):
    if method not in ("MASK", "SLOT"):
        raise ValueError("P1 only runs MASK/SLOT")
    import numpy as np
    origin = PROJECT / "_maskco_cross_dataset_20261119"
    sys.path.insert(0, str(origin))
    import compare
    # 扩展案例注册表：62 个统一16法案例的输入按原冻结适配器格式注册。
    registry = read(HERE / "heldout_sources.json")
    if case not in registry["derived_instances"]:
        raise ValueError("Unregistered P1 case: " + case)
    compare.CFG.update(derived_instances=registry["derived_instances"], paths=dict(workspace=str(HERE)))
    access, core, source, locked = compare.setup(method)
    import timed_search
    from run_dev72_clock import clock_type, semantic_check
    timed_search.single_cpu_guard()
    semantic_check(timed_search.SearchClock)
    started = time.perf_counter()
    policy, scope = compare.load(method, access, core, source, locked, case)
    loading = time.perf_counter() - started
    random.seed(seed)
    np.random.seed(seed)
    role = "validation"
    seconds = 60
    with access.restricted_scope(scope):
        access.require_case(case, 0, role)
        start = time.perf_counter()
        utc = dt.datetime.now(dt.timezone.utc)
        cpu = time.process_time()
        task = core.ProfitTask(case, fold=0, role=role)
        clock = clock_type(timed_search.SearchClock)(task, start=start, utc=utc, seconds=seconds)
        clock.candidate_source = method
        provider = locked.RepairProvider(method, policy, method in ("DIST", "DIST_SLOT"))
        with ExitStack() as stack:
            if method == "SLOT":
                locked.setup_operators(stack, provider, "SLOT")
            stack.enter_context(patch.object(timed_search, "SearchClock", lambda *a, **kw: clock))
            result = timed_search.guided_search(task, provider, "maskco_base", seconds=seconds,
                                                iterations=1000000, seed=seed, remove_count=10, topk=8)
        forwards = provider.forward_calls
        if result["failure"] or not result["selected_incumbent_in_budget"]:
            raise ValueError(str(result["failure"]))
        final = clock.events[-1]
        fresh = core.ProfitTask(case, fold=0, role=role)
        audit = fresh.evaluate(final["routes"], audit=True)
        if (not audit["independently_audited"] or final["elapsed_s"] > seconds
                or abs(audit["net_benefit_U"] - result["final"]["net_benefit_U"]) > 1e-6):
            raise ValueError("Unverified 60-second endpoint")
        cache = task.route.cache_info()
        return dict(case=case, seed=seed, method=method, role=role, status="AUDITED", U=audit["net_benefit_U"],
                    endpoint=audit, selected_s=final["elapsed_s"], source=final["source"],
                    initial_U=clock.events[0]["score"]["net_benefit_U"], initial_routes=clock.events[0]["routes"],
                    routes=final["routes"], iterations=result["iterations"], forwards=forwards,
                    setup_s=loading, requests=cache.hits + cache.misses, cpu_s=time.process_time() - cpu,
                    timing=result["timing"], cpu=list(os.sched_getaffinity(0)))


def main():
    idx, method = sys.argv[1:]
    cfg = read(HERE / "phase.json")
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or set(block["methods"]) != {"MASK", "SLOT"} or cfg["seconds"] != 60:
        raise ValueError("Unregistered method or wall time")
    row = execute(method, block["case"], block["seed"])
    check(row, method, block, sorted(os.sched_getaffinity(0)))
    with (HERE / "out" / f"{idx}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main()
