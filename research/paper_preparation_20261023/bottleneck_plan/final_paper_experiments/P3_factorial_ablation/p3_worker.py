# -*- coding: utf-8 -*-
"""P3 双因素消融工作器（作为 future_dispatch.py 部署）。

在当前锁版实现（冻结 carryprobe01/pool_search.py）上仅切换 split_enabled：
  ORDER_SPLIT  = HYBRID + 每三轮跨路线访问顺序搜索与动态规划经济分车；
  HYBRID       = 完全相同的掩码、预测复用、三种候选排序、两种重插算子、经济试选，
                 仅关闭跨车分车分支（split_enabled=False）。
输入注册沿用冻结 xtw_adapter 格式，由 heldout_sources.json 覆盖 26 个前瞻案例。
"""
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(row, method, block, affinity, initial=None):
    if (row["case"], row["seed"], row["method"], row["role"]) != (block["case"], block["seed"], method, "validation"):
        raise ValueError("P3 cell identity mismatch")
    if len(affinity) != 1 or row["cpu"] != affinity or row["status"] != "AUDITED" or not row["endpoint"]["independently_audited"]:
        raise ValueError("Single-core or full-fleet audit mismatch")
    if row["timing"]["budget_s"] != 60 or not 0 <= row["selected_s"] <= 60 or row["U"] < row["initial_U"] - 1e-6:
        raise ValueError("Time limit or incumbent mismatch")
    if not math.isclose(row["U"], row["endpoint"]["net_benefit_U"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Economic objective mismatch")
    if initial is not None and (row["initial_routes"] != initial[0] or abs(row["initial_U"] - initial[1]) > 1e-6):
        raise ValueError("Paired initial fleet differs")


def main():
    idx, method = sys.argv[1:]
    cfg = read(HERE / "phase.json")
    block = cfg["blocks"][int(idx)]
    if method not in block["methods"] or not set(block["methods"]) <= {"HYBRID", "HYBRID_D"} or cfg["seconds"] != 60:
        raise ValueError("Unregistered method or wall time")
    case, seed = block["case"], block["seed"]
    origin = PROJECT / "_maskco_cross_dataset_20261119"
    frozen = PROJECT / "_maskco_perishable_external_20261119" / "carryprobe01"
    sys.path.insert(0, str(origin))
    import compare
    registry = read(HERE / "heldout_sources.json")
    if case not in registry["derived_instances"]:
        raise ValueError("Unregistered P3 case: " + case)
    compare.CFG.update(derived_instances=registry["derived_instances"], paths=dict(workspace=str(HERE)))
    sys.path.insert(0, str(frozen))
    import pool_search
    started = time.perf_counter()
    lib = pool_search.library()
    access, core, source, locked = compare.setup("SLOT")
    from timed_search import single_cpu_guard
    affinity = single_cpu_guard()
    policy, scope = compare.load("SLOT", access, core, source, locked, case)
    setup = time.perf_counter() - started
    with access.restricted_scope(scope):
        row = pool_search.solve(lib, core, locked, policy, case, "validation", seed, method, 60)
    row["cpu"] = affinity
    row["setup_s"] = setup
    check(row, method, block, affinity)
    with (HERE / "out" / f"{idx}_{method}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)


if __name__ == "__main__":
    main()
