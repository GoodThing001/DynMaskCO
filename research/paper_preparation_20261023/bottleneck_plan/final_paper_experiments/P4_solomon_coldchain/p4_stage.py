# -*- coding: utf-8 -*-
"""P4 远端准备：导出 9 例 Solomon 冷链投影 → 复制统一链文件 → 初解准入 → 核心准入。"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
UNI = PROJECT / "_maskco_unified_external_20261120"
WS = PROJECT / "C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace"

CASES = ["c105", "c106", "c107", "c108", "c109", "c205", "c206", "c207", "c208"]


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def dump_inputs():
    """从冻结工作区导出 9 例冷链投影，SHA 与几何组按 fold0 定义锁定。"""
    sp = WS / "C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"
    sys.path.insert(0, str(sp))
    from profit_core import fold_definition, load_original, load_config
    fd0 = fold_definition(0)
    geo = {}
    for c in CASES:
        if c.startswith("c1"):
            geo[c] = fd0["test_geometry"]
        else:
            geo[c] = fd0["validation_geometry"]
    (HERE / "inputs").mkdir(exist_ok=True)
    derived = {}
    for c in CASES:
        ins = load_original(c, load_config())
        rows = ins["nodes"]
        payload = dict(case=c, nodes=rows,
                       note="VRPTWQ-Solomon 冷链投影；来自冻结工作区 load_original 与 fold0 几何定义")
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        (HERE / "inputs" / (c + ".json")).write_bytes(raw)
        derived[c] = dict(file="inputs/" + c + ".json", sha256=sha, geometry_sha256=geo[c])
    registry = dict(status="PASS", cases=CASES, derived_instances=derived,
                    note="P4 VRPTWQ-Solomon 冷链投影注册表（9 例，两几何组）")
    (HERE / "heldout_sources.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print("9 Solomon projections dumped with SHA", flush=True)


def copy_chain():
    source = UNI / "retail01"
    base = read(source / "phase.json")
    files = set(base.get("reused_code", [])) | {"external_common.py", "future_dispatch.py", "locked_method.json",
                                                "run_dev72_clock.py", "pool_search.py", "block_split.py",
                                                "calibrated.json", "calibration.py", "masked_repair.cpp",
                                                "masked_repair.so", "split_order.cpp", "split_order.so"}
    files.discard("unified_worker.py")
    files.discard("future_dispatch.py")  # 本批 p4_dispatch 覆盖调度器
    for name in files:
        src = source / name
        if src.is_file():
            shutil.copy2(src, HERE / name)
    # run_dev72_clock 在冻结工作区；从轨迹批次复制
    for name in ("run_dev72_clock.py",):
        src = UNI / "trajectory01" / name
        if src.is_file():
            shutil.copy2(src, HERE / name)
    print("unified chain files staged", flush=True)


def admission():
    """共同初解准入：每例用冻结链构造任务并运行初始器，记录可行性与车辆数。"""
    import external_common
    manifest = read(HERE / "locked_method.json")
    registry = read(HERE / "heldout_sources.json")
    out = dict(status="PASS", cases={})
    source_arm, is_new, access, core, source, locked = external_common.install("ORDER_SPLIT", manifest, False)
    import timed_search
    from run_dev72_clock import clock_type
    import datetime as dt
    for c in CASES:
        scope = access.Scope("p4-admission", frozenset([(0, "external_test", c)]),
                             frozenset(["initialize", "infer", "search"]))
        with access.restricted_scope(scope):
            task = core.ProfitTask(c, fold=0, role="external_test")
            clock = clock_type(timed_search.SearchClock)(task, start=time.perf_counter(),
                                                         utc=dt.datetime.now(dt.timezone.utc), seconds=60)
            first = clock.initialize()
            audit = core.ProfitTask(c, fold=0, role="external_test").evaluate(first, audit=True)
        if not audit["independently_audited"]:
            raise ValueError("initializer audit failed for " + c)
        out["cases"][c] = dict(initializer_routes=len(first),
                               infeasible_singletons={},
                               initial_U=audit["net_benefit_U"])
    (HERE / "heldout_admission.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("common initializer admitted for 9 cases", flush=True)


def reserve():
    from unified_runner import available
    reserved = set()
    for name in ("benchmark01", "prospective01", "retail01", "trajectory01", "origin01"):
        p = UNI / name / "status.json"
        if p.is_file():
            st = read(p)
            if not st["end"]:
                reserved.update(st["cores"])
    eligible, physical, ram = available(48, reserved)
    count = min(30, len(eligible) - 4)
    if count < 2:
        return None
    return dict(cores=eligible[:count], retained=eligible[count:], other_jobs=sorted(reserved),
                physical=physical, available_ram_gib=ram)


if __name__ == "__main__":
    dump_inputs()
    copy_chain()
    admission()
    while not (r := reserve()):
        time.sleep(20)
    (HERE / "resources.json").write_text(json.dumps(r))
    cfg = read(HERE / "phase.json")
    cfg["max_cores"] = len(r["cores"])

    def identity(core):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology")
        return tuple((path / n).read_text().strip() for n in ("physical_package_id", "core_id"))

    allowed = {identity(core) for core in r["cores"]}
    cfg["excluded_cores"] = sorted(core for core in os.sched_getaffinity(0) if identity(core) not in allowed)
    (HERE / "phase.json").write_text(json.dumps(cfg))
    print("admitted", len(r["cores"]), "cores", flush=True)
