"""P2 远端准备：从零售三批复用部署文件与八城输入，隔离核心后准入。"""
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNI = HERE.parents[1] / "_maskco_unified_external_20261120"


def main():
    source = UNI / "retail01"
    phase_path = HERE / "phase.json"
    config = json.loads(phase_path.read_text(encoding="utf-8"))
    base = json.loads((source / "phase.json").read_text(encoding="utf-8"))
    excluded = set()
    for stage in ("benchmark01", "prospective01", "retail01", "trajectory01", "origin01"):
        p = UNI / stage / "status.json"
        if p.exists():
            peer = json.loads(p.read_text(encoding="utf-8"))
            if not peer["end"]:
                excluded.update(peer["cores"])
    config["excluded_cores"] = sorted(excluded)
    from unified_runner import available
    idle, _, _ = available(48, excluded)
    if len(idle) <= 4:
        raise RuntimeError("Not enough unoccupied cores to retain four for other work")
    config["max_cores"] = min(config["max_cores"], len(idle) - 4)
    phase_path.write_text(json.dumps(config), encoding="utf-8")
    files = set(base.get("reused_code", base.get("reused_recent_files", []))) | {
        "future_dispatch.py", "locked_method.json", "heldout_admission.json", "heldout_sources.json",
        "external_common.py", "run_dev72_clock.py", "pool_search.py", "block_split.py", "calibrated.json",
        "calibration.py", "masked_repair.cpp", "masked_repair.so", "split_order.cpp", "split_order.so"}
    files.discard("unified_worker.py")  # 本批轨迹工作器必须覆盖默认工作器
    for name in files:
        src = source / name
        if src.is_file():
            shutil.copy2(src, HERE / name)
    # 预编译内核从冻结 carryprobe01 复用（与统一批次同一编译产物身份）
    frozen = HERE.parents[1] / "_maskco_perishable_external_20261119" / "carryprobe01"
    for name in ("masked_repair.so", "split_order.so"):
        src = frozen / name
        if src.is_file():
            shutil.copy2(src, HERE / name)
    # 覆盖为 P2 注册表（同一 8 城输入，仅登记身份）
    meta = json.loads((HERE / "p2_heldout_sources.json").read_text(encoding="utf-8"))
    (HERE / "heldout_sources.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in sorted({b["case"] for b in config["blocks"]}):
        rel = meta["derived_instances"][case]["file"]
        dst = HERE / rel
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(source / rel, dst)
    print("P2 staged: 8 cities, files reused from retail01. Excluded cores:", sorted(excluded), flush=True)


if __name__ == "__main__":
    main()
