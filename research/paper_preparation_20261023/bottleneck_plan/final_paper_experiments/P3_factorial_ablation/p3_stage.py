"""P3 远端准备：按 SHA 复制 26 例输入 → 核心准入。"""
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
UNI = HERE.parents[1] / "_maskco_unified_external_20261120"

BATCH_DIRS = {"prospective01": UNI / "prospective01", "retail01": UNI / "retail01"}
CASE_BATCH = {}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stage_inputs():
    registry = read(HERE / "heldout_sources.json")
    entries = registry["derived_instances"]
    assert len(entries) == 26 and set(entries) == set(registry["cases"])
    for batch, d in BATCH_DIRS.items():
        src = read(d / "heldout_sources.json")
        for case in src["derived_instances"]:
            CASE_BATCH[case] = (d, src["derived_instances"][case]["file"])
    if len(CASE_BATCH) != 32 or not set(entries) <= set(CASE_BATCH):
        raise ValueError("P3 registry not covered by frozen batch registries")
    (HERE / "inputs").mkdir(exist_ok=True)
    for case, entry in entries.items():
        src_dir, src_file = CASE_BATCH[case]
        src_path = src_dir / src_file
        raw = src_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen input SHA differs: " + case)
        dst = HERE / "inputs" / Path(entry["file"]).name
        dst.write_bytes(raw)
    print("26 frozen inputs copied with SHA verification", flush=True)


if __name__ == "__main__":
    stage_inputs()
    from unified_runner import available
    reserved = set()
    for name in ("benchmark01", "prospective01", "retail01", "trajectory01", "origin01"):
        p = UNI / name / "status.json"
        if p.is_file():
            state = json.loads(p.read_text())
            if not state["end"]:
                reserved.update(state["cores"])
    eligible, physical, ram = available(48, reserved)
    count = min(24, len(eligible) - 4)
    if count < 2:
        raise RuntimeError("Not enough cores")
    r = dict(cores=eligible[:count], retained=eligible[count:], other_jobs=sorted(reserved),
             physical=physical, available_ram_gib=ram)
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
