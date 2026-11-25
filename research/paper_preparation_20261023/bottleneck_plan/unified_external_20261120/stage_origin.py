"""Admit the paired origin comparison after neighbouring core allocations are known."""
import json
import os
from pathlib import Path
import time
from unified_runner import available

HERE = Path(__file__).resolve().parent

def reserve():
    reserved = set()
    for name in ("prospective01", "trajectory01", "retail01", "benchmark01"):
        p = HERE.parent/name/"status.json"
        if not p.is_file():
            return None
        state = json.loads(p.read_text())
        if name == "prospective01" and not state["end"]:
            return None
        if not state["end"]:
            reserved.update(state["cores"])
    service = HERE.parent/"service01"
    if not (service/"exit_code").is_file():
        if not (service/"resource.json").is_file():
            return None
        service_cores = json.loads((service/"resource.json").read_text())["cores"]
        if len(service_cores) != 2:
            raise ValueError("Service allocation must contain two cores")
        reserved.update(service_cores)
    eligible, physical, ram = available(48, reserved)
    count = min(8, len(eligible) - 4)
    if count < 1:
        return None
    return dict(cores=eligible[:count], retained=eligible[count:], other_jobs=sorted(reserved),
                physical=physical, available_ram_gib=ram)

if __name__ == "__main__":
    while not (r := reserve()):
        time.sleep(20)
    (HERE/"resources.json").write_text(json.dumps(r))
    cfg = json.loads((HERE/"phase.json").read_text())
    cfg["max_cores"] = len(r["cores"])
    def identity(core):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology")
        return tuple((path/name).read_text().strip() for name in ("physical_package_id", "core_id"))
    allowed = {identity(core) for core in r["cores"]}
    cfg["excluded_cores"] = sorted(core for core in os.sched_getaffinity(0) if identity(core) not in allowed)
    (HERE/"phase.json").write_text(json.dumps(cfg))

