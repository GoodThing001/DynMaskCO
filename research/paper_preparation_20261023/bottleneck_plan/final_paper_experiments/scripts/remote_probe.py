# -*- coding: utf-8 -*-
"""P1/P2/P3 执行前远端只读探测：冻结目录、批次状态、磁盘、核心与tmux。"""
import json
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
PROJECT = "/home/hzeng/project/MASKCO-Main"

PROBE = r'''
import json, os, shutil, subprocess, psutil
from pathlib import Path
r = Path("/home/hzeng/project/MASKCO-Main")
uni = r / "_maskco_unified_external_20261120"
names = ["_maskco_cross_dataset_20261119", "_maskco_perishable_external_20261119",
         "_maskco_recent5_20261120", "_maskco_unified_external_20261120", "_ofs01r",
         "_maskco_distance_slot_20261119"]
frozen = {n: (r/n).is_dir() for n in names}
repeat_ws = r/"C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace"
stages = {}
for s in ("benchmark01","prospective01","retail01","trajectory01","origin01"):
    p = uni/s/"status.json"
    if p.is_file():
        st = json.loads(p.read_text())
        stages[s] = dict(end=st.get("end") is not None, complete=st.get("complete"), failed=len(st.get("failed",[])))
tmux = subprocess.run(["tmux","ls"],capture_output=True,text=True).stdout.splitlines()
load = psutil.cpu_percent(interval=1,percpu=True)
groups={}
for i in os.sched_getaffinity(0):
    p=Path(f"/sys/devices/system/cpu/cpu{i}/topology")
    k=tuple((p/f).read_text().strip() for f in ("physical_package_id","core_id"))
    groups.setdefault(k,[]).append(i)
eligible=sum(max(load[i] for i in g)<40 for g in groups.values())
print(json.dumps(dict(disk_gib=round(shutil.disk_usage(r).free/2**30,2),
                      ram_gib=round(psutil.virtual_memory().available/2**30,2),
                      physical=len(groups), eligible=eligible,
                      frozen=frozen, repeat_workspace=repeat_ws.is_dir(),
                      orderdev01=(r/"_maskco_perishable_external_20261119/orderdev01").is_dir(),
                      pool_search=(r/"_maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py").is_file(),
                      block_split=(r/"_maskco_perishable_external_20261119/reserved_validation/solver/block_split.py").is_file(),
                      stages=stages, tmux=tmux)))
'''


def main():
    transport.record = lambda *a, **k: None  # 只读探测，不向历史目录写传输事件
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE))
    finally:
        client.close()
    info = json.loads(out)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    (HERE.parent / "audit" / "remote_preflight.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
