# -*- coding: utf-8 -*-
"""下载 P5 官方实例与 BKS 参考到本地 data/ 以便离线独立核验。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
JOBS = [("cvrplib/instances", "data/cvrplib/instances", (".vrp",)),
        ("cvrplib/bks", "data/cvrplib/bks", (".sol",)),
        ("solomon/instances", "data/solomon/instances", (".vrp",)),
        ("solomon/bks", "data/solomon/bks", (".sol",))]

transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    with client.open_sftp() as s:
        for rem, loc, exts in JOBS:
            dest = V2 / loc
            dest.mkdir(parents=True, exist_ok=True)
            names = [n.filename for n in s.listdir_iter(P5 + "/" + rem)
                     if n.filename.endswith(exts)]
            done = 0
            for name in names:
                target = dest / name
                if target.exists() and target.stat().st_size > 0:
                    done += 1
                    continue
                try:
                    s.get(P5 + "/" + rem + "/" + name, str(target))
                    done += 1
                except OSError as exc:
                    print("skip", name, exc)
            print("%-22s %d/%d" % (loc, done, len(names)))
finally:
    client.close()
