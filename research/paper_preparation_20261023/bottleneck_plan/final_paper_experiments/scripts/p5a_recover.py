# -*- coding: utf-8 -*-
"""回收远端 P5-A 工作区收据（out/*.json + status + phase + runner.log）到本地。"""
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"


def main(ws, out_name=None):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        base = P5 + "/" + ws
        out_dir = HERE.parent / "P5_standard_vrp" / ("results_" + (out_name or ws))
        out_dir.mkdir(exist_ok=True)
        transport.remote_exec(client, "cd " + base + " && tar -czf receipts.tar.gz --ignore-failed-read "
                              "out status.json phase.json runner.log")
        with client.open_sftp() as s:
            s.get(base + "/receipts.tar.gz", str(out_dir / "receipts.tar.gz"))
        with tarfile.open(out_dir / "receipts.tar.gz") as z:
            for m in z:
                p = (out_dir / m.name).resolve()
                if not p.is_relative_to(out_dir.resolve()):
                    raise ValueError(m.name)
                if m.isdir():
                    p.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    p.write_bytes(z.extractfile(m).read())
        n_json = len(list((out_dir / "out").glob("*.json")))
        print("recovered", out_dir, "cells:", n_json)
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
