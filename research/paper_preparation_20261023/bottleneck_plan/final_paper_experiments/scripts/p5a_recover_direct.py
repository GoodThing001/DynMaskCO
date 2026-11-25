# -*- coding: utf-8 -*-
"""远端收据直取回收（sftp 逐文件，避免 tar 需要额外磁盘）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5_REMOTE = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
P5_LOCAL = HERE.parent / "P5_standard_vrp"


def main(ws, dst_name, prefix=""):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            remote = P5_REMOTE + "/" + ws
            out_dir = P5_LOCAL / dst_name / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            names = [n for n in s.listdir(remote + "/out") if n.endswith(".json")]
            n = 0
            for name in names:
                s.get(remote + "/out/" + name, str(out_dir / (prefix + name)))
                n += 1
            for meta in ("phase.json", "status.json"):
                try:
                    s.get(remote + "/" + meta, str(P5_LOCAL / dst_name / meta))
                except IOError:
                    pass
        print("%-16s -> %-24s cells=%d prefix=%s" % (ws, dst_name, n, prefix or "-"))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
