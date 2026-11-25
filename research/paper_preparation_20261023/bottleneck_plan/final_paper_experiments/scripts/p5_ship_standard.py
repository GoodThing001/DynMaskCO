# -*- coding: utf-8 -*-
"""上传 standard_vrp 模块到远端 P5 工作区。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SRC = HERE.parent / "P5_standard_vrp" / "standard_vrp"
REMOTE = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/standard_vrp"


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            import posixpath
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            makedirs(REMOTE)
            for f in sorted(SRC.rglob("*.py")):
                rel = f.relative_to(SRC).as_posix()
                d = posixpath.dirname(rel)
                cur = REMOTE
                for part in d.split("/"):
                    cur = cur + "/" + part
                    makedirs(cur)
                s.put(str(f), REMOTE + "/" + rel)
                print("OK", rel)
    finally:
        client.close()


if __name__ == "__main__":
    main()
