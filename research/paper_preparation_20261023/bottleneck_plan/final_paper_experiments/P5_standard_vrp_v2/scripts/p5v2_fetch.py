# -*- coding: utf-8 -*-
"""下载远程 P5-V2 端点结果（逐文件 SFTP，避免 tar 在满盘失败）。

用法：python p5v2_fetch.py [rundir] [localname]
默认 rundir=dev01，落地到 development_results/<localname>/{out/*.json, phase.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"


def main(run="dev01", local=None, prefix="out/"):
    local = local or run
    dest = V2 / "development_results" / local
    (dest / "out").mkdir(parents=True, exist_ok=True)
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    got = 0
    try:
        listing = transport.remote_exec(
            client, "ls " + P5V2 + "/" + run + "/out 2>/dev/null | grep '\\.json$' || true")
        names = [n.strip() for n in str(listing).splitlines() if n.strip().endswith(".json")]
        with client.open_sftp() as s:
            for name in names:
                try:
                    s.get(P5V2 + "/" + run + "/" + prefix + name, str(dest / "out" / name))
                    got += 1
                except OSError as exc:
                    print("skip", name, exc)
            for extra in ("phase.json", "status.json"):
                try:
                    s.get(P5V2 + "/" + run + "/" + extra, str(dest / extra))
                except OSError:
                    pass
        print("downloaded", got, "->", dest)
    finally:
        client.close()
    return got


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
