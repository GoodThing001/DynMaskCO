# -*- coding: utf-8 -*-
"""通用结果抓取：远端目录 out/*.json → 本地目录。

用法：python p5v2_fetch_dir.py <remote_dir> <local_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport


def main(remote_dir, local_dir):
    dest = Path(local_dir) if not str(local_dir).startswith("/") else V2 / str(local_dir)
    (dest / "out").mkdir(parents=True, exist_ok=True)
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    got = 0
    try:
        names = transport.remote_exec(client, "ls " + remote_dir + "/out 2>/dev/null || true")
        names = [n.strip() for n in str(names).splitlines() if n.strip()]
        with client.open_sftp() as s:
            for name in names:
                if not name.endswith(".json"):
                    continue
                try:
                    s.get(remote_dir + "/out/" + name, str(dest / "out" / name))
                    got += 1
                except OSError as exc:
                    print("skip", name, exc)
            for extra in ("phase.json", "status.json", "runner.log"):
                try:
                    s.get(remote_dir + "/" + extra, str(dest / extra))
                except OSError:
                    pass
        print("downloaded", got, "->", dest)
    finally:
        client.close()
    return got


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
