# -*- coding: utf-8 -*-
"""部署包回执（最终版）：逐文件 SHA-256 对照本地审计代码与远端实际运行副本。

覆盖：包（standard_vrp/**）、开发面板执行器/phase、正式三段执行器/phase、诊断执行器。
输出 manifests/package_receipt.json。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
PAIRS = [
    ({"root": V2 / "standard_vrp", "remote": P5V2, "glob": "**/*"},
     "package"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect():
    items = []
    for f in sorted((V2 / "standard_vrp").rglob("*")):
        if f.is_file() and f.suffix in (".py", ".json"):
            items.append((f, P5V2 + "/" + f.relative_to(V2).as_posix()))
    items.append((HERE / "p5v2_worker.py", "/tmp/p5v2_dev/dev01/unified_worker.py"))
    items.append((V2 / "configs/phase_dev.json", "/tmp/p5v2_dev/dev01/phase.json"))
    for stage, phase in (("cvrp", "phase_formal_cvrp.json"), ("vrptw", "phase_formal_vrptw.json"),
                         ("norm", "phase_formal_norm.json")):
        items.append((HERE / "p5v2_worker.py", "/tmp/p5v2_formal/%s/unified_worker.py" % stage))
        items.append((V2 / "configs" / phase, "/tmp/p5v2_formal/%s/phase.json" % stage))
    items.append((HERE / "p5v2_diag_worker.py", P5V2 + "/diag01/unified_worker.py"))
    items.append((HERE / "p5v2_fixed_candidates.py", P5V2 + "/diag01/p5v2_fixed_candidates.py"))
    return items


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    rows = []
    try:
        with client.open_sftp() as s:
            for local, remote in collect():
                try:
                    with s.open(remote, "rb") as f:
                        data = f.read()
                    rsha = sha(data)
                except OSError:
                    rsha = None
                lsha = sha(local.read_bytes())
                rows.append(dict(local=local.relative_to(V2).as_posix(), remote=remote,
                                 local_sha=lsha, remote_sha=rsha, match=(lsha == rsha)))
    finally:
        client.close()
    ok = sum(1 for r in rows if r["match"])
    receipt = dict(files=len(rows), matched=ok,
                   mismatched=[r["remote"] for r in rows if not r["match"]], rows=rows)
    out = V2 / "manifests/package_receipt.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(files=len(rows), matched=ok, mismatched=len(receipt["mismatched"]))))
    for r in receipt["mismatched"][:5]:
        print("MISMATCH", r)
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
