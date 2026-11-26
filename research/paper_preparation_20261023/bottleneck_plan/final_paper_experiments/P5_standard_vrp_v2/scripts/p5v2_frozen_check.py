# -*- coding: utf-8 -*-
"""冻结身份核验（P5-V2 增补）：逐文件 SHA-256 对照 manifests/frozen_identity.json。

用法：python p5v2_frozen_check.py [--update]
审计项：冷链锁版求解器、P5-V1 源、冻结 profit 源（含 checkpoints/f0_r0）均未被修改。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
MANIFEST = V2 / "manifests/frozen_identity.json"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def roots():
    base = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
    plan = ROOT / "research/paper_preparation_20261023/bottleneck_plan"
    return {
        "frozen_coldchain_source": plan / ("maskco_perishable_external_20261119/"
                                           "reserved_validation/solver"),
        "p5_v1_source": plan / "final_paper_experiments/P5_standard_vrp",
        "frozen_profit_source": base / "scripts/static_profit",
    }


def main(update=False):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    locate = roots()
    report = dict(groups={}, mismatched=[], missing=[])
    for group, files in man.items():
        if not isinstance(files, dict):
            continue
        root = locate.get(group)
        if root is None:
            continue
        entry = dict(root=str(root.relative_to(ROOT)).replace("\\", "/"),
                     files=len(files), ok=0, mismatch=0, missing=0)
        if update:
            for name in files:
                p = root / name
                if not p.exists():
                    report["missing"].append("%s/%s" % (group, name))
                    entry["missing"] += 1
                    continue
                files[name] = sha(p)
                entry["ok"] += 1
        else:
            for name, want in files.items():
                p = root / name
                if not p.exists():
                    report["missing"].append("%s/%s" % (group, name))
                    entry["missing"] += 1
                    continue
                got = sha(p)
                if got == want:
                    entry["ok"] += 1
                else:
                    entry["mismatch"] += 1
                    report["mismatched"].append(dict(group=group, file=name,
                                                     expected=want, actual=got))
        report["groups"][group] = entry
    if update:
        MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        report["updated"] = True
    total = sum(g["files"] for g in report["groups"].values())
    report["summary"] = dict(
        files=total,
        ok=sum(g["ok"] for g in report["groups"].values()),
        mismatch=len(report["mismatched"]),
        missing=len(report["missing"]))
    dest = V2 / "audit/frozen_identity_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for m in report["mismatched"]:
        print("MISMATCH", m["group"], m["file"])
    for m in report["missing"]:
        print("MISSING", m)
    return 1 if (report["mismatched"] or report["missing"]) else 0


if __name__ == "__main__":
    sys.exit(main("--update" in sys.argv))
