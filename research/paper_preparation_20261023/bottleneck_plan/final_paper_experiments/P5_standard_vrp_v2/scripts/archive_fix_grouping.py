# -*- coding: utf-8 -*-
"""修正归档清单的按树归属（P5_standard_vrp 是 P5_standard_vrp_v2 的前缀），并重建清单与回执。"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
OUT = ROOT / "archive/p5_frozen_20261126"
TREES = ["research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp",
         "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp_v2"]


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_of(path: str) -> str:
    hits = [t for t in TREES if path == t or path.startswith(t + "/")]
    if not hits:
        raise KeyError(path)
    return max(hits, key=len)   # 最长前缀 = 精确归属


def main():
    man = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    per = {}
    for row in man["files"]:
        key = tree_of(row["path"])
        agg = per.setdefault(key, dict(files=0, bytes=0))
        agg["files"] += 1
        agg["bytes"] += row["bytes"]
    man["totals"]["per_tree"] = per
    man["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    man["path_grouping_note"] = "按最长路径前缀归属（P5_standard_vrp 与 P5_standard_vrp_v2 为并列的两棵树）"
    (OUT / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    rec = json.loads((OUT / "ARCHIVE_RECEIPT.json").read_text(encoding="utf-8"))
    rec["manifest_sha256"] = sha(OUT / "MANIFEST.json")
    rec["per_tree"] = per
    (OUT / "ARCHIVE_RECEIPT.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps(dict(total=man["totals"]["files"],
                          per_tree={k.split("/")[-1]: v for k, v in per.items()},
                          manifest_sha256=rec["manifest_sha256"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
