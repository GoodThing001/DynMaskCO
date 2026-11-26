# -*- coding: utf-8 -*-
"""归档自校验：解包 tar.gz 到临时目录，按 MANIFEST.json 逐文件比对 SHA256。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
OUT = ROOT / "archive/p5_frozen_20261126"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "ARCHIVE_RECEIPT.json").read_text(encoding="utf-8"))
    archive = OUT / receipt["archive"]
    got = sha(archive)
    if got != receipt["sha256"]:
        print("ARCHIVE SHA MISMATCH", got, receipt["sha256"])
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(["tar", "-xzf", str(archive), "-C", tmp], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print("extract failed:", proc.stderr[:400])
            return 1
        base = Path(tmp)
        missing, bad = 0, 0
        for row in man["files"]:
            p = base / row["path"]
            if not p.exists():
                missing += 1
                continue
            if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
                bad += 1
                if bad <= 5:
                    print("MISMATCH", row["path"])
        result = dict(files=len(man["files"]), missing=missing, mismatch=bad,
                      archive_sha256=got, archive_bytes=archive.stat().st_size)
        print(json.dumps(result, ensure_ascii=False))
        (OUT / "VERIFY_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
        return 0 if (missing == 0 and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
