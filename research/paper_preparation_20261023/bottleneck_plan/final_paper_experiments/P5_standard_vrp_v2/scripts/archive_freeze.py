# -*- coding: utf-8 -*-
"""P5 / P5-V2 冻结归档：逐文件 SHA256 清单 + tar.gz 封存 + 回执。

用法：python archive_freeze.py [--verify-only]
输出（仓库外，避免污染 git 与体积守卫）：
  D:\\PyCharm_\\MASKCO-Main\\archive\\p5_frozen_20261126\\
    MANIFEST.json      逐文件 SHA256（相对路径/字节数/摘要）+ 汇总
    INDEX.csv          可读索引
    ARCHIVE.md         归档说明（冻结范围、清点、校验、复现、限制）
    ARCHIVE_RECEIPT.json  归档包 SHA256 与自校验结果
    P5_P5V2_FROZEN_20261126.tar.gz
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
TREES = ["research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp",
         "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp_v2"]
OUT = ROOT / "archive/p5_frozen_20261126"
STAMP = "20261126"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {"P5_P5V2_FROZEN_20261126.tar.gz"}


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.stdout.strip(), proc.returncode


def walk():
    rows = []
    for tree in TREES:
        base = ROOT / tree
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.name in SKIP_FILES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            rows.append(dict(path=rel, bytes=path.stat().st_size, sha256=sha(path)))
    return rows


def main():
    verify_only = "--verify-only" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    head, _ = git("rev-parse", "HEAD")
    branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    remote, _ = git("ls-remote", "origin", "refs/heads/" + branch)
    remote_sha = remote.split()[0] if remote else None
    status, _ = git("status", "--porcelain", *TREES)
    dirty = [line for line in status.splitlines()
             if not any(k in line for k in ("/logs/", "/out/"))]

    if verify_only:
        man = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
        bad = 0
        for row in man["files"]:
            p = ROOT / row["path"]
            if not p.exists() or sha(p) != row["sha256"] or p.stat().st_size != row["bytes"]:
                bad += 1
                print("MISMATCH", row["path"])
        print(json.dumps(dict(files=len(man["files"]), mismatch=bad), ensure_ascii=False))
        return 0 if bad == 0 else 1

    rows = walk()
    per_tree = {}
    for row in rows:
        key = next(t for t in TREES if row["path"].startswith(t))
        agg = per_tree.setdefault(key, dict(files=0, bytes=0))
        agg["files"] += 1
        agg["bytes"] += row["bytes"]
    manifest = dict(
        stamp=STAMP, created_utc=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        git=dict(head=head, branch=branch, origin_sha=remote_sha,
                 tree_dirty_excluding_logs_and_out=dirty),
        algorithm="sha256", files=rows,
        totals=dict(files=len(rows), bytes=sum(r["bytes"] for r in rows),
                    per_tree=per_tree))
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    with (OUT / "INDEX.csv").open("w", newline="", encoding="utf-8-sig") as f:
        f.write("path,bytes,sha256\n")
        for row in rows:
            f.write("%s,%d,%s\n" % (row["path"], row["bytes"], row["sha256"]))
    print("manifest: %d files, %.2f MB" % (len(rows), manifest["totals"]["bytes"] / 1e6))

    archive = OUT / ("P5_P5V2_FROZEN_%s.tar.gz" % STAMP)
    if archive.exists():
        archive.unlink()
    files_from = OUT / "_files_from.txt"
    files_from.write_text("\n".join(TREES) + "\n", encoding="utf-8")
    proc = subprocess.run(["tar", "-czf", str(archive), "-C", str(ROOT), "-T", str(files_from)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    files_from.unlink()
    if proc.returncode != 0:
        print("tar failed:", proc.stderr[:500])
        return 1
    digest = sha(archive)
    receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=digest,
                   members=["%s/**" % t for t in TREES],
                   manifest_sha256=sha(OUT / "MANIFEST.json"),
                   verified_files=len(rows),
                   note="解包后可用 MANIFEST.json 逐文件校验 SHA256")
    (OUT / "ARCHIVE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print("archive: %s  %.2f MB  sha256=%s" % (archive.name, receipt["bytes"] / 1e6, digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
