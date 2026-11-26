# -*- coding: utf-8 -*-
"""为归档目录生成 SHA256SUMS.txt（自校验用，排除自身）。"""
import hashlib
from pathlib import Path

OUT = Path(r"D:\PyCharm_\MASKCO-Main\archive\p5_frozen_20261126")


def sha(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS.txt":
            continue
        rows.append("%s  %s" % (sha(p), p.relative_to(OUT).as_posix()))
    (OUT / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("wrote %d entries" % len(rows))
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()
