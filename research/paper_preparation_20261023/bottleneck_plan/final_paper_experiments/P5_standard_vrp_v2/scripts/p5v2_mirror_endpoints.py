# -*- coding: utf-8 -*-
"""把端点 JSON 复制到 endpoints/ 子目录（交付用；out/ 被 git 排除策略挡住）。

保留 out/ 原样以便审计脚本可直接读取；endpoints/ 与 V1 的交付约定一致。
"""
import shutil
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
ROOTS = ["development_results", "formal_results", "external_results", "diagnostics"]
total = 0
for root in ROOTS:
    base = V2 / root
    if not base.exists():
        continue
    for out in sorted(base.rglob("out")):
        if not out.is_dir():
            continue
        dest = out.parent / "endpoints"
        dest.mkdir(exist_ok=True)
        for f in sorted(out.glob("*.json")):
            target = dest / f.name
            if not target.exists():
                shutil.copy2(f, target)
                total += 1
print("copied %d endpoint files" % total)
