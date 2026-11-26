# -*- coding: utf-8 -*-
"""把 V2 README 的状态行改为"已冻结归档"，并指向 ARCHIVE_NOTICE。"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "README.md"
t = p.read_text(encoding="utf-8")
old = ("> **状态：完成（2026-11-26）。** 起始 commit `736414664773d83ef55007a541b8bfb6ca3aaae0`\n"
       "> （分支 `codex/o0cc-cal-phys`）。P0—P4 冷链实验、锁版机制、`f0_r0` 检查点与 P5-V1 全部冻结，\n"
       "> 本目录未修改其中任何文件。")
new = ("> **状态：已完成并冻结归档（2026-11-26，实验终止迭代）。** 起始 commit\n"
       "> `736414664773d83ef55007a541b8bfb6ca3aaae0`，交付 commit `350e18f0`（分支 `codex/o0cc-cal-phys`）。\n"
       "> 归档包与逐文件 SHA256 清单见 `ARCHIVE_NOTICE.md`。P0—P4 冷链实验、锁版机制、`f0_r0` 检查点\n"
       "> 与 P5-V1 全部冻结，本目录未修改其中任何文件。")
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("README status updated")
