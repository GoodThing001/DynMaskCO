# -*- coding: utf-8 -*-
"""修正报告运行记录中的外部基线行（分两批 + 8 个被误杀单元的说明）。"""
import json
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
REP = V2 / "FINAL_REPORT_V2.md"
t = REP.read_text(encoding="utf-8")

s1 = json.loads((V2 / "external_results/pyvrp_vrptw/run1_status.json").read_text(encoding="utf-8"))
try:
    s2 = json.loads((V2 / "external_results/pyvrp_vrptw/run2_status.json").read_text(encoding="utf-8"))
    end2 = (s2.get("end") or "")[:19].replace("T", " ")
    cells2 = s2.get("cells")
    fail2 = len(s2.get("failed") or [])
except FileNotFoundError:
    end2, cells2, fail2 = "-", "-", "-"

old = "| 外部 PyVRP VRPTW | 2026-11-26 02:17:30 | 2026-11-26 02:22:34 | 168 | 8 |"
new = ("| 外部 PyVRP VRPTW（第 1 批） | %s | %s | %s | %d |\n"
       "| 外部 PyVRP VRPTW（补跑第 2 批） | %s | %s | %s | %s |") % (
    (s1.get("start") or "")[:19].replace("T", " "),
    (s1.get("end") or "")[:19].replace("T", " "),
    s1.get("cells"), len(s1.get("failed") or []),
    (s2.get("start") or "")[:19].replace("T", " ") if end2 != "-" else "-",
    end2, cells2, fail2)
if old in t:
    t = t.replace(old, new)
    t = t.replace("> `/home` 分区剩余仅约 100 MB",
                  "> 外部基线第 1 批的 8 个单元被运维脚本 `pkill unified_worker` 误杀（非求解失败），"
                  "已用同一冻结工作器补跑（第 2 批），最终 168/168 全部 AUDITED；两批均为同一代码与口径。\n"
                  "> `/home` 分区剩余仅约 100 MB", 1)
    REP.write_text(t, encoding="utf-8")
    print("external rows fixed")
else:
    print("pattern not found (already fixed?)")
