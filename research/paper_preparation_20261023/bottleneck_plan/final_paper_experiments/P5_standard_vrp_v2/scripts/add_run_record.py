# -*- coding: utf-8 -*-
"""把运行记录（各面板 start/end、端点、耗时）写入报告，并修正回执文件数。"""
import json
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
REP = V2 / "FINAL_REPORT_V2.md"
rows = []
panels = [("开发面板", V2 / "development_results/dev01_final/status.json"),
          ("正式 CVRP", V2 / "formal_results/cvrp/status.json"),
          ("正式 VRPTW", V2 / "formal_results/vrptw/status.json"),
          ("正式 STD_NORM", V2 / "formal_results/norm/status.json"),
          ("外部 PyVRP VRPTW", V2 / "external_results/pyvrp_vrptw/run1_status.json"),
          ("固定候选诊断", V2 / "diagnostics/dev01/status.json")]
for name, path in panels:
    if not path.exists():
        rows.append((name, "-", "-", "-", "-"))
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    start = (d.get("start") or "")[:19].replace("T", " ")
    end = (d.get("end") or "")[:19].replace("T", " ")
    rows.append((name, start, end, d.get("total_cells") or d.get("cells"), len(d.get("failed") or [])))

table = ["", "### 8.9 运行记录（UTC）", "",
         "| 面板 | 开始 | 结束 | 端点数 | 失败 |", "|---|---|---|---|---|"]
for r in rows:
    table.append("| %s | %s | %s | %s | %s |" % r)
table += ["",
          "> `/home` 分区剩余仅约 100 MB（全盘 1.7T 已满），故全部分面板输出写入 `/tmp`（tmpfs，124 GB）；",
          "> 结果为小 JSON，未对内存造成压力。运行期间机器为共享状态，lane 数按可用物理核自动分配并设上限",
          "> （正式 cvrp 16 / vrptw 12 / norm 8、开发 8、诊断 8）。", ""]

t = REP.read_text(encoding="utf-8")
if "### 8.9 运行记录" not in t:
    t = t.replace("## 9. 独立复算审计", "\n".join(table) + "\n## 9. 独立复算审计", 1)
t = t.replace("此外部署包 37 个文件 SHA-256 与本地审计过的代码逐一相同",
              "此外部署包 22 个文件（包 + 各面板执行器/phase）SHA-256 与本地审计过的代码逐一相同")
REP.write_text(t, encoding="utf-8")
print("run record + receipt count written")
