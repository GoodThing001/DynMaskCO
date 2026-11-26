# -*- coding: utf-8 -*-
"""把 tables/*.csv 转成 MD / LaTeX / Excel（单一工作簿多 sheet）。

输出：tables/md/*.md、tables/latex/*.tex、tables/P5V2_tables.xlsx
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook

V2 = Path(__file__).resolve().parent.parent
TAB = V2 / "tables"


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def esc_tex(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")
            .replace("{", "\\{").replace("}", "\\}"))


def to_md(rows, name):
    if not rows:
        return ""
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    for r in rows[1:]:
        r = (r + [""] * width)[:width]
        lines.append("| " + " | ".join(x.replace("|", "\\|") for x in r) + " |")
    return "\n".join(lines) + "\n"


def to_tex(rows, name):
    if not rows:
        return ""
    width = len(rows[0])
    lines = [r"\begin{tabular}{" + "l" + "r" * (width - 1) + "}", r"\toprule",
             " & ".join(esc_tex(h) for h in rows[0]) + r" \\", r"\midrule"]
    for r in rows[1:]:
        r = (r + [""] * width)[:width]
        lines.append(" & ".join(esc_tex(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    (TAB / "md").mkdir(exist_ok=True)
    (TAB / "latex").mkdir(exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    names = sorted(p.stem for p in TAB.glob("V2*.csv"))
    for stem in names:
        rows = read_csv(TAB / (stem + ".csv"))
        (TAB / "md" / (stem + ".md")).write_text(to_md(rows, stem), encoding="utf-8")
        (TAB / "latex" / (stem + ".tex")).write_text(to_tex(rows, stem), encoding="utf-8")
        ws = wb.create_sheet(title=stem[:31])
        for r in rows:
            ws.append(r)
        for col in ws.columns:
            width = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(40, max(8, width + 2))
    wb.save(TAB / "P5V2_tables.xlsx")
    print("converted %d tables -> md/latex/xlsx" % len(names))


if __name__ == "__main__":
    main()
