# -*- coding: utf-8 -*-
"""表12–16 生成器：P5-A CVRP 与 P5-B VRPTW 汇总表（MD/CSV/LaTeX + 导师版 Excel 合并）。

输入：P5_standard_vrp/analysis_p5a.json（全部阶段+外部方法）与 analysis_p5b.json。
输出：tables/表12–16.{md,csv,tex} 并合并进 论文实验汇总_导师版.xlsx。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
P5 = HERE.parent / "P5_standard_vrp"
METHOD_ORDER = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT",
                "ALNS", "VNS", "PyVRP", "NDS-Econ", "AttentionModel", "POMO",
                "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder", "RRNCO"]
FAMILY_ORDER = ["C1", "C2", "R1", "R2", "RC1", "RC2"]


def load(name):
    return json.loads((P5 / name).read_text(encoding="utf-8"))


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return ("%." + str(digits) + "f") % x


def build_12(a):
    """表12 P5-A：分规模段平均 Gap%（vs 官方 BKS，逐例等权，3 种子平均）。"""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in a["summary"])]
    header = ["方法", "100–199", "200–499", "500–1000", "总体", "参与实例数", "平均排名"]
    rows = []
    for m in methods:
        r = next(x for x in a["summary"] if x["method"] == m)
        rank = a["ranks"].get(m, {})
        rows.append([m, fmt(r.get("100-199")), fmt(r.get("200-499")), fmt(r.get("500-1000")),
                     fmt(r.get("overall")), str(r.get("overall_n", 0)), fmt(rank.get("avg_rank"))])
    return header, rows, "表12 P5-A 标准 CVRP：分规模段平均 Gap%（vs 官方 BKS，逐例等权、3 种子平均）"


def build_13(a):
    """表13 P5-A：逐实例排名统计（种子平均 D 排名）。"""
    methods = [m for m in METHOD_ORDER if m in a["ranks"]]
    header = ["方法", "平均排名", "第1名次数", "第2名次数", "第3名次数", "参与实例数"]
    rows = []
    counts = {}
    for k, v in a["per"].items():
        inst, m = k.split("|")
        if "rank" not in v:
            continue
        counts.setdefault(m, [0, 0, 0])
        if v["rank"] <= 1.5:
            counts[m][0] += 1
        elif v["rank"] <= 2.5:
            counts[m][1] += 1
        elif v["rank"] <= 3.5:
            counts[m][2] += 1
    for m in methods:
        r = a["ranks"][m]
        c = counts.get(m, [0, 0, 0])
        rows.append([m, fmt(r["avg_rank"]), str(c[0]), str(c[1]), str(c[2]), str(r["n"])])
    return header, rows, "表13 P5-A：逐实例排名统计（每实例按 3 种子平均 D 排名）"


def build_14(b):
    """表14 P5-B：分类平均（K,D）（3 种子平均）。"""
    methods = [m for m in METHOD_ORDER if m in b["methods"]]
    header = ["方法", "C1", "C2", "R1", "R2", "RC1", "RC2", "总体(56例)"]
    rows = []
    for m in methods:
        row = [m]
        for fam in FAMILY_ORDER:
            cells = [v for k, v in b["per"].items() if k.endswith("|" + m) and v["family"] == fam]
            if cells:
                row.append("%.1f / %.1f" % (sum(c["mean_k"] for c in cells) / len(cells),
                                            sum(c["mean_d"] for c in cells) / len(cells)))
            else:
                row.append("—")
        cells = [v for k, v in b["per"].items() if k.endswith("|" + m)]
        row.append("%.1f / %.1f" % (sum(c["mean_k"] for c in cells) / len(cells),
                                    sum(c["mean_d"] for c in cells) / len(cells)))
        rows.append(row)
    return header, rows, "表14 P5-B 标准 VRPTW：分类平均车辆数与平均距离（K / D，3 种子平均）"


def build_15(b):
    """表15 P5-B：字典序 (K,D) 两两胜负（种子配对、逐实例累计，行对列）。"""
    methods = [m for m in METHOD_ORDER if m in b["methods"]]
    header = ["方法"] + methods + ["总胜", "总负", "总平"]
    rows = []
    for a in methods:
        row = [a]
        w_all = l_all = t_all = 0
        for c in methods:
            if a == c:
                row.append("—")
                continue
            w, l, t = b["wins"].get(a, {}).get(c, [0, 0, 0])
            w_all += w
            l_all += l
            t_all += t
            row.append("%d/%d/%d" % (w, l, t))
        rows.append(row + [str(w_all), str(l_all), str(t_all)])
    return header, rows, "表15 P5-B：逐实例字典序 (K,D) 两两胜负（胜/负/平，种子配对累计；行方法对列方法）"


def build_16(b):
    """表16 数据泄漏审计标注：56 例按冷链角色分组（训练几何/验证几何/折外测试几何）。"""
    header = ["实例", "家族", "冷链角色", "注释"]
    rows = []
    for n in sorted(b["instances"]):
        if n in b["train"]:
            role, note = "训练几何(f0_r0)", "R/RC 38 例之一；结果不得称未见数据"
        elif n in b["validation"]:
            role, note = "验证几何(C2)", "f0_r0 训练未用；训练期验证"
        elif n in b["foldtest"]:
            role, note = "折外测试几何(C1)", "f0_r0 训练与验证未用"
        else:
            role, note = "RC205", "RC2 家族但不在 38 例训练集"
        rows.append([n, "R/RC/C 家族", role, note])
    return header, rows, "表16 P5-B 数据泄漏审计：Solomon 56 例与冷链训练/验证/折外几何重叠标注"


def write_table(idx, header, rows, title):
    stem = TABLES / ("表%d" % idx)
    lines = ["# " + title, "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    (stem.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tex = ["\\begin{table}[htbp]", "\\centering", "\\caption{" + title + "}",
           "\\label{tab:p5_%d}" % idx, "\\begin{tabular}{" + "l" * 1 + "c" * (len(header) - 1) + "}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(str(x).replace("%", "\\%") for x in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (stem.with_suffix(".tex")).write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("written", stem.with_suffix(".md").name)


def merge_excel():
    """把表12–16 合并进 论文实验汇总_导师版.xlsx（新增 P5 页签）。"""
    try:
        import openpyxl
    except ImportError:
        print("openpyxl missing, skip excel merge")
        return
    xlsx = TABLES / "论文实验汇总_导师版.xlsx"
    if not xlsx.exists():
        print("excel not found, skip merge")
        return
    wb = openpyxl.load_workbook(xlsx)
    for idx in (12, 13, 14, 15, 16):
        sheet = wb.create_sheet("表%d" % idx) if ("表%d" % idx) not in wb.sheetnames else wb["表%d" % idx]
        sheet.delete_rows(1, sheet.max_row)
        with open(TABLES / ("表%d.csv" % idx), encoding="utf-8-sig") as f:
            for line in f:
                sheet.append([c.strip() for c in line.strip().split(",")])
    wb.save(xlsx)
    print("excel merged:", xlsx.name)


def main():
    a = load("analysis_p5a.json")
    b = load("analysis_p5b.json")
    tables = [(12, build_12(a)), (13, build_13(a)), (14, build_14(b)), (15, build_15(b)), (16, build_16(b))]
    for idx, (header, rows, title) in tables:
        write_table(idx, header, rows, title)
    merge_excel()
    print("TABLES 12-16 DONE")


if __name__ == "__main__":
    main()
