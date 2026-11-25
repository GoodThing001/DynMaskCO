# -*- coding: utf-8 -*-
"""由 表3/表6b CSV 生成表3-6 的 Markdown/LaTeX 版本。"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "tables"

PANEL_GROUPS = {
    "表3_四数据集统一18法比较表": ["PDPCDPG-D100", "GH-p100", "XTW-p100(21)", "VRPPDL-p100"],
    "表4_既有三来源复测表": ["PDPCDPG-D100", "XTW-p100(既有13)", "GH-p100"],
    "表5_三前瞻来源表": ["VRPPDL-p100", "XTW-p100(新8)", "RWVRP-p100"],
    "表6_GH压力与原版对比表": ["GH-p100-block2"],
}


def load_table3():
    rows = list(csv.DictReader((TAB / "表3_四数据集统一18法比较表.csv").open(encoding="utf-8-sig")))
    return rows


def main():
    rows = load_table3()
    if not rows:
        print("表3 CSV not available")
        return
    methods = [r["method"] for r in rows if r["panel"] == rows[0]["panel"]]
    meta = {r["method"]: (r["category"], r["venue"], r["year"], r["grade"]) for r in rows if r["panel"] == rows[0]["panel"]}
    for table, panels in PANEL_GROUPS.items():
        by_panel = {p: {r["method"]: r for r in rows if r["panel"] == p} for p in panels}
        md = [f"# {table.replace('_', ' ')}", "",
              "单元格为平均净效益 U（名次/18；内部参照亦列入，名次按完整 18 法名单）。"
              "聚合：案例内种子均值→几何内等权→来源内几何等权；并列取末位。", ""]
        header = ["方法", "类别", "发表载体", "年份", "等级"] + [f"{p} U (r/18)" for p in panels]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        tex_rows = []
        for m in methods:
            cat, venue, year, grade = meta[m]
            cells = []
            for p in panels:
                r = by_panel[p].get(m)
                cells.append(f"{float(r['mean_U']):,.2f}（{r['rank18']}）" if r else "—")
            md.append(f"| {m} | {cat} | {venue} | {year} | {grade} | " + " | ".join(cells) + " |")
            tex_rows.append([m, cat, venue, str(year), grade] + [c.replace("（", " (").replace("）", ")") for c in cells])
        (TAB / (table + ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
        # LaTeX
        tex = [r"\begin{table}[t]", r"\centering", r"\caption{" + table + "}",
               r"\label{tab:" + table + "}",
               r"\begin{tabular}{lllll" + "r" * len(panels) + "}", r"\toprule",
               " & ".join(header) + r" \\", r"\midrule"]
        for r in tex_rows:
            tex.append(" & ".join(r) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TAB / (table + ".tex")).write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(table, "written")
    # 表6b 与 表10
    inc_path = TAB / "表6b_相对MASK与SLOT配对增量表.csv"
    if inc_path.is_file():
        inc = list(csv.DictReader(inc_path.open(encoding="utf-8-sig")))
        md = ["# 表6b 完整方法相对 MASK、SLOT 与同规则距离版的配对增量", "",
              "ΔU 为 ORDER_SPLIT − 对照（正值为完整方法更优）；逐案例逐种子计算后按登记规则聚合；"
              "区间为几何整簇重采样 20,000 次（种子 20261120）。经济损失降低率逐实例计算后平均，分母过小已剔除并计数。", "",
              "| 面板 | 对照 | ΔU | 95% 区间 | 胜/平/负几何 | 改善实例 | 改善几何 | 经济损失降低率% | 逐种子 |",
              "|---|---|---:|---|---:|---:|---:|---:|---|"]
        for r in inc:
            lr = f"{float(r['loss_reduction_pct']):+.4f}%" if r["loss_reduction_pct"] else "不可用"
            if int(r["loss_reduction_unusable"]) > 0:
                lr += f"（{r['loss_reduction_unusable']} 例分母过小）"
            md.append(f"| {r['panel']} | {r['comparator']} | {float(r['delta_U']):+,.4f} | "
                      f"[{float(r['ci95_lo']):+,.4f}, {float(r['ci95_hi']):+,.4f}] | "
                      f"{r['wins']}/{r['ties']}/{r['losses']} | {r['improved_instances']}/{r['total_instances']} | "
                      f"{r['improved_geometries']}/{r['total_geometries']} | {lr} | {r['by_seed']} |")
        (TAB / "表6b_相对MASK与SLOT配对增量表.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        md10 = ["# 表10 品质损失、里程费用、制冷费用及经济损失降低率", "",
                "三项分解为对照成本 − 本方法成本（正值对本方法有利），三项之和等于 ΔU；"
                "经济损失降低率 = (L对照 − L本方法)/L对照 ×100%，逐实例计算后平均。", "",
                "| 面板 | 对照 | 品质改善 U | 里程节省 U | 制冷节省 U | ΔU | 经济损失降低率% |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for r in inc:
            lr = f"{float(r['loss_reduction_pct']):+.4f}%" if r["loss_reduction_pct"] else "不可用"
            md10.append(f"| {r['panel']} | {r['comparator']} | {float(r['quality']):+,.4f} | "
                        f"{float(r['distance']):+,.4f} | {float(r['refrigeration']):+,.4f} | "
                        f"{float(r['delta_U']):+,.4f} | {lr} |")
        (TAB / "表10_经济分解与损失降低率.md").write_text("\n".join(md10) + "\n", encoding="utf-8")
        print("表6b / 表10 written")


if __name__ == "__main__":
    main()
