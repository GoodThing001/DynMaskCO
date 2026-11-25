# -*- coding: utf-8 -*-
"""论文表格生成器：表1 数据集统计、表2 方法与发表信息、表3/4/5/6 比较与增量表。

从 P0 审计与 analysis_18.py 输出 CSV 生成 Markdown/CSV/LaTeX 三格式。
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent
UNI = FP.parent / "unified_external_20261120"
TAB = FP / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def write_csv(name, rows):
    with (TAB / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def latex_table(name, caption, header, rows, align):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}",
             r"\label{tab:" + name + "}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table1():
    # 训练、验证、复测、前瞻及压力测试数据集统计
    rows = [
        dict(dataset="VRPTWQ-Solomon R/RC（训练）", cases=38, geometries=38, seeds="2000+100×fold+10×repeat（训练）",
             methods="—", endpoints=0, role="训练（f0_r0 1440 步）"),
        dict(dataset="PDPCDPG-D100（既有复测）", cases=5, geometries=5, seeds="33001–33003", methods=16, endpoints=240, role="既有面板复测"),
        dict(dataset="XTW-p100（既有复测）", cases=13, geometries=13, seeds="33001–33003", methods=16, endpoints=624, role="既有面板复测"),
        dict(dataset="GH-p100（既有复测）", cases=12, geometries=5, seeds="33001–33003", methods=16, endpoints=576, role="既有面板复测"),
        dict(dataset="VRPPDL-p100（前瞻）", cases=10, geometries=10, seeds="65001–65003", methods=16, endpoints=480, role="前瞻主评价"),
        dict(dataset="新XTW-p100（前瞻）", cases=8, geometries=8, seeds="65001–65003", methods=16, endpoints=384, role="前瞻主评价"),
        dict(dataset="RWVRP-p100（前瞻）", cases=8, geometries=8, seeds="65001–65003", methods=16, endpoints=384, role="前瞻主评价"),
        dict(dataset="GH-p100-block2（压力）", cases=6, geometries=5, seeds="65001–65003", methods=16, endpoints=288, role="同母几何跨客户块压力测试"),
        dict(dataset="P1 MASK/SLOT 补充", cases=62, geometries=54, seeds="33001–33003 / 65001–65003", methods=2, endpoints=372, role="原版统一对比（新增）"),
        dict(dataset="P2 八城十种子", cases=8, geometries=8, seeds="65001–65010", methods=3, endpoints=240, role="八城稳定性与轨迹（新增）"),
        dict(dataset="P3 双因素消融", cases=26, geometries=26, seeds="65001–65003", methods=2, endpoints=156, role="HYBRID/HYBRID_D（新增）"),
        dict(dataset="合计（主实验）", cases=62, geometries=54, seeds="6", methods="16+2+2", endpoints="2,976+372+240+156", role="P0–P3"),
    ]
    write_csv("表1_数据集统计.csv", rows)
    md = ["# 表1 训练、验证、复测、前瞻及压力测试数据集统计", "",
          "| 数据集 | 案例数 | 独立客户几何数 | 种子 | 方法数 | 终点数 | 角色 |",
          "|---|---:|---:|---|---:|---:|---|"]
    for r in rows:
        md.append("| {} | {} | {} | {} | {} | {} | {} |".format(*r.values()))
    (TAB / "表1_数据集统计.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latex_table("tab_datasets", "训练、验证、复测、前瞻及压力测试数据集统计",
                ["数据集", "案例", "几何", "种子", "方法", "终点", "角色"],
                [[str(r[k]) for k in ("dataset", "cases", "geometries", "seeds", "methods", "endpoints", "role")] for r in rows],
                "lrrrrrr")
    print("table1 written")


def table2():
    from analysis_18 import PUB
    rows = []
    for i, (m, (cat, venue, year, grade)) in enumerate(PUB.items()):
        rows.append(dict(seq=i + 1, method=m, category=cat, venue=venue, year=year, grade=grade))
    write_csv("表2_对比方法及发表信息.csv", rows)
    md = ["# 表2 所有对比方法及论文发表信息", "",
          "| 序号 | 方法 | 类别 | 发表期刊/会议 | 年份 | CCF等级或中科院分区 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['seq']} | {r['method']} | {r['category']} | {r['venue']} | {r['year']} | {r['grade']} |")
    md += ["", "注：会议等级统一采用 CCF 2026 第七版推荐目录；期刊采用中科院 2025 升级版大类分区；"
             "带 * 的期刊为公开逐刊资料一致报告、官方逐刊原表未直核。本项目自行实现的方法不继承基础 MaskCO 的 ICLR 等级；"
             "TMLR 分区未核实。ALNS 的 2006 论文是算法思想来源，通用软件实现另引 Wouda & Lan, JOSS 2023；"
             "PyVRP 的 2024 为软件论文年份。"]
    (TAB / "表2_对比方法及发表信息.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latex_table("tab_methods", "对比方法及论文发表信息",
                ["序号", "方法", "类别", "发表期刊/会议", "年份", "CCF/中科院等级"],
                [[str(r[k]) for k in ("seq", "method", "category", "venue", "year", "grade")] for r in rows],
                "rlllrl")
    print("table2 written")


def main():
    table1()
    table2()


if __name__ == "__main__":
    main()
