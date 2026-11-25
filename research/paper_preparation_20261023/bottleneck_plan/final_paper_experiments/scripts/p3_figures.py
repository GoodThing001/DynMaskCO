# -*- coding: utf-8 -*-
"""P3 双因素消融四图（科研论文风格）。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P3 = FP / "P3_factorial_ablation"
FIG = FP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from paper_style import METHOD_COLORS, style_ax, save  # noqa: E402


def load_summaries():
    rows = list(csv.DictReader((P3 / "P3_双因素消融汇总表.csv").open(encoding="utf-8-sig")))
    return rows


def errs(r):
    return (float(r["mean"]) - float(r["ci95_lo"]), float(r["ci95_hi"]) - float(r["mean"]))


def fig1_factorial(rows):
    import matplotlib.pyplot as plt
    sources = ["VRPPDL-p100", "XTW-p100(新8)", "RWVRP-p100"]
    arms = [("HYBRID_D", "D"), ("HYBRID", "A"), ("ORDER_SPLIT_D", "B"), ("ORDER_SPLIT", "C")]
    # 直接使用端点均值（U），从逐案例 CSV 算
    per = list(csv.DictReader((P3 / "P3_逐案例配对结果.csv").open(encoding="utf-8-sig")))
    means = {}
    for src in sources:
        sub = [r for r in per if r["source"] == src]
        means[src] = {k: float(np.mean([float(r[k]) for r in sub])) for k in
                      ("HYBRID_D", "HYBRID", "ORDER_SPLIT_D", "ORDER_SPLIT")}
    # 以 HYBRID_D 为基准画相对值
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    offsets = {a: i for i, (a, _) in enumerate(arms)}
    x = np.arange(len(sources))
    w = 0.18
    for arm, label in arms:
        base = np.array([means[s]["HYBRID_D"] for s in sources])
        vals = np.array([means[s][arm] for s in sources]) - base
        ax.bar(x + (offsets[arm] - 1.5) * w, vals, width=w, label=arm,
               color=[METHOD_COLORS["HYBRID_D"], METHOD_COLORS["HYBRID"],
                      METHOD_COLORS["ORDER_SPLIT_D"], METHOD_COLORS["ORDER_SPLIT"]][offsets[arm]],
               edgecolor="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["VRPPDL-p100", "新XTW-p100", "RWVRP-p100"])
    style_ax(ax, ylabel="相对 HYBRID_D 的净效益增量 (U)", title="双因素消融：四臂相对增量")
    ax.legend(ncol=2, frameon=False)
    save(fig, FIG / "P3_图1_双因素消融对比")


def fig_contrib(rows, fig_name, quantities, title):
    import matplotlib.pyplot as plt
    sources = ["VRPPDL-p100", "XTW-p100(新8)", "RWVRP-p100"]
    lookup = {(r["source"], r["quantity"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(quantities))
    w = 0.26
    colors = {"VRPPDL-p100": "#1f4e79", "XTW-p100(新8)": "#c00000", "RWVRP-p100": "#548235"}
    for j, src in enumerate(sources):
        vals = [float(lookup[(src, q)]["mean"]) for q in quantities]
        lo = [float(lookup[(src, q)]["mean"]) - float(lookup[(src, q)]["ci95_lo"]) for q in quantities]
        hi = [float(lookup[(src, q)]["ci95_hi"]) - float(lookup[(src, q)]["mean"]) for q in quantities]
        ax.bar(x + (j - 1) * w, vals, width=w, yerr=[lo, hi], capsize=2.5, label=src,
               color=colors[src], edgecolor="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["A 无分车学习增益", "B 有分车学习增益"] if len(quantities) == 2 else
                       ["C 学习下分车增益", "D 纯距离分车增益"])
    style_ax(ax, ylabel="配对 ΔU", title=title)
    ax.legend(frameon=False, ncol=3)
    save(fig, FIG / fig_name)


def fig4_interaction(rows):
    import matplotlib.pyplot as plt
    sources = ["VRPPDL-p100", "XTW-p100(新8)", "RWVRP-p100"]
    lookup = {(r["source"], r["quantity"]): r for r in rows}
    eq = list(csv.DictReader((P3 / "P3_三来源等权汇总.csv").open(encoding="utf-8-sig")))
    eqm = {r["quantity"]: float(r["equal_source_mean"]) for r in eq}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    vals = [float(lookup[(s, "interaction_B_minus_A")]["mean"]) for s in sources]
    lo = [float(lookup[(s, "interaction_B_minus_A")]["mean"]) - float(lookup[(s, "interaction_B_minus_A")]["ci95_lo"]) for s in sources]
    hi = [float(lookup[(s, "interaction_B_minus_A")]["ci95_hi"]) - float(lookup[(s, "interaction_B_minus_A")]["mean"]) for s in sources]
    x = np.arange(len(sources) + 1)
    ax.bar(x[:-1], vals, yerr=[lo, hi], capsize=3, color="#7030a0", edgecolor="white", zorder=3, label="各来源交互 B−A")
    ax.bar(x[-1], eqm["interaction_B_minus_A"], color="#1f4e79", edgecolor="white", zorder=3, label="三来源等权")
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(sources + ["三来源等权"])
    style_ax(ax, ylabel="交互 ΔU (B−A = C−D)", title="学习引导与经济分车的交互效应")
    ax.legend(frameon=False)
    save(fig, FIG / "P3_图4_交互效应")


def main():
    rows = load_summaries()
    if not rows:
        print("P3 summaries not available")
        return
    fig1_factorial(rows)
    fig_contrib(rows, "P3_图2_学习引导贡献对比", ("A_learning_without_split", "B_learning_with_split"),
                "学习引导贡献：无分车 (A) 与有分车 (B)")
    fig_contrib(rows, "P3_图3_跨车分车贡献对比", ("C_split_with_learning", "D_split_without_learning"),
                "跨车经济分车贡献：学习引导下 (C) 与纯距离下 (D)")
    fig4_interaction(rows)
    print("P3 figures written")


if __name__ == "__main__":
    main()
