# -*- coding: utf-8 -*-
"""P4 科研图：Solomon C1/C2 18 法结果、配对增量与轨迹（论文风格）。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P4 = FP / "P4_solomon_coldchain"
TAB = FP / "tables"
FIG = FP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from paper_style import METHOD_COLORS, style_ax, save  # noqa: E402

C1 = ["c105", "c106", "c107", "c108", "c109"]
C2 = ["c205", "c206", "c207", "c208"]
SEEDS = [67001, 67002, 67003]
OUR = "ORDER_SPLIT"


def load():
    data = {}
    for f in sorted((P4 / "results" / "out").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        data[(d["case"], d["seed"], d["method"])] = d
    return data


def fig1_18methods(data):
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((TAB / "表11_Solomon冷链18法.csv").open(encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, panel in zip(axes, ("C1", "C2")):
        sub = [r for r in rows if r["panel"] == panel]
        names = [r["method"] for r in sub]
        vals = [float(r["mean_U"]) for r in sub]
        # 以完整方法为基准的相对值
        base = next(float(r["mean_U"]) for r in sub if r["method"] == OUR)
        rel = [v - base for v in vals]
        colors = [METHOD_COLORS.get(m, "#888888") for m in names]
        order = np.argsort(rel)
        ax.barh(range(len(names)), [rel[i] for i in order], color=[colors[i] for i in order],
                edgecolor="white", zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
        ax.axvline(0, color="black", linewidth=0.8, zorder=2)
        style_ax(ax, xlabel="相对 ORDER_SPLIT 的净效益 (U)", title=panel + "（" +
                 ("折外" if panel == "C1" else "验证") + "，等权）")
    fig.tight_layout()
    save(fig, FIG / "P4_图1_Solomon18法对比")


def fig2_paired(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    panels = {"C1": C1, "C2": C2, "合并9例": C1 + C2}
    comps = ("MASK", "SLOT", "ALNS", "ORDER_SPLIT_D")
    x = np.arange(len(panels))
    w = 0.2
    for j, comp in enumerate(comps):
        vals, lo, hi = [], [], []
        for cases in panels.values():
            d = [data[c, s, OUR]["U"] - data[c, s, comp]["U"] for c in cases for s in SEEDS]
            vals.append(np.mean(d))
            lo.append(np.mean(d) - np.percentile(d, 2.5))
            hi.append(np.percentile(d, 97.5) - np.mean(d))
        ax.bar(x + (j - 1.5) * w, vals, width=w, yerr=[lo, hi], capsize=2.5,
               color=METHOD_COLORS.get(comp, "#888888"), edgecolor="white", zorder=3, label="对 " + comp)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(list(panels.keys()))
    style_ax(ax, ylabel="ORDER_SPLIT 配对 ΔU（每种子）", title="对原版与消融的配对增量")
    ax.legend(frameon=False, ncol=2)
    save(fig, FIG / "P4_图2_配对增量")


def fig3_trajectory(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ts = [5, 15, 30, 60]
    for panel, cases in (("C1", C1), ("C2", C2)):
        for comp in ("ALNS", "ORDER_SPLIT_D"):
            ys = []
            for t in ts:
                vals = []
                for c in cases:
                    for s in SEEDS:
                        tr = data[c, s, OUR].get("trajectory", {})
                        tc = data[c, s, comp].get("trajectory", {})
                        if tr.get(str(t)) and tc.get(str(t)):
                            vals.append(tr[str(t)]["U"] - tc[str(t)]["U"])
                ys.append(np.mean(vals))
            ax.plot(ts, ys, marker="o", markersize=4, label=f"{panel} 对 {comp}",
                    color={"C1": "#1f4e79", "C2": "#548235"}[panel],
                    linestyle="-", linewidth=1.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(ts)
    ax.set_xlabel("搜索时间 (s)")
    style_ax(ax, ylabel="ORDER_SPLIT 配对 ΔU", title="C1/C2 的 5/15/30/60 秒配对轨迹")
    ax.legend(frameon=False, ncol=2)
    save(fig, FIG / "P4_图3_搜索轨迹")


def main():
    data = load()
    print("P4 endpoints:", len(data))
    if len(data) < 486:
        print("P4 batch not complete yet")
        return
    fig1_18methods(data)
    fig2_paired(data)
    fig3_trajectory(data)
    print("P4 figures written")


if __name__ == "__main__":
    main()
