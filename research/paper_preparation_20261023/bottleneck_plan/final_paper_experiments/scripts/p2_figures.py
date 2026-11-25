# -*- coding: utf-8 -*-
"""P2 八城十种子五图（科研论文风格）。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P2 = FP / "P2_retail_analysis"
FIG = FP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from paper_style import METHOD_COLORS, DECOMPOSITION_COLORS, style_ax, save  # noqa: E402

CITIES = ["rw_zg170_p100", "rw_br100_p100", "rw_cp150_p100", "rw_ru120_p100",
          "rw_sg200_p100", "rw_me340_p100", "rw_ca240_p100", "rw_sf300_p100"]
NAMES = {"rw_zg170_p100": "Zagreb", "rw_br100_p100": "Brasilia", "rw_cp150_p100": "Copenhagen",
         "rw_ru120_p100": "Ruhr", "rw_sg200_p100": "Singapore", "rw_me340_p100": "Melbourne",
         "rw_ca240_p100": "Cairo", "rw_sf300_p100": "SF Bay Area"}
SEEDS = [65001, 65002, 65003, 65004, 65005, 65006, 65007, 65008, 65009, 65010]
OUR = "ORDER_SPLIT"


def load_ep():
    rows = list(csv.DictReader((P2 / "P2_八城十种子逐端点结果.csv").open(encoding="utf-8-sig")))
    data = {}
    for r in rows:
        data[(r["case"], int(r["seed"]), r["method"])] = r
    return data


def fig1_paired(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(CITIES))
    w = 0.3
    for j, comp in enumerate(("ALNS", "ORDER_SPLIT_D")):
        vals, lo, hi = [], [], []
        for c in CITIES:
            d = [float(data[c, s, OUR]["U"]) - float(data[c, s, comp]["U"]) for s in SEEDS]
            vals.append(np.mean(d))
            lo.append(np.mean(d) - np.percentile(d, 2.5))
            hi.append(np.percentile(d, 97.5) - np.mean(d))
        ax.bar(x + (j - 0.5) * w, vals, width=w, yerr=[lo, hi], capsize=2.5,
               color=METHOD_COLORS[comp], edgecolor="white", zorder=3,
               label="对 ALNS" if comp == "ALNS" else "对 ORDER_SPLIT_D")
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in CITIES], rotation=30, ha="right")
    style_ax(ax, ylabel="ORDER_SPLIT 配对 ΔU（十种子，城市内平均）",
             title="八城配对收益：对 ALNS 与同规则距离版")
    ax.legend(frameon=False)
    save(fig, FIG / "P2_图1_八城配对收益对比")


def fig2_scatter(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(7)
    for i, c in enumerate(CITIES):
        d = [float(data[c, s, OUR]["U"]) - float(data[c, s, "ALNS"]["U"]) for s in SEEDS]
        jx = rng.normal(0, 0.05, size=len(d))
        ax.scatter(np.full(len(d), i) + jx, d, s=22, color="#1f4e79", alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.mean(d)] * 2, color="#c00000", linewidth=1.6, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(CITIES)))
    ax.set_xticklabels([NAMES[c] for c in CITIES], rotation=30, ha="right")
    style_ax(ax, ylabel="ORDER_SPLIT − ALNS (U)", title="八城十种子配对增量散点（红线为城市均值）")
    save(fig, FIG / "P2_图2_八城十种子配对增量散点")


def fig3_decomposition():
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((P2 / "P2_逐城市经济收益分解.csv").open(encoding="utf-8-sig")))
    city = {}
    for r in rows:
        if r["comparator"] != "ALNS":
            continue
        city.setdefault(r["case"], []).append(r)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(CITIES))
    w = 0.26
    parts = {"quality": "品质", "distance": "里程", "refrigeration": "制冷"}
    for j, (k, lab) in enumerate(parts.items()):
        vals = [np.mean([float(r[k]) for r in city[c]]) for c in CITIES]
        ax.bar(x + (j - 1) * w, vals, width=w, label=lab, color=DECOMPOSITION_COLORS[k],
               edgecolor="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in CITIES], rotation=30, ha="right")
    style_ax(ax, ylabel="对照成本 − 本方法成本 (U)", title="对 ALNS 的三项经济收益分解（十种子平均）")
    ax.legend(frameon=False, ncol=3)
    save(fig, FIG / "P2_图3_经济收益分解")


def fig4_trajectory(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ts = [5, 15, 30, 60]
    for comp in ("ALNS", "ORDER_SPLIT_D"):
        ys, lo, hi = [], [], []
        for t in ts:
            d = []
            for c in CITIES:
                for s in SEEDS:
                    a = float(data[c, s, OUR][f"U_{t}s"])
                    b = float(data[c, s, comp][f"U_{t}s"])
                    d.append(a - b)
            ys.append(np.mean(d))
            lo.append(np.mean(d) - np.percentile(d, 2.5))
            hi.append(np.percentile(d, 97.5) - np.mean(d))
        ax.errorbar(ts, ys, yerr=[lo, hi], capsize=3, marker="o", markersize=4,
                    color=METHOD_COLORS[comp], label="对 ALNS" if comp == "ALNS" else "对 ORDER_SPLIT_D", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(ts)
    ax.set_xlabel("搜索时间 (s)")
    style_ax(ax, ylabel="ORDER_SPLIT 配对 ΔU（80 配对均值）", title="5/15/30/60 秒配对搜索轨迹")
    ax.legend(frameon=False)
    save(fig, FIG / "P2_图4_配对搜索轨迹")


def fig5_loo():
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((P2 / "P2_留一城市敏感性分析.csv").open(encoding="utf-8-sig")))
    comp_rows = [r for r in rows if r["comparator"] == "ALNS"]
    full = next(r for r in comp_rows if r["leave_out"] == "全八城")
    rest = [r for r in comp_rows if r["leave_out"] != "全八城"]
    labels = ["全八城"] + [r["leave_out"] for r in rest]
    vals = [float(full["mean"])] + [float(r["mean"]) for r in rest]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = ["#1f4e79"] + ["#4f81bd"] * len(rest)
    ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(0, float(full["mean"]), yerr=[[float(full["mean"]) - float(full["ci95_lo"])],
                                              [float(full["ci95_hi"]) - float(full["mean"])]],
                fmt="none", ecolor="black", capsize=3, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    style_ax(ax, ylabel="ORDER_SPLIT − ALNS 城市等权 ΔU", title="留一城市敏感性：剔除后余下城市的等权均值")
    save(fig, FIG / "P2_图5_留一城市敏感性")


def main():
    ep_path = P2 / "P2_八城十种子逐端点结果.csv"
    if not ep_path.is_file():
        print("P2 endpoint table not available yet")
        return
    data = load_ep()
    fig1_paired(data)
    fig2_scatter(data)
    fig3_decomposition()
    fig4_trajectory(data)
    fig5_loo()
    print("P2 figures written")


if __name__ == "__main__":
    main()
