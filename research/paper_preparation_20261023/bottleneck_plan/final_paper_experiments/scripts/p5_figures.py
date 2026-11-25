# -*- coding: utf-8 -*-
"""P5 科研图（八类）：P5-A 四图 + P5-B 四图（PDF/SVG/PNG，paper_style 统一风格）。

输入：P5_standard_vrp/analysis_p5a.json、analysis_p5b.json。
输出：figures/P5A_图1–4、P5B_图1–4。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
P5 = HERE.parent / "P5_standard_vrp"
sys.path.insert(0, str(HERE))
import paper_style as ps

import numpy as np
import matplotlib.pyplot as plt

METHOD_ORDER = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT",
                "ALNS", "VNS", "PyVRP", "NDS-Econ", "AttentionModel", "POMO",
                "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder", "RRNCO"]
STAGES = ("100-199", "200-499", "500-1000")
FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")


def load(name):
    return json.loads((P5 / name).read_text(encoding="utf-8"))


def methods_in(a_or_b, key="summary"):
    if key == "summary":
        return [r["method"] for r in a_or_b["summary"]]
    return [m for m in METHOD_ORDER if m in a_or_b["methods"]]


def fig1_stage_gap(a):
    """P5A_图1：分规模段平均 Gap% 分组柱。"""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in a["summary"])]
    data = {m: [next((r.get(s) for r in a["summary"] if r["method"] == m), None) for s in STAGES]
            for m in methods}
    x = np.arange(len(STAGES))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, m in enumerate(methods):
        vals = [v if v is not None else 0.0 for v in data[m]]
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w, label=m,
               color=ps.METHOD_COLORS.get(m, "#999999"))
    ax.set_xticks(x)
    ax.set_xticklabels(["100–199 客户", "200–499 客户", "500–1000 客户"])
    ps.style_ax(ax, None, "平均 Gap vs BKS（%）", "P5-A 标准 CVRP：分规模段平均 Gap")
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ps.save(fig, FIG / "P5A_图1_分规模段Gap")


def fig2_rank_heatmap(a):
    """P5A_图2：实例×方法 Gap% 热图（行=实例按规模排序，列=方法）。"""
    methods = [m for m in METHOD_ORDER if any(k.endswith("|" + m) for k in a["per"])]
    instances = sorted(a["instances"], key=lambda s: int(s.split("-n")[1].split("-")[0]))
    M = np.full((len(instances), len(methods)), np.nan)
    for i, n in enumerate(instances):
        for j, m in enumerate(methods):
            v = a["per"].get(n + "|" + m)
            if v:
                M[i, j] = v["gap_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=np.nanmax(M) or 1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(0, len(instances), 8))
    ax.set_yticklabels([instances[i] for i in range(0, len(instances), 8)])
    fig.colorbar(im, ax=ax, label="Gap vs BKS（%）")
    ps.style_ax(ax, None, None, "P5-A：实例×方法 Gap 热图（行按规模升序）")
    ps.save(fig, FIG / "P5A_图2_实例方法Gap热图")


def fig3_gap_scale(a):
    """P5A_图3：Gap% 与实例规模关系（方法均值折线）。"""
    methods = [m for m in METHOD_ORDER if any(k.endswith("|" + m) for k in a["per"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for m in methods:
        pts = []
        for k, v in a["per"].items():
            if k.endswith("|" + m):
                n_cust = int(k.split("|")[0].split("-n")[1].split("-")[0]) - 1
                pts.append((n_cust, v["gap_mean"]))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2.5,
                color=ps.METHOD_COLORS.get(m, "#999999"), label=m)
    ps.style_ax(ax, "客户规模", "Gap vs BKS（%）", "P5-A：Gap 随规模变化（方法均值）")
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ps.save(fig, FIG / "P5A_图3_Gap规模趋势")


def fig4_budget(a_dirs_json):
    """P5A_图4：在线预算核验（各方法 online_s 箱线，红线=60s）。"""
    rows = []
    for d in a_dirs_json:
        out = Path(d) / "out"
        if out.exists():
            for f in out.glob("*.json"):
                r = json.loads(f.read_text(encoding="utf-8"))
                if r.get("objective") == "cvrp" and r.get("status") == "AUDITED":
                    rows.append(r)
    methods = sorted({r["method"] for r in rows}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    data = [[r["online_s"] for r in rows if r["method"] == m] for m in methods]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.boxplot(data, labels=methods, showfliers=False, widths=0.6)
    ax.axhline(60, color="#c00000", lw=1.0, ls="--")
    ax.text(0.5, 60.1, "60 s 预算", color="#c00000", fontsize=8)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ps.style_ax(ax, None, "在线耗时 online_s（秒）", "P5-A：在线预算核验")
    ps.save(fig, FIG / "P5A_图4_在线预算核验")


def fig5_family_kd(b):
    """P5B_图1：分类平均 K 与 D（双子图分组柱）。"""
    methods = methods_in(b, "methods")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, key in zip(axes, ("mean_k", "mean_d")):
        x = np.arange(len(FAMILIES))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = []
            for fam in FAMILIES:
                cells = [v for k, v in b["per"].items() if k.endswith("|" + m) and v["family"] == fam]
                vals.append(sum(c[key] for c in cells) / len(cells) if cells else 0.0)
            ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w,
                   color=ps.METHOD_COLORS.get(m, "#999999"), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES)
        ps.style_ax(ax, None, "平均车辆数 K" if key == "mean_k" else "平均距离 D",
                    "P5-B：分类平均 K" if key == "mean_k" else "P5-B：分类平均 D")
    axes[1].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ps.save(fig, FIG / "P5B_图1_分类KD")


def fig6_win_matrix(b):
    """P5B_图2：字典序两两胜负矩阵（行方法胜列方法的净胜数）。"""
    methods = methods_in(b, "methods")
    n = len(methods)
    M = np.zeros((n, n))
    for i, a_ in enumerate(methods):
        for j, c in enumerate(methods):
            if a_ == c:
                continue
            w, l, t = b["wins"].get(a_, {}).get(c, [0, 0, 0])
            M[i, j] = w - l
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    vmax = np.abs(M).max() or 1
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(methods)
    fig.colorbar(im, ax=ax, label="净胜（胜−负，种子配对累计）")
    ps.style_ax(ax, None, None, "P5-B：字典序 (K,D) 两两净胜矩阵（行对列）")
    ps.save(fig, FIG / "P5B_图2_字典序净胜矩阵")


def fig7_leakage(b):
    """P5B_图3：泄漏审计对比——训练几何 vs 折外几何的方法平均 (K,D)。"""
    methods = methods_in(b, "methods")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    roles = [("train", "训练几何(38)"), ("foldtest", "折外测试(9)"), ("validation", "验证(8)")]
    for ax, key in zip(axes, ("mean_k", "mean_d")):
        x = np.arange(len(roles))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = []
            for role, _ in roles:
                cells = [v for k, v in b["per"].items() if k.endswith("|" + m) and v["role"] == role]
                vals.append(sum(c[key] for c in cells) / len(cells) if cells else 0.0)
            ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w,
                   color=ps.METHOD_COLORS.get(m, "#999999"), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([r[1] for r in roles])
        ps.style_ax(ax, None, "平均 K" if key == "mean_k" else "平均 D",
                    "P5-B：泄漏审计·平均 K" if key == "mean_k" else "P5-B：泄漏审计·平均 D")
    axes[1].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ps.save(fig, FIG / "P5B_图3_泄漏审计对比")


def fig8_k_dist(b):
    """P5B_图4：各方法 K 分布（小提琴/箱线，56 例 3 种子平均 K）。"""
    methods = methods_in(b, "methods")
    data = []
    for m in methods:
        ks = [v["mean_k"] for k, v in b["per"].items() if k.endswith("|" + m)]
        data.append(ks)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.boxplot(data, labels=methods, showfliers=True, widths=0.6)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ps.style_ax(ax, None, "每实例 3 种子平均 K（56 例）", "P5-B：车辆数 K 分布")
    ps.save(fig, FIG / "P5B_图4_车辆数分布")


def main(dirs_a=None):
    a = load("analysis_p5a.json")
    b = load("analysis_p5b.json")
    fig1_stage_gap(a)
    fig2_rank_heatmap(a)
    fig3_gap_scale(a)
    fig4_budget(dirs_a or [str(P5 / "results_stage01"), str(P5 / "results_stage02"),
                           str(P5 / "results_stage03"), str(P5 / "results_external_pyvrp"),
                           str(P5 / "results_external_author")])
    fig5_family_kd(b)
    fig6_win_matrix(b)
    fig7_leakage(b)
    fig8_k_dist(b)
    print("P5 FIGURES DONE")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
