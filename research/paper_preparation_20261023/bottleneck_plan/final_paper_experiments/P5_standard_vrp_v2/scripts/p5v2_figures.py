# -*- coding: utf-8 -*-
"""P5-V2 图表生成（12 图）：只读分析 JSON/CSV，不参与任何求解。

输出：figures/F01..F12（PNG 300dpi + PDF）。
用法：python p5v2_figures.py [--dev-only]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
FIG = V2 / "figures"
ORDER = ["STD_OLD", "STD_BUGFIX", "STD_FIX", "STD_NORM", "STD_DIST"]
INTERNAL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT"]
COLORS = {"STD_OLD": "#7f7f7f", "STD_BUGFIX": "#1f77b4", "STD_FIX": "#2ca02c",
          "STD_NORM": "#ff7f0e", "STD_DIST": "#d62728"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _setup_fonts():
    """优先使用系统 CJK 字体，避免中文标题缺字。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "Arial Unicode MS"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    return None


CJK_FONT = _setup_fonts()


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ F01–F05（开发面板）
def dev_figures(dev_dir=None):
    root = Path(dev_dir) if dev_dir else V2 / "development_results/dev01_final"
    d = load_json(root / "analysis_dev.json")
    if not d:
        print("skip dev figures (no analysis_dev.json)")
        return
    cells = d["cell_table"]
    # F01：版本相对初始见证的改善分布
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    data, labels = [], []
    for v in ORDER:
        vals = [c["versions"][v]["gain_vs_initial_pct"] for c in cells
                if v in c["versions"] and c["versions"][v].get("gain_vs_initial_pct") is not None]
        if vals:
            data.append(vals)
            labels.append("%s\n(n=%d)" % (v, len(vals)))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    for patch, v in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[v])
        patch.set_alpha(0.55)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(i).normal(0, 0.05, len(vals)),
                   vals, s=8, color="k", alpha=0.5, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("距离相对初始见证变化 (%)")
    ax.set_title("F01 开发面板：五版本相对构造式初解的改善")
    save(fig, "F01_dev_gain_vs_initial")

    # F02：STD_FIX vs STD_BUGFIX 逐单元散点（谁是主方法）
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    xs, ys, cs = [], [], []
    for c in cells:
        a = c["versions"].get("STD_BUGFIX", {})
        b = c["versions"].get("STD_FIX", {})
        if not a.get("distance") or not b.get("distance"):
            continue
        xs.append(a["distance"])
        ys.append(b["distance"])
        cs.append("#1f77b4" if c["objective"] == "cvrp" else "#9467bd")
    if xs:
        ax.scatter(xs, ys, c=cs, s=18)
        lo, hi = min(xs + ys), max(xs + ys)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("STD_BUGFIX 距离")
        ax.set_ylabel("STD_FIX 距离")
        ax.set_title("F02 全机制 vs 仅缺陷修复（点在对角线下方 = FIX 更优）")
        save(fig, "F02_fix_vs_bugfix")

    # F03：神经引导净贡献（STD_FIX vs STD_DIST）
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    xs, ys = [], []
    for c in cells:
        a = c["versions"].get("STD_DIST", {})
        b = c["versions"].get("STD_FIX", {})
        if not a.get("distance") or not b.get("distance"):
            continue
        xs.append(a["distance"])
        ys.append(b["distance"])
    if xs:
        ax.scatter(xs, ys, s=18, color="#2ca02c")
        lo, hi = min(xs + ys), max(xs + ys)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("STD_DIST（纯距离）")
        ax.set_ylabel("STD_FIX（神经+校准）")
        ax.set_title("F03 神经引导净贡献（开发面板）")
        save(fig, "F03_neural_vs_distance")

    # F04：STD_NORM vs STD_FIX
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    xs, ys = [], []
    for c in cells:
        a = c["versions"].get("STD_FIX", {})
        b = c["versions"].get("STD_NORM", {})
        if not a.get("distance") or not b.get("distance"):
            continue
        xs.append(a["distance"])
        ys.append(b["distance"])
    if xs:
        ax.scatter(xs, ys, s=18, color="#ff7f0e")
        lo, hi = min(xs + ys), max(xs + ys)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("STD_FIX（冻结编码）")
        ax.set_ylabel("STD_NORM（训练语义编码）")
        ax.set_title("F04 输入编码对齐的效果")
        save(fig, "F04_norm_vs_fix")

    # F05：机制计数（试选选项采用 / 掩码模式）
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    mech = d.get("mechanisms", {})
    vers = [v for v in ORDER if mech.get(v, {}).get("ranking_counts")]
    if vers:
        rc = np.zeros((len(vers), 3))
        for i, v in enumerate(vers):
            cnt = mech[v]["ranking_counts"]
            for k in range(3):
                rc[i, k] = cnt.get(str(k), 0)
        bottoms = np.zeros(len(vers))
        for k, lab in enumerate(["距离", "原始热图", "校准热图"]):
            ax.barh(vers, rc[:, k], left=bottoms, label=lab)
            bottoms += rc[:, k]
        ax.set_xlabel("被采用的试选配置次数")
        ax.legend(fontsize=7, ncol=3)
        ax.set_title("F05 六配置试选的选项采用分布")
        save(fig, "F05_trial_option_usage")


# ------------------------------------------------------------------ F06–F12（正式面板）
def formal_figures():
    a = load_json(V2 / "tables/analysis_formal.json")
    if not a:
        print("skip formal figures (no analysis_formal.json)")
        return
    # F06：CVRP 各规模段平均 Gap
    v22 = a.get("v2_2") or []
    if v22:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        width = 0.13
        xs = np.arange(len(v22))
        for i, m in enumerate(INTERNAL):
            vals = [row.get(m + "_mean_gap%", np.nan) for row in v22]
            ax.bar(xs + i * width, vals, width, label=m)
        ax.set_xticks(xs + 2.5 * width)
        ax.set_xticklabels([r["bucket"] for r in v22])
        ax.set_ylabel("平均 Gap 到 BKS (%)")
        ax.legend(fontsize=6, ncol=3)
        ax.set_title("F06 正式 CVRP：六内部组合的规模段 Gap")
        save(fig, "F06_cvrp_gap_by_size")

    # F07：神经 vs 距离（正式，CVRP+VRPTW）
    v27 = a.get("v2_7") or []
    if v27:
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        labels = ["%s\n%s" % (r["objective"], r["neural"]) for r in v27]
        vals = [r["mean_delta"] for r in v27]
        cols = ["#2ca02c" if r["neural_better"] > r["distance_better"] else "#d62728" for r in v27]
        ax.bar(range(len(vals)), vals, color=cols)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("神经 − 距离 (%, K 差时记 ΔK)")
        ax.set_title("F07 神经引导净贡献（负值=神经更优）")
        save(fig, "F07_neural_contribution_formal")

    # F08：VRPTW 类别平均车辆数
    v25 = a.get("v2_5") or []
    if v25:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        width = 0.13
        xs = np.arange(len(v25))
        for i, m in enumerate(INTERNAL):
            vals = [row.get(m + "_mean_K", np.nan) for row in v25]
            ax.bar(xs + i * width, vals, width, label=m, alpha=0.85)
        bks = [row.get(INTERNAL[0] + "_bks_K", np.nan) for row in v25]
        ax.plot(xs + 2.5 * width, bks, "k*", ms=7, label="BKS 路线 K")
        ax.set_xticks(xs + 2.5 * width)
        ax.set_xticklabels([r["category"] for r in v25])
        ax.set_ylabel("平均车辆数 K")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("F08 正式 VRPTW：各类别平均车辆数")
        save(fig, "F08_vrptw_K_by_category")

    # F09：VRPTW K 胜负计数
    if v25:
        fig, ax = plt.subplots(figsize=(5.6, 2.8))
        xs = np.arange(len(v25))
        width = 0.25
        win = [sum(row.get(m + "_K_win", 0) for m in INTERNAL) for row in v25]
        tie = [sum(row.get(m + "_K_tie", 0) for m in INTERNAL) for row in v25]
        loss = [sum(row.get(m + "_K_loss", 0) for m in INTERNAL) for row in v25]
        ax.bar(xs - width, win, width, label="优于 BKS K", color="#2ca02c")
        ax.bar(xs, tie, width, label="相同", color="#7f7f7f")
        ax.bar(xs + width, loss, width, label="劣于 BKS K", color="#d62728")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["category"] for r in v25])
        ax.set_ylabel("端点计数（6 组合汇总）")
        ax.legend(fontsize=7)
        ax.set_title("F09 正式 VRPTW：车辆数相对 BKS 的胜负")
        save(fig, "F09_vrptw_K_winloss")

    # F10：机制消融
    v28 = a.get("v2_8") or []
    if v28:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        labels = ["%s\n%s" % (r["objective"], r["ablation"]) for r in v28]
        vals = [r["mean_delta"] for r in v28]
        cols = ["#1f77b4" if v < 0 else "#d62728" for v in vals]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("开启机制 − 关闭机制 (%)")
        ax.set_title("F10 机制消融（Split / 插位保护）")
        save(fig, "F10_mechanism_ablation")

    # F11：历史最好解收敛
    v210 = a.get("v2_10") or []
    if v210:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
        for ax, obj in zip(axes, ("cvrp", "vrptw")):
            for m in INTERNAL:
                rows = [r for r in v210 if r["objective"] == obj and r["method"] == m]
                if not rows:
                    continue
                r = rows[0]
                xs, ys = [], []
                for cp, x in (("5", 5), ("15", 15), ("30", 30), ("60", 60)):
                    if "mean_D_t%s" % cp in r:
                        xs.append(x)
                        ys.append(r["mean_D_t%s" % cp])
                ax.plot(xs, ys, marker="o", ms=3, label=m)
            ax.set_xlabel("在线秒数")
            ax.set_ylabel("平均距离")
            ax.set_title(obj.upper())
            ax.legend(fontsize=6)
        fig.suptitle("F11 5/15/30/60 秒历史最好解")
        save(fig, "F11_history_best")

    # F12：固定候选数量诊断（四种排序的插位质量）
    d = load_json(V2 / "tables/analysis_diag.json")
    if d:
        agg = d["per_variant"]
        names = ["distance", "heat_raw", "heat_calibrated", "heat_norm"]
        label = {"distance": "距离", "heat_raw": "原始热图",
                 "heat_calibrated": "校准热图", "heat_norm": "归一化热图"}
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        xs = range(len(names))
        axes[0].bar(xs, [agg[n]["mean_recall1"] for n in names],
                    color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"])
        axes[0].set_xticks(list(xs))
        axes[0].set_xticklabels([label[n] for n in names], fontsize=7)
        axes[0].set_ylabel("最优插位召回率 @1")
        axes[0].set_title("插位选择质量")
        axes[1].bar(xs, [agg[n]["mean_regret"] for n in names],
                    color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"])
        axes[1].set_xticks(list(xs))
        axes[1].set_xticklabels([label[n] for n in names], fontsize=7)
        axes[1].set_ylabel("平均后悔值（越小越好）")
        axes[1].set_title("插位后悔值")
        fig.suptitle("F12 固定候选数量诊断（同候选集/同插入顺序，仅排序不同）")
        save(fig, "F12_fixed_candidate_diagnostics")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev_figures(args[0] if args else None)
    if "--dev-only" not in sys.argv:
        formal_figures()


if __name__ == "__main__":
    main()
