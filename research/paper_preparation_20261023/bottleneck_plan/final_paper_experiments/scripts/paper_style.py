# -*- coding: utf-8 -*-
"""科研论文绘图公共模块：统一字体、配色与输出（PDF/SVG/300DPI PNG）。

字体：中文宋体（SimSun）、西文与数字 Times New Roman；白底；低饱和度统一配色；
不同方法在全部论文图片中使用同一颜色映射。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体
for f in ("C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "figure.dpi": 150,
})

# 全论文统一方法配色（低饱和度）
METHOD_COLORS = {
    "ORDER_SPLIT": "#1f4e79",      # 深蓝（完整方法）
    "ORDER_SPLIT_D": "#4f81bd",    # 浅蓝（同规则距离）
    "HYBRID": "#7030a0",           # 紫
    "HYBRID_D": "#b4a7d6",         # 浅紫
    "MASK": "#7f7f7f",             # 灰
    "SLOT": "#a6a6a6",             # 浅灰
    "ALNS": "#c00000",             # 暗红（最强外部基线）
    "VNS": "#e36c09",
    "NDS-Econ": "#548235",
    "CaR": "#bf8f00",
    "DeepACO": "#806000",
    "POMO": "#7030a0",
    "AttentionModel": "#5b9bd5",
    "RouteFinder": "#7030a0",
    "CaDA": "#00b0f0",
    "MVMoE": "#ffc000",
    "RRNCO": "#ed7d31",
    "PyVRP": "#00b050",
    "DIST": "#808080",
    "DIST_SLOT": "#c0c0c0",
}

DECOMPOSITION_COLORS = {"quality": "#1f4e79", "distance": "#c00000", "refrigeration": "#548235"}


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, path_stem):
    path_stem = str(path_stem)
    fig.savefig(path_stem + ".pdf", bbox_inches="tight")
    fig.savefig(path_stem + ".svg", bbox_inches="tight")
    fig.savefig(path_stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
