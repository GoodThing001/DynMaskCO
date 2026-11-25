# -*- coding: utf-8 -*-
"""P4 分析：C1/C2 分报告 18 法表、配对增量、轨迹汇总、可行审计、表11。"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P4 = FP / "P4_solomon_coldchain"
TAB = FP / "tables"
TAB.mkdir(parents=True, exist_ok=True)

EPS = 1e-6
METHODS = ["ORDER_SPLIT", "ALNS", "VNS", "NDS-Econ", "AttentionModel", "POMO", "DeepACO", "CaR",
           "PyVRP", "MVMoE", "CaDA", "RouteFinder", "RRNCO", "ORDER_SPLIT_D", "DIST", "DIST_SLOT",
           "MASK", "SLOT"]
EXTERNAL = ["ALNS", "VNS", "NDS-Econ", "AttentionModel", "POMO", "DeepACO", "CaR", "PyVRP",
            "MVMoE", "CaDA", "RouteFinder", "RRNCO"]
OUR = "ORDER_SPLIT"
C1 = ["c105", "c106", "c107", "c108", "c109"]
C2 = ["c205", "c206", "c207", "c208"]

PUB = {
    "ORDER_SPLIT": ("完整方法（本项目实现）", "本项目实现", "—", "本项目无独立发表等级"),
    "ORDER_SPLIT_D": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "DIST": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "DIST_SLOT": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "MASK": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "SLOT": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "ALNS": ("外部方法", "Transportation Science", "2006", "中科院工程技术2区*（2025升级版）"),
    "VNS": ("外部方法", "Computers & Operations Research", "1997", "中科院工程技术2区*（2025升级版）"),
    "NDS-Econ": ("外部方法", "TMLR", "2025", "中科院分区未核实（2025）"),
    "AttentionModel": ("外部方法", "ICLR", "2019", "CCF A（CCF2026第七版）"),
    "POMO": ("外部方法", "NeurIPS", "2020", "CCF A（CCF2026第七版）"),
    "DeepACO": ("外部方法", "NeurIPS", "2023", "CCF A（CCF2026第七版）"),
    "CaR": ("外部方法", "ICLR", "2026", "CCF A（CCF2026第七版）"),
    "PyVRP": ("外部方法", "INFORMS Journal on Computing", "2024", "中科院计算机科学3区*（2025升级版）"),
    "MVMoE": ("外部方法", "ICML", "2024", "CCF A（CCF2026第七版）"),
    "CaDA": ("外部方法", "ICML", "2025", "CCF A（CCF2026第七版）"),
    "RouteFinder": ("外部方法", "TMLR", "2025", "中科院分区未核实（2025）"),
    "RRNCO": ("外部方法", "ICLR", "2026", "CCF A（CCF2026第七版）"),
}


def rank_of(values):
    return {k: 1 + sum(x >= v - EPS for m, x in values.items() if k != m) for k, v in values.items()}


def main():
    out_dir = P4 / "results" / "out"
    files = sorted(out_dir.glob("*.json"))
    print("P4 endpoints:", len(files))
    if len(files) < 486:
        print("P4 batch not complete yet")
        return
    data = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        data[(d["case"], d["seed"], d["method"])] = d
    # 审计
    issues = []
    for (c, s, m), d in data.items():
        if d["status"] != "AUDITED" or not d["endpoint"]["independently_audited"]:
            issues.append((c, s, m, "not audited"))
        if not 0 <= d["selected_s"] <= 60 or d["U"] < d["initial_U"] - EPS:
            issues.append((c, s, m, "timing/incumbent"))
        if abs(d["U"] - d["endpoint"]["net_benefit_U"]) > EPS:
            issues.append((c, s, m, "objective"))
        tr = d.get("trajectory")
        if tr is None and not d.get("no_search_clock"):
            issues.append((c, s, m, "no trajectory"))
    print("audit issues:", len(issues))
    for i in issues[:10]:
        print(" -", i)
    # 面板聚合：案例内种子均值 → 面板等权
    panels = {"C1": C1, "C2": C2}
    scores = {}
    rows = []
    for panel, cases in panels.items():
        sc = {}
        for m in METHODS:
            vals = [np.mean([data[c, s, m]["U"] for s in (67001, 67002, 67003)]) for c in cases]
            sc[m] = float(np.mean(vals))
        scores[panel] = sc
        rank13 = rank_of({k: sc[k] for k in [OUR] + EXTERNAL})
        rank18 = rank_of(sc)
        for m in METHODS:
            cat, venue, year, grade = PUB[m]
            rows.append(dict(panel=panel, method=m, category=cat, venue=venue, year=year, grade=grade,
                             mean_U=round(sc[m], 6), rank13=rank13.get(m, ""), rank18=rank18.get(m, "")))
    with (TAB / "表11_Solomon冷链18法.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 配对增量（C1、C2、合并9例）与经济损失降低率
    inc = []
    for panel, cases in [("C1", C1), ("C2", C2), ("合并9例", C1 + C2)]:
        for comp in ("MASK", "SLOT", "ALNS", "ORDER_SPLIT_D"):
            diffs = [data[c, s, OUR]["U"] - data[c, s, comp]["U"] for c in cases for s in (67001, 67002, 67003)]
            loss_reds = []
            for c in cases:
                lo = np.mean([data[c, s, OUR]["endpoint"]["equivalent_loss_cost_U"] for s in (67001, 67002, 67003)])
                lb = np.mean([data[c, s, comp]["endpoint"]["equivalent_loss_cost_U"] for s in (67001, 67002, 67003)])
                loss_reds.append(100.0 * (lb - lo) / lb if lb > 0.5 else None)
            usable = [x for x in loss_reds if x is not None]
            inc.append(dict(panel=panel, comparator=comp, delta_U=round(float(np.mean(diffs)), 4),
                            min_delta=round(float(np.min(diffs)), 4), max_delta=round(float(np.max(diffs)), 4),
                            positive_cells=sum(1 for x in diffs if x > EPS), total_cells=len(diffs),
                            loss_reduction_pct=round(float(np.mean(usable)), 4) if usable else None,
                            loss_reduction_unusable=sum(1 for x in loss_reds if x is None)))
    with (TAB / "表11b_Solomon配对增量.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inc[0].keys()))
        w.writeheader()
        w.writerows(inc)
    # 轨迹汇总：面板 × 方法 × 5/15/30/60 均值
    traj_rows = []
    for panel, cases in panels.items():
        for m in METHODS:
            for t in ("5", "15", "30", "60"):
                vals = []
                for c in cases:
                    for s in (67001, 67002, 67003):
                        tr = data[c, s, m].get("trajectory")
                        if tr and tr.get(t):
                            vals.append(tr[t]["U"])
                if vals:
                    traj_rows.append(dict(panel=panel, method=m, cutoff_s=t, mean_U=round(float(np.mean(vals)), 4),
                                          n=len(vals)))
    with (TAB / "表11c_Solomon轨迹汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)
    # 表11 MD
    md = ["# 表11 Solomon 冷链 18 法比较（P4）", "",
          "C1=c105–109（fold0 折外几何，历史 confirm01 已访问）；C2=c205–208（fold0 验证几何）。"
          "案例内种子均值→面板等权；并列取末位；单一空间几何无簇区间。", ""]
    for panel in ("C1", "C2"):
        md += [f"## {panel}", "",
               "| 方法 | 类别 | 平均净效益 U | 名次/13 | 名次/18 |",
               "|---|---|---:|---:|---:|"]
        for r in rows:
            if r["panel"] == panel:
                md.append(f"| {r['method']} | {r['category']} | {r['mean_U']:,.4f} | {r['rank13']} | {r['rank18']} |")
        md.append("")
    (TAB / "表11_Solomon冷链18法.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("P4 tables written")


if __name__ == "__main__":
    main()
