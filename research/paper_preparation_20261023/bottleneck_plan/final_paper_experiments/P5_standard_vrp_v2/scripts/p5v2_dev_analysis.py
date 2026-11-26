# -*- coding: utf-8 -*-
"""P5-V2 开发面板分析：五版本对照 + 预测器/机制诊断。

用法：python p5v2_dev_analysis.py [results_dir]
输入：<results_dir>/out/*.json + <results_dir>/phase.json
输出：analysis_dev.json、tables_dev.md、tables_dev.csv
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ORDER = ["STD_OLD", "STD_BUGFIX", "STD_FIX", "STD_NORM", "STD_DIST"]


def load(results_dir: Path):
    rows = []
    for f in sorted((results_dir / "out").glob("*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return rows


def pct(a, b):
    """a 相对 b 的变化百分比（正=更差，距离目标）。"""
    return None if not b else 100.0 * (a - b) / b


def main(results_dir=None):
    results_dir = Path(results_dir) if results_dir else V2 / "development_results/dev01"
    rows = load(results_dir)
    if not rows:
        raise SystemExit("no endpoints in " + str(results_dir))
    cells = {}
    for r in rows:
        key = (r["case"], r["seed"], r["objective"])
        a = cells.setdefault(key, {"case": r["case"], "seed": r["seed"],
                                   "objective": r["objective"],
                                   "n": (r.get("stats") or {}).get("_n"),
                                   "versions": {}})
        a["versions"][r["version"]] = dict(
            distance=r["distance"], vehicles=r["vehicles"], online_s=r["online_s"],
            status=r["status"], epochs=(r.get("stats") or {}).get("epochs"),
            iterations=(r.get("stats") or {}).get("iterations"),
            improvements=(r.get("stats") or {}).get("improvements"),
            split_blocks=(r.get("stats") or {}).get("split_blocks"),
            split_orders=(r.get("stats") or {}).get("split_orders"),
            split_offers=(r.get("stats") or {}).get("split_offers"),
            regret_epochs=(r.get("stats") or {}).get("regret_epochs"),
            ranking_counts=(r.get("stats") or {}).get("ranking_counts"),
            mask_mode_counts=(r.get("stats") or {}).get("mask_mode_counts"),
            fallbacks=(r.get("stats") or {}).get("fallbacks"),
            predictor=(r.get("stats") or {}).get("predictor"),
            initial_distance=r["initial_U"]["distance"],
            initial_vehicles=r["initial_U"]["vehicles"])
        a["n"] = a["n"] or len(r["routes"][0]) if r.get("routes") else a["n"]

    # 每单元：以初始见证为基准
    for key, cell in cells.items():
        for v in ORDER:
            d = cell["versions"].get(v)
            if d and d.get("distance") is not None:
                d["gain_vs_initial_pct"] = pct(d["distance"], d["initial_distance"])
    # 每版本聚合
    agg = {}
    for v in ORDER:
        ds, gains, vs, ep, on, fb, crashed = [], [], [], [], [], [], []
        for cell in cells.values():
            d = cell["versions"].get(v)
            if not d:
                continue
            if d.get("distance") is None:
                crashed.append(d.get("status"))
                continue
            ds.append(d["distance"])
            gains.append(d["gain_vs_initial_pct"])
            vs.append(d["vehicles"])
            if d.get("epochs") is not None:
                ep.append(d["epochs"])
            on.append(d["online_s"])
            fb.append(d.get("fallbacks") or 0)
        if not ds:
            continue
        agg[v] = dict(endpoints=len(ds),
                      mean_distance=st.mean(ds), median_distance=st.median(ds),
                      mean_gain_vs_initial_pct=st.mean(gains),
                      median_gain_vs_initial_pct=st.median(gains),
                      mean_vehicles=st.mean(vs),
                      mean_epochs=st.mean(ep) if ep else None,
                      mean_online_s=st.mean(on), max_online_s=max(on),
                      total_fallbacks=sum(fb),
                      crashed=len(crashed))
    # 版本两两配对：按 (case,seed) 对齐
    pairs = {}
    for a, b in [("STD_OLD", "STD_BUGFIX"), ("STD_BUGFIX", "STD_FIX"),
                 ("STD_FIX", "STD_NORM"), ("STD_FIX", "STD_DIST"),
                 ("STD_NORM", "STD_DIST"), ("STD_OLD", "STD_FIX"),
                 ("STD_OLD", "STD_DIST")]:
        diffs, wins = [], 0
        for cell in cells.values():
            x, y = cell["versions"].get(a), cell["versions"].get(b)
            if not x or not y or x.get("distance") is None or y.get("distance") is None:
                continue
            diffs.append(pct(y["distance"], x["distance"]))
            if y["distance"] < x["distance"] - 1e-9:
                wins += 1
        if diffs:
            pairs["%s->%s" % (a, b)] = dict(n=len(diffs), mean_delta_pct=st.mean(diffs),
                                            median_delta_pct=st.median(diffs),
                                            b_better=wins)
    # 预测器诊断
    pred = {}
    for v in ORDER:
        attempts = success = failed = fwd = 0
        errors = []
        for cell in cells.values():
            p = (cell["versions"].get(v) or {}).get("predictor")
            if not p:
                continue
            attempts += p.get("attempts", 0)
            success += p.get("success", 0)
            failed += p.get("failed", 0)
            fwd += p.get("forwards", p.get("forward", 0)) or 0
            errors.extend(p.get("errors", [])[:1])
        pred[v] = dict(attempts=attempts, success=success, failed=failed,
                       forwards=fwd,
                       success_rate=(success / attempts) if attempts else None,
                       error_samples=errors[:3])
    # 机制计数（V2 版本）
    mech = {}
    for v in ORDER:
        rc, mm, re_ = {}, {}, 0
        for cell in cells.values():
            d = cell["versions"].get(v) or {}
            for src, dst in ((d.get("ranking_counts"), rc), (d.get("mask_mode_counts"), mm)):
                if isinstance(src, dict):
                    for k, n in src.items():
                        dst[str(k)] = dst.get(str(k), 0) + n
                elif isinstance(src, (list, tuple)):
                    for i, n in enumerate(src):
                        dst[str(i)] = dst.get(str(i), 0) + n
            re_ += d.get("regret_epochs") or 0
        mech[v] = dict(ranking_counts=rc, mask_mode_counts=mm, regret_epochs=re_)

    out = dict(endpoints=len(rows), cells=len(cells),
               completed_versions={v: sum(1 for c in cells.values() if v in c["versions"])
                                   for v in ORDER},
               per_version=agg, pairs=pairs, predictor=pred, mechanisms=mech,
               cell_table=[dict(case=k[0], seed=k[1], objective=k[2], n=v.get("n"),
                                versions=v["versions"]) for k, v in sorted(cells.items())])
    (results_dir / "analysis_dev.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# P5-V2 开发面板结果（五版本对照）", "",
             "端点数 %d；单元数 %d。" % (len(rows), len(cells)), "",
             "## 版本聚合（距离目标，CVRP 与 VRPTW 混合，仅作开发诊断）", "",
             "| 版本 | 端点 | 平均距离 | 中位距离 | 相对初始改善% | 平均车辆 | 平均 epochs | 平均在线s | 最大在线s | 回退总计 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v in ORDER:
        a = agg.get(v)
        if not a:
            continue
        lines.append("| %s | %d | %.2f | %.2f | %+.2f | %.2f | %s | %.2f | %.2f | %d |" % (
            v, a["endpoints"], a["mean_distance"], a["median_distance"],
            a["mean_gain_vs_initial_pct"], a["mean_vehicles"],
            ("%.1f" % a["mean_epochs"]) if a["mean_epochs"] is not None else "-",
            a["mean_online_s"], a["max_online_s"], a["total_fallbacks"]))
    lines += ["", "## 配对比较（后者相对前者的距离变化%）", "",
              "| 对比 | n | 平均Δ% | 中位Δ% | 后者更优 |", "|---|---|---|---|---|"]
    for k, p in pairs.items():
        lines.append("| %s | %d | %+.3f | %+.3f | %d |" % (
            k, p["n"], p["mean_delta_pct"], p["median_delta_pct"], p["b_better"]))
    lines += ["", "## 预测器诊断", "",
              "| 版本 | attempts | success | failed | forward | 成功率 |", "|---|---|---|---|---|---|"]
    for v in ORDER:
        p = pred.get(v) or {}
        if not p.get("attempts"):
            continue
        lines.append("| %s | %d | %d | %d | %d | %s |" % (
            v, p["attempts"], p["success"], p["failed"], p["forwards"],
            ("%.4f" % p["success_rate"]) if p["success_rate"] is not None else "-"))
    lines += ["", "## 机制计数", "", "| 版本 | ranking_counts | mask_modes | regret_epochs |",
              "|---|---|---|---|"]
    for v in ORDER:
        m = mech.get(v) or {}
        lines.append("| %s | %s | %s | %d |" % (v, m.get("ranking_counts"),
                                                m.get("mask_mode_counts"),
                                                m.get("regret_epochs") or 0))
    (results_dir / "tables_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (results_dir / "tables_dev.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["case", "seed", "objective", "version", "distance", "vehicles",
                    "initial_distance", "gain_vs_initial_pct", "online_s", "epochs",
                    "iterations", "improvements", "split_orders", "regret_epochs",
                    "fallbacks"])
        for key, cell in sorted(cells.items()):
            for v in ORDER:
                d = cell["versions"].get(v)
                if not d:
                    continue
                w.writerow([cell["case"], cell["seed"], cell["objective"], v,
                            "%.6f" % d["distance"], d["vehicles"],
                            "%.6f" % d["initial_distance"],
                            ("%.4f" % d["gain_vs_initial_pct"])
                            if d.get("gain_vs_initial_pct") is not None else "",
                            "%.3f" % d["online_s"], d.get("epochs"), d.get("iterations"),
                            d.get("improvements"), d.get("split_orders"),
                            d.get("regret_epochs"), d.get("fallbacks")])
    print("\n".join(lines[:40]))
    return out


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
