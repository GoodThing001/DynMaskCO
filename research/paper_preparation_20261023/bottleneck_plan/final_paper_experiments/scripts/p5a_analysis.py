# -*- coding: utf-8 -*-
"""P5-A 标准 CVRP 分析：逐例 Gap vs 官方 BKS、分规模段汇总、方法排名（按 P5_预登记 §8）。

输入：各阶段结果目录（results_stage01/02/03、results_external_*）。
输出：控制台汇总 + P5_standard_vrp/analysis_p5a.json（供绘图与表12–14 使用）。

口径（预登记 §6）：BKS 值 = 官方 .sol 路线用本地评价器逐边整数舍入复算（97/100 与官方
Cost 完全一致，其余 3 例两口径重合）；每 (实例,方法) 用 3 种子平均 D 计算 Gap；
分规模段（100–199/200–499/500–1000）逐例 Gap 等权平均；排名按种子平均 D。
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"
sys.path.insert(0, str(P5))
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp

METHODS_ALL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT",
               "ALNS", "VNS", "PyVRP", "NDS-Econ", "AttentionModel", "POMO",
               "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder", "RRNCO"]


def bks_value(name):
    sol = P5 / "data/cvrplib/bks" / (name + ".sol")
    text = sol.read_text(encoding="utf-8")
    inst = CVRPInstance.parse_vrp((P5 / "data/cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))
    routes = []
    for line in text.splitlines():
        if line.startswith("Route #"):
            routes.append(tuple(int(x) for x in line.split(":")[1].split()))
    ok, d, loads, k = evaluate_cvrp(inst, routes, rounded=True)
    if not ok:
        raise ValueError("BKS infeasible: " + name)
    m = re.search(r"Cost\s+([0-9.]+)", text)
    official = float(m.group(1)) if m else None
    return d, official, k, routes


def load_results(dirs):
    rows = []
    for d in dirs:
        base = Path(d)
        out = base / "out"
        if not out.exists():
            out = base / "endpoints"
        if not out.exists():
            continue
        for f in sorted(out.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("objective") == "cvrp" and r.get("status") == "AUDITED":
                rows.append(r)
    return rows


def main(dirs, out_json):
    rows = load_results(dirs)
    by_key = {}
    for r in rows:
        key = (r["case"], r["method"], r["seed"])
        by_key[key] = r
    instances = sorted({r["case"] for r in rows},
                       key=lambda s: int(re.search(r"X-n(\d+)-", s).group(1)))
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: METHODS_ALL.index(m) if m in METHODS_ALL else 99)
    per = {}  # (instance, method) -> dict(mean_d, gaps[], k_median, bks, gap_mean)
    for name in instances:
        bks_d, official, bks_k, bks_routes = bks_value(name)
        for m in methods:
            cells = [by_key[(name, m, s)] for s in (68001, 68002, 68003) if (name, m, s) in by_key]
            if not cells:
                continue
            ds = [c["distance"] for c in cells]
            mean_d = sum(ds) / len(ds)
            gaps = [(d - bks_d) / bks_d * 100 for d in ds]
            per[(name, m)] = dict(mean_d=mean_d, gaps=gaps, gap_mean=sum(gaps) / len(gaps),
                                  ks=[c["vehicles"] for c in cells],
                                  bks_d=bks_d, official=official, bks_k=bks_k,
                                  seeds=len(cells))
    # 规模段
    def stage_of(name):
        n = int(re.search(r"X-n(\d+)-", name).group(1)) - 1
        return "100-199" if n <= 199 else ("200-499" if n <= 499 else "500-1000")
    summary = []
    for m in methods:
        row = dict(method=m)
        for stage in ("100-199", "200-499", "500-1000", "overall"):
            gaps = [per[(n, m)]["gap_mean"] for n in instances
                    if (n, m) in per and (stage == "overall" or stage_of(n) == stage)]
            if gaps:
                row[stage] = sum(gaps) / len(gaps)
                row[stage + "_n"] = len(gaps)
        if row.get("overall") is not None:
            summary.append(row)
    # 排名：每实例按 mean_d 排名
    ranks = {}
    for n in instances:
        cands = [(per[(n, m)]["mean_d"], m) for m in methods if (n, m) in per]
        cands.sort()
        for i, (_, m) in enumerate(cands, 1):
            per[(n, m)]["rank"] = i
            ranks.setdefault(m, []).append(i)
    rank_summary = {m: dict(avg_rank=sum(v) / len(v), n=len(v)) for m, v in ranks.items()}

    print("== P5-A CVRP 汇总（种子平均 D、Gap% vs BKS）==")
    print("%-14s %8s %8s %8s %8s %6s %7s" % ("method", "100-199", "200-499", "500-1000", "overall", "n", "avgRank"))
    for row in summary:
        print("%-14s %8.2f %8.2f %8.2f %8.2f %6d %7.2f"
              % (row["method"], row.get("100-199", float("nan")), row.get("200-499", float("nan")),
                 row.get("500-1000", float("nan")), row.get("overall", float("nan")),
                 row.get("overall_n", 0), rank_summary.get(row["method"], {}).get("avg_rank", float("nan"))))
    out = P5 / out_json
    out.write_text(json.dumps(dict(per={("%s|%s" % k): v for k, v in per.items()},
                                   summary=summary, ranks=rank_summary,
                                   instances=instances, methods=methods),
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, "cells:", len(rows))


if __name__ == "__main__":
    dirs = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        str(P5 / "results_stage01"), str(P5 / "results_stage02"), str(P5 / "results_stage03")]
    out = sys.argv[2] if len(sys.argv) > 2 else "analysis_p5a.json"
    main(dirs, out)
