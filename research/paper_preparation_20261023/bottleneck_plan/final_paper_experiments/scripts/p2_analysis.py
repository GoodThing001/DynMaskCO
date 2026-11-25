# -*- coding: utf-8 -*-
"""P2 八城十种子分析：结构特征、经济分解、轨迹、留一城市敏感性。

特征定义与分组阈值严格按 P2_预登记.md；全部统计在读取端点前已登记。
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P2 = FP / "P2_retail_analysis"
UNI = FP.parent / "unified_external_20261120"
OUT = P2
EPS = 1e-6
BOOT_SEED = 20261126
DRAWS = 20000

CITIES = ["rw_zg170_p100", "rw_br100_p100", "rw_cp150_p100", "rw_ru120_p100",
          "rw_sg200_p100", "rw_me340_p100", "rw_ca240_p100", "rw_sf300_p100"]
CITY_NAMES = {"rw_zg170_p100": "Zagreb", "rw_br100_p100": "Brasilia", "rw_cp150_p100": "Copenhagen",
              "rw_ru120_p100": "Ruhr", "rw_sg200_p100": "Singapore", "rw_me340_p100": "Melbourne",
              "rw_ca240_p100": "Cairo", "rw_sf300_p100": "SF Bay Area"}
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS"]
SEEDS = [65001, 65002, 65003, 65004, 65005, 65006, 65007, 65008, 65009, 65010]
CUTOFFS = ("5", "15", "30", "60")
CAPACITY_T = 20.0


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_inputs():
    out = {}
    for c in CITIES:
        d = read(P2 / "inputs" / (c + ".json"))
        out[c] = d["nodes"]
    return out


def city_features(nodes):
    cust = [n for n in nodes if n[0] != 0]
    coords = np.asarray([[n[1], n[2]] for n in cust], dtype=float)
    depot = np.asarray([nodes[0][1], nodes[0][2]], dtype=float)
    n = len(coords)
    d2d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    nn = np.partition(d2d, 1, axis=1)[:, 1]
    dep = np.linalg.norm(coords - depot, axis=1)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    expected_nn = 0.5 * math.sqrt(max(area, 1e-12) / n)
    clark_evans = float(nn.mean() / expected_nn) if expected_nn > 0 else float("nan")
    demand = np.asarray([x[3] for x in cust], dtype=float)
    width = np.asarray([x[5] - x[4] for x in cust], dtype=float)
    service = np.asarray([x[6] for x in cust], dtype=float)
    return dict(mean_pairwise=float(d2d.sum() / (n * (n - 1))), mean_nn=float(nn.mean()),
                bbox_diagonal=diag, bbox_area=area, mean_depot=float(dep.mean()),
                clark_evans=clark_evans, depot_cv=float(dep.std() / dep.mean()),
                demand_mean=float(demand.mean()), demand_std=float(demand.std()),
                demand_cv=float(demand.std() / demand.mean()), tw_width=float(width.mean()),
                service_mean=float(service.mean()), total_demand_t=float(demand.sum() / 1000.0))


def store_shares():
    sel = read(UNI / "retail_source" / "selection.json")
    out = {}
    for c in CITIES:
        st = sel["cases"][c]["store_types"]
        sup = st.get("supermarket", 0)
        conv = st.get("convenience", 0)
        out[c] = sup / (sup + conv) if sup + conv else float("nan")
    return out


def load_endpoints():
    base = P2 / "results" / "out"
    data = {}
    for f in sorted(base.glob("*.json")):
        d = read(f)
        key = (d["case"], d["seed"], d["method"])
        data[key] = d
    return data


def main():
    nodes = load_inputs()
    shares = store_shares()
    feats = {}
    for c in CITIES:
        feats[c] = city_features(nodes[c])
        feats[c]["city"] = CITY_NAMES[c]
        feats[c]["store_share"] = round(shares[c], 4)
        feats[c]["store_group"] = "高(≥2/3)" if shares[c] >= 2 / 3 else ("中(1/3–2/3)" if shares[c] >= 1 / 3 else "低(<1/3)")
    with (OUT / "P2_城市结构特征统计.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feats[CITIES[0]].keys()))
        w.writeheader()
        for c in CITIES:
            w.writerow(feats[c])
    # 端点
    data = load_endpoints()
    print("P2 endpoints loaded:", len(data))
    if not data:
        print("P2 not run yet; features written.")
        return
    ep_rows = []
    for key in sorted(data):
        c, s, m = key
        d = data[key]
        traj = d.get("trajectory", {})
        row = dict(city=CITY_NAMES[c], case=c, seed=s, method=m, status=d["status"], U=d["U"],
                   initial_U=d["initial_U"], selected_s=d["selected_s"],
                   vehicles=d["endpoint"]["used_vehicles"], distance_km=d["endpoint"]["distance_km"],
                   quality_loss_U=d["endpoint"]["quality_loss_value_U"],
                   refrigeration_U=d["endpoint"]["refrigeration_cost_U"],
                   minimum_quality=d["endpoint"]["minimum_quality"])
        for t in CUTOFFS:
            sn = traj.get(t)
            row[f"U_{t}s"] = sn["U"] if sn else ""
            row[f"vehicles_{t}s"] = len(sn["routes"]) if sn else ""
        ep_rows.append(row)
    with (OUT / "P2_八城十种子逐端点结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
        w.writeheader()
        w.writerows(ep_rows)
    # 逐城市经济分解（逐种子）
    dec_rows = []
    for c in CITIES:
        for s in SEEDS:
            os_ = data[c, s, "ORDER_SPLIT"]
            for comp in ("ALNS", "ORDER_SPLIT_D"):
                b = data[c, s, comp]
                q = b["endpoint"]["quality_loss_value_U"] - os_["endpoint"]["quality_loss_value_U"]
                dist = b["endpoint"]["distance_km"] - os_["endpoint"]["distance_km"]
                refr = b["endpoint"]["refrigeration_cost_U"] - os_["endpoint"]["refrigeration_cost_U"]
                dec_rows.append(dict(city=CITY_NAMES[c], case=c, seed=s, comparator=comp,
                                     quality=q, distance=dist, refrigeration=refr,
                                     total=q + dist + refr, delta_U=os_["U"] - b["U"]))
    with (OUT / "P2_逐城市经济收益分解.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dec_rows[0].keys()))
        w.writeheader()
        w.writerows(dec_rows)
    # 留一城市敏感性（城市内先均种子）
    rng = np.random.default_rng(BOOT_SEED)
    city_diff = {c: {comp: np.mean([data[c, s, "ORDER_SPLIT"]["U"] - data[c, s, comp]["U"] for s in SEEDS])
                     for comp in ("ALNS", "ORDER_SPLIT_D")} for c in CITIES}
    order = CITIES
    loo_rows = []
    for comp in ("ALNS", "ORDER_SPLIT_D"):
        all_mean = float(np.mean([city_diff[c][comp] for c in order]))
        draws = np.asarray([city_diff[c][comp] for c in order])[rng.integers(len(order), size=(DRAWS, len(order)))].mean(axis=1)
        lo, hi = np.quantile(draws, [.025, .975])
        loo_rows.append(dict(leave_out="全八城", comparator=comp, mean=round(all_mean, 6),
                             ci95_lo=round(float(lo), 6), ci95_hi=round(float(hi), 6)))
        for c in order:
            rest = [x for x in order if x != c]
            m = float(np.mean([city_diff[x][comp] for x in rest]))
            loo_rows.append(dict(leave_out=CITY_NAMES[c], comparator=comp, mean=round(m, 6),
                                 ci95_lo="", ci95_hi=""))
    with (OUT / "P2_留一城市敏感性分析.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loo_rows[0].keys()))
        w.writeheader()
        w.writerows(loo_rows)
    # 稳定性 MD 摘要
    lines = ["# P2 八城稳定性分析", "",
             "城市内先均十种子；八城总体为城市等权。95% 描述区间为城市整簇有放回重采样 20,000 次（种子 20261126）。",
             "正值表示 ORDER_SPLIT 更高。", "", "## 城市配对差（对 ALNS 与同规则距离版）", "",
             "| 城市 | 超市占比 | 分组 | 对 ALNS ΔU | 对 ORDER_SPLIT_D ΔU |",
             "|---|---|---:|---:|---:|"]
    for c in order:
        f_ = feats[c]
        lines.append(f"| {CITY_NAMES[c]} | {f_['store_share']:.2f} | {f_['store_group']} | "
                     f"{city_diff[c]['ALNS']:+,.4f} | {city_diff[c]['ORDER_SPLIT_D']:+,.4f} |")
    lines += ["", "## 逐种子方向", ""]
    for comp in ("ALNS", "ORDER_SPLIT_D"):
        for s in SEEDS:
            vals = [data[c, s, "ORDER_SPLIT"]["U"] - data[c, s, comp]["U"] for c in CITIES]
            lines.append(f"- 种子 {s} 对 {comp}：八城均值 {np.mean(vals):+,.4f} U，正城 {sum(1 for v in vals if v > EPS)}/8")
    (OUT / "P2_八城稳定性分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P2 analysis outputs written")


if __name__ == "__main__":
    main()
