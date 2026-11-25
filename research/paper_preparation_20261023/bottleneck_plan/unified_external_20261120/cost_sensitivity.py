"""Decompose paired objective differences at the delivered fleet and schedule."""
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
import csv
import json


HERE = Path(__file__).resolve().parent
SYNC = HERE.parent / "supplement_sync_20261120"
PHASE = SYNC / "results" / "sync01"
METHOD = "ORDER_SPLIT"
CONTROLS = ("ORDER_SPLIT_D", "ALNS", "PyVRP")
EPS = 1e-6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired():
    if read(PHASE / "independent_audit.json")["status"] != "PASS":
        raise ValueError("A complete independently audited batch is required")
    groups = defaultdict(dict)
    meta = read(SYNC / "metadata.json")
    base = Path(__file__).resolve().parents[4]
    price = read(base / "CC_Compare/configs/publication_model_v1.json")["money"]["distance_U_per_km"]
    for path in (PHASE / "out").glob("*.json"):
        r = read(path)
        group = groups[r["case"], r["seed"]]
        if r["method"] in group:
            raise ValueError(f"Repeated {path}")
        endpoint = r["endpoint"]
        losses = (endpoint["quality_loss_value_U"], price * endpoint["distance_km"], endpoint["refrigeration_cost_U"])
        if abs(sum(losses) - endpoint["equivalent_loss_cost_U"]) > EPS:
            raise ValueError(f"Cost coefficients differ from audited endpoint {path}")
        group[r["method"]] = (losses, r["U"])
    if len(groups) != 90:
        raise ValueError("Incomplete synchronized batch")
    geometry = defaultdict(lambda: defaultdict(list))
    for (case, _seed), group in groups.items():
        if set(group) != {METHOD, *CONTROLS}:
            raise ValueError(f"Incomplete method group {case}")
        item = meta["cases"][case]
        panels = []
        if case in meta["primary"]:
            panels.append(item["dataset"])
        if case in meta["low_overlap"]:
            panels.append("XTW 低重叠")
        losses, u = group[METHOD]
        for control in CONTROLS:
            other, other_u = group[control]
            change = tuple(x-y for x, y in zip(other, losses))
            if abs(sum(change)-(u-other_u)) > EPS:
                raise ValueError(f"Paired decomposition differs for {case} {control}")
            for p in panels:
                geometry[p, control, item["geometry"]][case].append(change)
    return {key: list(map(mean, zip(*(list(map(mean, zip(*rows))) for rows in cases.values()))))
            for key, cases in geometry.items()}


def main():
    geom = paired()
    output = HERE / "economic_sensitivity"
    output.mkdir(exist_ok=True)
    corners = list(product((0.8, 1.2), repeat=3))
    axes = [tuple(scale if j==i else 1 for j in range(3)) for i in range(3) for scale in (0.8, 1.2)]
    panels = ("PDPCDPG-D100", "XTW-p100", "GH-p100", "XTW 低重叠")
    lines = ["# 已选车队的冷链成本与价格敏感性", "",
             "数据来自单核 60 秒、四法同期复核的 360 个已审计终点。正值表示 MASKCO 相对对照节约的 U。"
             "品质、里程和制冷三列之和等于该比较的原始净效益差。", "",
             "| 面板 | 对照 | 品质损失减少 U | 里程费用减少 U | 制冷费用减少 U | 基准净效益差 U | 价格同时波动 ±20% 的最低至最高 ΔU |",
             "|---|---|---:|---:|---:|---:|---:|"]
    with (output / "cost_scenarios.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(("panel","control","quality_price_factor","distance_price_factor","cold_price_factor",
                    "mean_delta_U","geometry_wins","geometry_ties","geometry_losses"))
        for p in panels:
            for m in CONTROLS:
                sample=[v for (panel,ctrl,_),v in geom.items() if panel==p and ctrl==m]
                if not sample:
                    raise ValueError(f"Missing {p} {m}")
                components=list(map(mean,zip(*sample)))
                extrema=[]
                for weights in [(1,1,1),*axes,*corners]:
                    delta=[sum(x*y for x,y in zip(weights,row)) for row in sample]
                    w.writerow((p,m,*weights,mean(delta),sum(x>EPS for x in delta),sum(abs(x)<=EPS for x in delta),sum(x< -EPS for x in delta)))
                    if weights in corners:
                        extrema.append(mean(delta))
                values=" | ".join(f"{v:+.2f}" for v in (*components,sum(components)))
                lines.append(f"| {p} | {m} | {values} | {min(extrema):+.2f} 至 {max(extrema):+.2f} |")
    lines+=["","固定既有车队、访问顺序和原审计时刻，各单价相对于主模型独立改变。因同例货量相同，货品总标价在方案差中抵消。"
            "表中给定价格区间的极值来自线性成本差的八个角点。XTW 低重叠与主面板共用七例，分别统计。",
            "","[全部价格情景和几何胜负](cost_scenarios.csv) · [四法同期复核](../../supplement_sync_20261120/results/sync01/RESULTS.md)"]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
