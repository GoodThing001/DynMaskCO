"""Measure a two-start racing decision from audited, pre-sampled development paths."""
from collections import defaultdict
from statistics import mean
import json

from trajectory_routes import HERE, ARMS, read

OUT = HERE / "results/trajectory_choice01"


def choice(first, second, checkpoint):
    a, b = (path["256"]["U"] for path in (first, second))
    winner = b if second[checkpoint]["U"] > first[checkpoint]["U"] + 1.e-6 else a
    return dict(U=winner, oracle=max(a,b), loss=max(a,b)-winner,
                correct=int(winner >= max(a,b)-1.e-6), distinct=int(abs(a-b)>1.e-6))


def aggregate(rows, field, arm):
    groups = defaultdict(list)
    for r in rows:
        groups[r["geometry"]].append(field(r["arms"][arm], r["arms"]["distance"]))
    return {g:mean(v) for g,v in groups.items()}


def main():
    config = read(HERE/"carryprobe01.json")
    catalog = read(HERE/"heldout_sources.json")["cases"]
    analysis = read(HERE/"results/carryprobe01/analysis.json")
    if len(config["blocks"]) != 27 or analysis["seeds"] != [51291,51292]:
        raise ValueError("Two complete development trajectories per case required")
    rows = []
    for i,b in enumerate(config["blocks"]):
        r = read(HERE/f"results/carryprobe01/out/{i}.json")
        if b["case"] != r["case"] or [z["seed"] for z in r["seeds"]] != analysis["seeds"]:
            raise ValueError("Checkpoint block differs")
        for j,donor in enumerate(r["seeds"]):
            anchor = r["seeds"][1-j]["arms"]["distance"]["checkpoints"]
            arms = {a:{k:choice(anchor,donor["arms"][a]["checkpoints"],k) for k in ("0","64")} for a in ARMS}
            rows.append(dict(case=b["case"],**catalog[b["case"]],seed=donor["seed"],arms=arms))
    scores={}
    for source in sorted({r["dataset"] for r in rows}):
        group=[r for r in rows if r["dataset"]==source]
        def delta(checkpoint,arm="eight",part=group):
            return aggregate(part,lambda a,d:a[checkpoint]["U"]-d[checkpoint]["U"],arm)
        geo=delta("64")
        metrics = dict(geometries=geo,wins=sum(v>1.e-6 for v in geo.values()),mean=mean(geo.values()),
                       no_scout=mean(delta("0").values()),original=mean(delta("64","original").values()),
                       shuffled=mean(delta("64","shuffled").values()),
                       by_seed={s:mean(delta("64",part=[r for r in group if r["seed"]==s]).values()) for s in analysis["seeds"]})
        for metric,fun in (("oracle_over_distance",lambda a,d:a["64"]["oracle"]-d["64"]["U"]),
                           ("choice_loss",lambda a,d:a["64"]["loss"]),
                           ("distance_loss",lambda a,d:d["64"]["loss"])):
            metrics[metric]=mean(aggregate(group,fun,"eight").values())
        different=[r["arms"]["eight"]["64"] for r in group if r["arms"]["eight"]["64"]["distinct"]]
        metrics["better_terminal_chosen"]=[sum(r["correct"] for r in different),len(different)]
        scores[source]=metrics
    passed=all(min(v["by_seed"].values())>1.e-6 and 2*v["wins"]>len(v["geometries"]) for v in scores.values())
    passed=passed and all(mean(v["mean"]-v[k] for v in scores.values())>1.e-6 for k in ("original","shuffled"))
    OUT.mkdir(exist_ok=True)
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=scores,passed=passed,decisions=54),indent=2),encoding="utf8")
    (OUT/"decisions.json").write_text(json.dumps(rows,allow_nan=False),encoding="utf8")
    lines=["# 搜索 64 步后选择一条轨迹","",
        "各列为混合起点（距离、八视图热图）减双距离的几何等权净效益 U。上限允许在混合起点中事后挑选，其参照仍为实际的双距离选择。","",
        "| 来源 | 起点即选择 | 第 64 步选择 | 改善几何 | 供体 51291 | 供体 51292 | 事后上限 | 选对较优终点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for src in ("PDPCDPG-D100","XTW-p100","GH-p100"):
        v=scores[src];a,b=v["better_terminal_chosen"]
        lines.append(f"| {src} | {v['no_scout']:+.2f} | {v['mean']:+.2f} | {v['wins']}/{len(v['geometries'])} | "
                     +" | ".join(f"{x:+.2f}" for x in v["by_seed"].values())+f" | {v['oracle_over_distance']:+.2f} | {a}/{b} |")
    lines+=["","新开发种子复核门槛"+("通过。" if passed else "未通过。")+"第 64 步两条轨迹各试 64 步，择优的一条再走 192 步。选对比例以两条终点收益有差异的案例与种子方向计数。","",
            "[原门槛](../../双轨迹的中途选择.md) · [几何和选择遗憾](analysis.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
