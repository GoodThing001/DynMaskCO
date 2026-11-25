"""Combine audited prospective panels at equal source weight without additional searches."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OURS = "ORDER_SPLIT"
MATCH = "ORDER_SPLIT_D"
EPS = 1e-6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    old = read(HERE/"results/prospective01/analysis.json")
    retail = read(HERE/"results/retail01/analysis.json")
    if old["status"]!="COMPLETE" or retail["status"]!="COMPLETE":
        raise ValueError("Both prospective trials must be complete")
    if old["methods"]!=retail["methods"] or old["seeds"]!=retail["seeds"]:
        raise ValueError("Trials used different arms or paired seeds")
    primary = [(name, old["panels"][name]) for name in old["primary_sources"]]
    primary += [(name,retail["panels"][name]) for name in retail["primary_sources"]]
    if len(primary)!=3 or len({s for s,_ in primary})!=3:
        raise ValueError("Expected three independent benchmark sources")
    results = {}
    for name,p in primary:
        heat=p["contrasts"][MATCH]
        best=p["strongest_external"]
        v=p["contrasts"][best]
        positive_seeds=sum(x>EPS for x in heat["by_seed"].values())
        positive_each=(heat["mean"]>EPS and 2*heat["wins"]>p["geometries"] and positive_seeds>=2)
        results[name]=dict(rank13=p["external_plus_candidate_ranks"][OURS],rank16=p["all_ranks"][OURS],
                           units=p["geometries"],strongest_external=best,delta_strongest=v["mean"],
                           wins_strongest=v["wins"],heat=heat["mean"],wins_heat=heat["wins"],
                           positive_seeds=positive_seeds,stable_heat=positive_each)
    rng=np.random.default_rng(20261120)
    samplers={}
    for name,p in primary:
        units=sorted(p["contrasts"][MATCH]["by_geometry"])
        samplers[name]=(units,rng.integers(len(units),size=(20000,len(units))))
    overall={}
    for method in old["methods"]:
        if method==OURS:
            continue
        draws=[]
        means=[]
        for name,p in primary:
            units,ix=samplers[name]
            paired=p["contrasts"][method]
            unit_values=np.array([paired["by_geometry"][g] for g in units])
            draws.append(unit_values[ix].mean(axis=1))
            means.append(paired["mean"])
        interval=np.quantile(np.mean(draws,axis=0),(.025,.975))
        overall[method]=dict(mean_U=float(np.mean(means)),interval_U=interval.tolist(),
                             by_seed={str(s):float(np.mean([p["contrasts"][method]["by_seed"][str(s)] for _,p in primary]))
                                      for s in old["seeds"]})
    verdict=dict(all_top3_external=all(z["rank13"]<=3 for z in results.values()),
                 all_top3_including_internal=all(z["rank16"]<=3 for z in results.values()),
                 beats_all_external_in_every_source=all(z["delta_strongest"]>EPS for z in results.values()),
                 stable_matched_heat_in_all_sources=all(z["stable_heat"] for z in results.values()))
    out=HERE/"results"
    (out/"three_sources.json").write_text(json.dumps(dict(primary=results,equal_source=overall,criteria=verdict),
                                                     indent=2,ensure_ascii=False), encoding="utf-8")
    lines=["# 三个前瞻来源的统一判断","",
           "| 来源 | 名次 /13 | 名次 /16 | 最强外部法 | 对最强 ΔU | 胜它的几何/日/城 | 同规则热图 ΔU | 热图胜几何/日/城 | 热图胜种子 | 稳定门槛 |",
           "|---|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for name,z in results.items():
        lines.append(f"| {name} | {z['rank13']} | {z['rank16']} | {z['strongest_external']} | {z['delta_strongest']:+,.2f} | "
                     f"{z['wins_strongest']}/{z['units']} | {z['heat']:+,.2f} | {z['wins_heat']}/{z['units']} | {z['positive_seeds']}/3 | "
                     +("通过" if z["stable_heat"] else "未通过")+" |")
    lines+=["","| 三来源等权的对照 | MASKCO 减对照 U | 95% 描述区间 U | 三个种子 ΔU |",
            "|---|---:|---|---|"]
    for m,z in overall.items():
        lo,hi=z["interval_U"]
        ss=" / ".join(f"{u:+,.2f}" for u in z["by_seed"].values())
        lines.append(f"| {m} | {z['mean_U']:+,.2f} | [{lo:+,.2f}, {hi:+,.2f}] | {ss} |")
    lines+=["","区间按来源内整组重采样后给三个来源等权。最强外部法在各来源内取实测均值最高者。[判定规则](../三来源前瞻判定.md) · [机器可读结果](three_sources.json)"]
    (out/"三来源前瞻结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(verdict)


if __name__=="__main__":
    main()
