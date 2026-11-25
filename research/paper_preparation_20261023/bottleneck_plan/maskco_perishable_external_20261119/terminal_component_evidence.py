"""Decompose paired archived terminal solutions into customer-disjoint exchanges."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean
import json
import math

from prepare_panel import core

HERE=Path(__file__).resolve().parent

def components(a,b):
    owners=[{v:j for j,r in enumerate(sol) for v in r} for sol in (a,b)]
    graph=defaultdict(set)
    for v in range(1,101):
        x,y=(0,owners[0][v]),(1,owners[1][v]);graph[x].add(y);graph[y].add(x)
    seen=set()
    for root in graph:
        if root in seen:continue
        stack=[root];parts=[[],[]]
        while stack:
            at=stack.pop()
            if at in seen:continue
            seen.add(at)
            side,j=at
            parts[side].append((a,b)[side][j])
            stack.extend(graph[at]-seen)
        yield parts

def read(p):return json.loads(p.read_text(encoding="utf8"))

def main():
    meta=read(HERE/"heldout_sources.json")
    plan=read(HERE/"replicate01.json");states=defaultdict(dict)
    prior=HERE/"results/replicate01"
    for i,block in enumerate(plan["blocks"]):
        out=read(prior/f"out/{i}.json")
        for arm in ("HYBRID_D","HYBRID"):
            r=out["arms"][arm]
            if r["status"]!="AUDITED":raise ValueError("Missing archive")
            states[block["case"]][arm,block["seed"]]=(tuple(map(tuple,r["routes"])),r["U"])
    source_geo=defaultdict(lambda:defaultdict(list))
    for case,records in states.items():
        nodes=read(HERE/meta["derived_instances"][case]["file"])["nodes"]
        dist=[[math.dist(a[1:3],b[1:3]) for b in nodes] for a in nodes]
        model=core.PublicationModel(dict(nodes=nodes,distance=dist),core.load_config(),"main_Product1",.8)
        @lru_cache(None)
        def price(route):
            val=model.route(route)
            if not val.feasible:raise ValueError(case)
            return val.net_benefit
        for rs,v in records.values():
            if abs(sum(map(price,rs))-v)>1.e-6:raise ValueError("Endpoint differs")
        parent=records["HYBRID_D",32001][0]
        for arm in ("HYBRID","HYBRID_D"):
            for seed in (32002,32003):
                donor=records[arm,seed][0]
                offer=too_wide=large=change_count=reorder=small=0.
                improvements=0
                for old,new in components(parent,donor):
                    if sorted(v for r in old for v in r)!=sorted(v for r in new for v in r):raise ValueError("Unpaired component")
                    diff=sum(map(price,new))-sum(map(price,old))
                    if diff<=1.e-7:continue
                    improvements+=1;offer+=diff
                    if sum(map(len,old))>40:too_wide+=diff
                    if len(old)>=4:large+=diff
                    if len(old)!=len(new):change_count+=diff
                    if len(old)==len(new)==1:reorder+=diff
                    if len(old)<=3 and sum(map(len,old))<=12:small+=diff
                source,geometry=(meta["cases"][case][key] for key in ("dataset","geometry"))
                source_geo[source][geometry].append(dict(offer=offer,above40=too_wide,four_routes=large,
                    change_count=change_count,reorder=reorder,small=small,count=improvements))
    grouped={s:{k:mean(mean(z[k] for z in samples) for samples in d.values())
                for k in ("offer","above40","four_routes","change_count","reorder","small","count")} for s,d in source_geo.items()}
    lines=["# 不同求解轨迹之间的剩余改进","","以旧开发案例的 32001 种子纯距离终态为同一起点，另取两枚种子的纯距离和热图终态作供体。每次沿客户归属连接两套路线，把能独立替换的整组分开。正差组的增益相加，并在各供体和几何上取平均。本表需要另一条完整搜索轨迹，只刻画已观察到的改进结构。","",
        "| 来源 | 可独立替换的正收益 U | 其中 ≥4 条原路线 U | 其中总客户 >40 U | 改变组内用车数 U | 同车重排 U | ≤3 原路线且 ≤12 客 U |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("PDPCDPG-D100","XTW-p100","GH-p100"):
        v=grouped[s];lines.append("| "+s+" | "+" | ".join(f"{v[k]:.2f}" for k in ("offer","four_routes","above40","change_count","reorder","small"))+" |")
    lines+=["","各结构列可重叠。整组超过 40 客表示两条完整终态之间的这个直接替换覆盖超过 40 客；逐步重构可以拆成多个动作。"]
    (HERE/"terminal_component_evidence.json").write_text(json.dumps(grouped,indent=2))
    (HERE/"终态改进结构.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
