"""Measure the remaining changes across three independent development distance solutions."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import audit_delivery
from mechanism_summary import source_means
from phase_plan import read
def neighbors(routes):
    return {i:tuple(sorted((a,b))) for r in routes for a,i,b in zip((0,)+r,r,r[1:]+(0,))}
def crossover(a,b,price):
    labels=[{} for _ in range(2)]
    for s,routes in enumerate((a,b)):
        for k,r in enumerate(routes):
            for i in r:labels[s][i]=(s,k)
    graph=defaultdict(set)
    for i in range(1,101):
        p,q=labels[0][i],labels[1][i]
        graph[p].add(q);graph[q].add(p)
    seen=set();chosen=[]
    for root in graph:
        if root in seen:continue
        stack=[root];sides=[[],[]]
        while stack:
            p=stack.pop()
            if p in seen:continue
            seen.add(p)
            side,k=p
            sides[side].append((a,b)[side][k])
            stack.extend(graph[p]-seen)
        chosen.extend(max(sides,key=lambda group:sum(price(r) for r in group)))
    if sorted(c for r in chosen for c in r)!=list(range(1,101)):raise ValueError("Inconsistent recombination")
    value=lambda tours:sum(price(r) for r in tours)
    return max(0.,value(chosen)-max(value(a),value(b)))
def main():
    old=PREV.parent/"maskco_cross_dataset_20261119"
    meta=read(old/"matrix.json");plan=read(PREV/"dev03.json");data=defaultdict(list)
    for i,b in enumerate(plan["blocks"]):
        rec=read(PREV/f"results/dev03/out/{i}.json")["arms"]["HYBRID_D"]
        data[b["case"]].append(tuple(tuple(r) for r in rec["routes"]))
    from sys import path
    path.insert(0,str(audit_delivery.ROOT/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"))
    import profit_core as core
    summary={}
    for c,records in data.items():
        if len(records)!=3:raise ValueError("Three development seeds required")
        model=audit_delivery.model_for(core,meta,c,False)
        @lru_cache(None)
        def price(r):return model.route(r).net_benefit
        out=[]
        for a,b in combinations(records,2):
            na,nb=neighbors(a),neighbors(b)
            d=sum(na[k]!=nb[k] for k in na)
            extra=crossover(a,b,price)
            out.append(dict(different_customers=d,cover_improvement_U=extra,cover_positive=int(extra>1e-6)))
        summary[c]={k:mean(r[k] for r in out) for k in out[0]}
    result=dict(cases=len(data),parent_pairs=3*len(data),sources=source_means(summary,meta["cases"]),by_case=summary)
    (HERE/"parent_disagreement.json").write_text(json.dumps(result,indent=2))
    print(result["sources"])
if __name__=="__main__":main()
