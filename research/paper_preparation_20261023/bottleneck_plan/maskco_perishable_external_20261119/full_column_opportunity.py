"""Separate regional capacity shortfalls from loss of profitable complete-route columns."""
from functools import lru_cache
import json
from statistics import mean
import time
import numpy as np
from route_columns import BASE,DEST,HERE,ARMS,matrices,proposals,partition,assemble,read,core,independent_audit


def spare_group(routes,anchor,dist,mass):
    a=routes[anchor];budget=28-len(a)
    dp={0:(0.,0.,())}
    for j,r in enumerate(routes):
        if j==anchor:continue
        space=20000.-sum(mass[i] for i in r);travel=dist[np.ix_(a,r)].min()
        for n,(slack,negative_km,prior) in sorted(dp.copy().items(),reverse=True):
            count=n+len(r)
            if count>budget:continue
            candidate=(slack+space,negative_km-travel,prior+(j,))
            if count not in dp or candidate>dp[count]:dp[count]=candidate
    count=max(dp,key=lambda n:(round(dp[n][0],6),-n,dp[n][1],dp[n][2]))
    return [a]+[routes[j] for j in dp[count][2]]


def feasible_columns(clients,mass,value):
    kept={};evaluated=0
    def visit(seq,used):
        nonlocal evaluated
        for i in clients:
            if i in seq or used+mass[i]>20000.+1.e-7:continue
            r=seq+(i,);price=value(r);evaluated+=1
            if np.isfinite(price):
                key=tuple(sorted(r))
                if key not in kept or price>kept[key][0]:kept[key]=price,r
            if len(r)<4:visit(r,used+mass[i])
    visit((),0.)
    return {d[1] for d in kept.values()},evaluated


def main():
    start=time.process_time();meta=read(HERE/"heldout_sources.json");archives=read(HERE/"archived_states.json")
    original=read(DEST/"regions.json");old=[r for r in original if r["state"]=="terminal"]
    contexts,_=assemble();folds=read(BASE/"fit.json")["folds"];records=[]
    for c in sorted({r["case"] for r in old}):
        ex=[e for e in contexts if e["case"]==c][6:]
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        dist=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(nodes=nodes.tolist(),distance=dist.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=100000)
        def value(r):
            price=model.route(r);return price.net_benefit if price.feasible else -np.inf
        routes=list(map(tuple,archives[c]["terminal"]));original_U=independent_audit(model,[model.route(r) for r in routes],True)
        k=folds[c];forward=dict(np.load(BASE/f"readout_fold{k}.npz"));back=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        for r in (r for r in old if r["case"]==c):
            rep=r["rep"];e=ex[rep];rank=matrices(e,forward,back,rep);a=next(j for j,path in enumerate(routes) if e["mask"][rep] in path)
            for kind,parts in (("neighbour",list(map(tuple,r["source_routes"]))),("max_spare",spare_group(routes,a,dist,mass))):
                ids=sorted(i for rr in parts for i in rr)
                cols,n=feasible_columns(ids,mass,value)
                winner,ncols,gap=partition(parts,cols,value,report_bound=True)
                outside=[rr for rr in routes if rr not in parts]
                u=independent_audit(model,[model.route(rr) for rr in outside+winner],True)
                if abs(sum(value(rr) for rr in outside+winner)-u)>1.e-6:raise ValueError("Full-cover audit")
                new=[rr for rr in winner if rr not in parts];short={}; captured={}
                for arm in ARMS:
                    shortlist=proposals(ids,rank[arm],mass)
                    if kind=="neighbour":
                        v=r["arms"][arm]["gain"]
                    else:
                        selected,_,_=partition(parts,shortlist,value)
                        v=independent_audit(model,[model.route(rr) for rr in outside+selected],True)-original_U
                    if v>u-original_U+gap+1.e-6:raise ValueError("Shortlist beats the full-pool bound")
                    short[arm]=v
                    # Equal net return counts even if a tied direction differs.
                    price={tuple(sorted(rr)):value(rr) for rr in new}
                    captured[arm]=len({tuple(sorted(rr)) for rr in shortlist|set(parts)|{(i,) for i in ids}
                        if tuple(sorted(rr)) in price and abs(price[tuple(sorted(rr))]-value(rr))<=1.e-6})
                records.append(dict(case=c,rep=rep,group=kind,customers=len(ids),parts=parts,spare=len(parts)-sum(mass[ids])/20000.,
                    candidates=n,priced_customer_sets=ncols,gain=u-original_U,upper_gain=u-original_U+gap,fleet_delta=len(winner)-len(parts),optimal=gap<=1.e-6,
                    winner=winner,new_routes=len(new),covered_new=captured,shortlist=short))
    if len(records)!=20:raise ValueError("Expected ten matched region pairs")
    (DEST/"full_opportunity.json").write_text(json.dumps(dict(rows=records,cpu_s=time.process_time()-start),indent=2))
    lines=["# 强距离终态上完整路线池的可达收益","","五个丹麦几何，每个用两个既有掩码。只替换选定区域，遍历其中所有至多四客的有向可行路线，另保留每条原路线。表内收益为相对共同原解的净效益 U。","",
        "| 28 客区域 | 空余容量够少一车 | 完整池严格改善 | 实际少用车 | 完整池可行均值 U | 均值上界 U | 证实最优 | 距离短名单 U | 热图短名单 U |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,label in (("neighbour","兼顾地理邻近"),("max_spare","最大化空余容量")):
        v=[r for r in records if r["group"]==name]
        lines.append(f"| {label} | {sum(r['spare']>=1-1.e-8 for r in v)}/10 | {sum(r['gain']>1.e-6 for r in v)}/10 | {sum(r['fleet_delta']<0 for r in v)}/10 | {mean(r['gain'] for r in v):.3f} | {mean(r['upper_gain'] for r in v):.3f} | {sum(r['optimal'] for r in v)}/10 | {mean(r['shortlist']['distance'] for r in v):.3f} | {mean(r['shortlist']['neural'] for r in v):.3f} |")
    lines+=["","[成对区域与候选边界](../../容量区域的完整候选对照.md)"]
    (DEST/"完整路线池.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__=="__main__":main()
