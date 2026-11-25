"""Close the remaining customers of each proposed multi-route exchange by exact subset partition."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations,permutations
from statistics import mean
import json
import time
import numpy as np
from route_columns import BASE,DEST,HERE,ARMS,matrices,assemble,read,core,independent_audit


def candidates(parts,mass,value):
    owner={i:k for k,r in enumerate(parts) for i in r}
    ids=sorted(owner);out=[]
    for size in (2,3):
        for group in combinations(ids,size):
            js=sorted({owner[i] for i in group})
            if len(js)<2 or sum(len(parts[k]) for k in js)-size>9 or sum(mass[i] for i in group)>20000.+1.e-7:continue
            profit,r=max((value(r),r) for r in permutations(group))
            if np.isfinite(profit):
                out.append((r,js,profit-sum(value((i,)) for i in r)))
    return out


def rank_order(v):
    a=np.array(v);vals,groups,counts=np.unique(a,return_inverse=True,return_counts=True)
    # Descending average position for equal values.
    return (len(a)-np.cumsum(counts)+.5*(counts-1))[groups]


def pick(options,scores):
    routes=[x[0] for x in options];eco=rank_order([x[2] for x in options])
    common=sorted(range(len(routes)),key=lambda j:(eco[j],routes[j]))[:4]
    pool=[j for j in range(len(routes)) if j not in common]
    chosen={}
    for m,score in scores.items():
        affinity=[float(np.mean(score[list((0,)+r),list(r+(0,))])) for r in routes]
        net=eco+rank_order(affinity)
        chosen[m]=common+sorted(pool,key=lambda j:(net[j],routes[j]))[:8]
    return chosen


def complete(parts,rec,value,best_order):
    seed,js,_=rec;taken=set(seed);others=[r for k,r in enumerate(parts) if k not in js]
    keep=[tuple(i for i in parts[k] if i not in taken) for k in js]
    keep=list(filter(None,keep));ids=sorted(i for r in keep for i in r);n=len(ids)
    if n>9:raise ValueError("Unregistered residual size")
    loc={i:k for k,i in enumerate(ids)}
    prices={}
    for size in range(1,min(n,4)+1):
        for items in combinations(ids,size):
            price,r=best_order(items)
            if np.isfinite(price):prices[sum(1<<loc[i] for i in items)]=(price,r)
    for r in keep:
        mask=sum(1<<loc[i] for i in r);old=prices.get(mask,(-np.inf,()))
        if value(r)>old[0]:prices[mask]=(value(r),r)
    by=[[] for i in ids]
    for bits,(u,r) in prices.items():
        for j in range(n):
            if bits&(1<<j):by[j].append((bits,u,r))
    @lru_cache(None)
    def best(left):
        if not left:return 0.,()
        j=(left&-left).bit_length()-1
        best_u=-np.inf;winner=()
        for subset,cost,r in by[j]:
            if subset&left==subset:
                tail,more=best(left^subset)
                if cost+tail>best_u:best_u=cost+tail;winner=(r,)+more
        return best_u,winner
    u,cover=best((1<<n)-1)
    direct=sum(map(value,keep))
    if u<direct-1.e-6:raise ValueError("DP lost original residual")
    return others+[seed]+list(cover),others+[seed]+keep


def main():
    ts=time.process_time();source=read(HERE/"heldout_sources.json");arch=read(HERE/"archived_states.json");folds=read(BASE/"fit.json")["folds"]
    groups=read(DEST/"regions.json");full=read(DEST/"full_opportunity.json")["rows"];data,_=assemble();rows=[]
    for c in sorted({r["case"] for r in groups}):
        records=[e for e in data if e["case"]==c];k=folds[c];fw=dict(np.load(BASE/f"readout_fold{k}.npz"));bw=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        nodes=np.array(read(HERE/source["derived_instances"][c]["file"])["nodes"]);m=nodes[:,3]
        dist=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
        model=core.PublicationModel(dict(nodes=nodes.tolist(),distance=dist.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(90000)
        def value(r):
            if sum(m[i] for i in r)>20000.+1.e-7:return -np.inf
            p=model.route(r);return p.net_benefit if p.feasible else -np.inf
        @lru_cache(60000)
        def best_order(group):
            if sum(m[i] for i in group)>20000.+1.e-7:return -np.inf,()
            return max((value(r),r) for r in permutations(group))
        for z in (q for q in groups if q["case"]==c):
            state=z["state"];rep=z["rep"];parent=list(map(tuple,arch[c][state]))
            ranks=matrices(records[(6 if state=="terminal" else 0)+rep],fw,bw,rep)
            regions={"neighbour":z["source_routes"]}
            if state=="terminal":
                regions["max_spare"]=next(q["parts"] for q in full if q["case"]==c and q["group"]=="max_spare" and q["rep"]==rep)
            for kind,raw in regions.items():
                parts=list(map(tuple,raw));outside=[r for r in parent if r not in parts]
                options=candidates(parts,m,value);choice=pick(options,ranks)
                if not options or len({len(v) for v in choice.values()})!=1:raise ValueError("Unequal path budget")
                joined={};direct={}
                for ix in set(j for ls in choice.values() for j in ls):
                    a,b=complete(parts,options[ix],value,best_order)
                    for pool,r in ((joined,a),(direct,b)):
                        sol=outside+r
                        u=independent_audit(model,[model.route(s) for s in sol],True)
                        if abs(sum(map(value,sol))-u)>1.e-6:raise ValueError("Independent route discrepancy")
                        pool[ix]=u-z["original_U"]
                def gains(v):return {a:max(0.,max(v[j] for j in idx)) for a,idx in choice.items()}
                rows.append(dict(case=c,rep=rep,state=state,group=kind,options=len(options),audits=2*len(joined),gains=gains(joined),direct=gains(direct)))
    if len(rows)!=30:raise ValueError("Incomplete complementary masks")
    lines=["# 提出一条路线后的剩余客户同步重分车","","每个区域各臂共享四条经济引导路线，再以相同名额提出八条；余下客户用子集动态规划求最优分车。表列各方法可保留的净效益增量 U。","",
           "| 起点与区域 | 距离 | 纯物理 | MASKCO | 置乱 | MASKCO 仅保留原残余路线 |","|---|---:|---:|---:|---:|---:|"]
    avg={}
    for state,kind,label in (("initial","neighbour","初解，邻近"),("terminal","neighbour","强终态，邻近"),("terminal","max_spare","强终态，最大余量")):
        sub=[r for r in rows if r["state"]==state and r["group"]==kind]
        avg[state,kind]={a:mean(r["gains"][a] for r in sub) for a in ARMS}
        lines.append("| "+label+" | "+" | ".join(f"{avg[state,kind][a]:.3f}" for a in ARMS)+f" | {mean(r['direct']['neural'] for r in sub):.3f} |")
    wins=sum(mean(r["gains"]["neural"]-r["gains"]["distance"] for r in rows if r["case"]==c and r["group"]=="neighbour")>1.e-7 for c in sorted({r["case"] for r in rows}))
    terminal=avg["terminal","neighbour"]
    passed=bool(terminal["neural"]>terminal["distance"]+1.e-7 and wins>=3 and all(mean(r["gains"]["neural"]-r["gains"][a] for r in rows)>1.e-7 for a in ("physical","shuffled")))
    lines+=["",f"{sum(r['audits'] for r in rows)} 次全车队独立复算。邻近区域两种起点合并后优于距离的几何为 {wins}/5，晋级门槛"+("通过。" if passed else "未通过。"),"","[方案与分叉规则](../../同步补全剩余客户.md)"]
    (DEST/"residual_partition.json").write_text(json.dumps(dict(rows=rows,passed=passed,wins=wins,cpu_s=time.process_time()-ts),indent=2))
    (DEST/"残余配车.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
