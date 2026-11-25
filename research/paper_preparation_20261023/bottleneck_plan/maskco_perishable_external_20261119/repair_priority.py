"""Evaluate every first customer under identical, exact economic insertions for the remaining mask."""
from collections import defaultdict
from functools import lru_cache
import json
from statistics import mean
import time
import numpy as np
from directed_decoder_fit import HERE,SOURCES,assemble,design,read
from directed_decoder_features import normalized_logits
from customer_mixture import combine
from prepare_panel import core,independent_audit

BASE=HERE/"results/headcapture01"
ARMS=("demand","economic","distance","physical","neural","shuffled","oracle")


def positions(routes,i,value,mass,weights=None):
    by_route=[];best=None
    for j,path in enumerate([*routes,()]):
        if mass[i]+sum(mass[v] for v in path)>20000.+1.e-7:continue
        options=[]
        for k in range(len(path)+1):
            rr=path[:k]+(i,)+path[k:]
            price=value(rr)
            if not np.isfinite(price):continue
            benefit=price-value(path)
            choice=(benefit,-j,-k,rr)
            if best is None or choice>best:best=choice
            if weights is not None:
                a=path[k-1] if k else 0
                b=path[k] if k<len(path) else 0
                options.append(np.r_[benefit,weights[0][:,i,a]+weights[1][:,i,b]])
        if options:by_route.append(np.max(options,axis=0))
    if best is None:raise ValueError("Singleton infeasible")
    if weights is None:return best
    matrix=np.array(by_route)
    # A single feasible receiving route means that all methods give this customer first priority.
    regret=np.full(5,np.inf) if len(matrix)==1 else np.sort(matrix,axis=0)[-1]-np.sort(matrix,axis=0)[-2]
    return best,regret


def insert(base,rec):
    _,j,k,path=rec;j=-j;result=list(base)
    if j==len(base):result.append(path)
    else:result[j]=path
    return result


def score_weights(row,models,stats,ix):
    out=[]
    for m in ("prev","next"):
        scores={}
        for arm in ("physical","neural"):
            x=design(row,stats,arm=="neural")
            scores[arm]=row["dist"]+(x@models[m][arm])@x.T
        p,h=scores["physical"],scores["neural"]
        out.append(np.stack((row["dist"],normalized_logits(p),combine(p,h),combine(p,h,row["dist"],60181+ix+1000*(m=="prev")))))
    return out


def main():
    start=time.process_time(); rows,_=assemble()
    reg=read(HERE/"heldout_sources.json"); archive=read(HERE/"archived_states.json")
    folds=read(BASE/"fit.json")["folds"]
    by_case=defaultdict(list)
    for ix,e in enumerate(rows):by_case[e["case"]].append((ix,e))
    contexts=[]; route_calls=audits=0
    for c,examples in by_case.items():
        nodes=read(HERE/reg["derived_instances"][c]["file"])["nodes"]
        xy=np.array(nodes)[:,1:3];d=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
        mass=np.array(nodes)[:,3];model=core.PublicationModel(dict(nodes=nodes,distance=d.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=100000)
        def value(route):
            if not route:return 0.
            result=model.route(route)
            return result.net_benefit if result.feasible else -np.inf
        k=folds[c];f=dict(np.load(BASE/f"readout_fold{k}.npz"));b=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        stats=[f[n] for n in ("mu","sd","eh","pca")];models={"prev":b,"next":f}
        for state_ix,state in enumerate(("initial","terminal")):
            parent=[tuple(r) for r in archive[c][state]]
            original=independent_audit(model,[model.route(r) for r in parent],True)
            for offset in (0,1,2):
                ix,e=examples[state_ix*6+offset]
                anchor=e["mask"][0]
                removed=sorted(map(int,e["mask"]),key=lambda i:(i!=anchor,d[i,anchor],i))[:8]
                base=[tuple(i for i in r if i not in removed) for r in parent]
                base=list(filter(None,base));weights=score_weights(e,models,stats,ix)
                demand=sorted(removed,key=lambda i:(-mass[i],i))
                first={};priority={}
                for i in removed:first[i],priority[i]=positions(base,i,value,mass,weights)
                selected={"demand":demand[0]}
                for j,arm in enumerate(("economic","distance","physical","neural","shuffled")):
                    selected[arm]=max(removed,key=lambda i:(priority[i][j],-i))
                gains={};terminal={}
                for i in removed:
                    rr=insert(base,first[i])
                    for j in demand:
                        if j!=i:rr=insert(rr,positions(rr,j,value,mass))
                    U=independent_audit(model,[model.route(r) for r in rr],True)
                    if abs(sum(map(value,rr))-U)>1.e-6:raise ValueError("Economic score mismatch")
                    audits+=1;gains[i]=U-original;terminal[i]=rr
                selected["oracle"]=max(removed,key=lambda i:gains[i])
                contexts.append(dict(case=c,geometry=e["geo"],source=e["src"],state=state,mask=offset,
                     removed=removed,first=selected,original_U=original,raw_gains=gains,final_routes=terminal))
        route_calls+=value.cache_info().hits+value.cache_info().misses;value.cache_clear()
    if len(contexts)!=162 or audits!=1296:raise ValueError("Incomplete first-customer fork")
    dest=HERE/"results/repairpriority01";dest.mkdir(exist_ok=True)
    (dest/"paired_forks.json").write_text(json.dumps(contexts,separators=(",",":")))
    src={}; passed=True;majorities=0
    lines=["# 多客户掩码的重构顺序","","每个起点和掩码都尝试八个可能的首客，后续使用相同的精确经济插位。表中为相对原解能保留的正净效益 U，三次掩码、两个起点和同几何案例依次取均值。","",
           "| 来源 | 载重优先 | 经济 regret-2 | 距离 | 纯物理 | MASKCO | 置乱 | 八选一上界 |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in SOURCES:
        geos=sorted({r["geometry"] for r in contexts if r["source"]==s})
        grouped={g:{m:mean(max(0.,r["raw_gains"][r["first"][m]]) for r in contexts if r["source"]==s and r["geometry"]==g) for m in ARMS} for g in geos}
        stats={m:mean(v[m] for v in grouped.values()) for m in ARMS}
        wins=sum(v["neural"]>v["distance"]+1.e-7 for v in grouped.values())
        passed &= stats["neural"]>stats["distance"]+1.e-7 and stats["oracle"]>stats["economic"]+1.e-7
        majorities+=int(wins*2>len(geos))
        src[s]=dict(means=stats,geometries=grouped,wins=wins)
        lines.append("| "+s+" | "+" | ".join(f"{stats[a]:.3f}" for a in ARMS)+" |")
    passed=bool(passed and majorities>=2 and all(mean(src[s]["means"]["neural"]-src[s]["means"][a] for s in SOURCES)>1.e-7 for a in ("physical","shuffled","economic")))
    cpu=time.process_time()-start
    lines+=["",f"162 个遮蔽上下文、1,296 个完整车队终态独立复算，{route_calls:,} 次含缓存路线请求，CPU {cpu:.1f} 秒。进入计时开发的门槛"+("通过。" if passed else "未通过。"),"","[选客规则与门槛](../../掩码客户重构顺序.md)"]
    (dest/"analysis.json").write_text(json.dumps(dict(sources=src,advance=passed,audits=audits,cpu_s=cpu,route_requests=route_calls),indent=2))
    (dest/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))

if __name__=="__main__":main()
