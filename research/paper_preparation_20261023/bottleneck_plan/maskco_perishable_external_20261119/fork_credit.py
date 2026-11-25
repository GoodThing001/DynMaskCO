"""Fit first-customer priority from independently audited downstream fleet profit."""
from collections import Counter,defaultdict
from functools import lru_cache
import json
from statistics import mean
import time
import numpy as np
from directed_decoder_fit import HERE,SOURCES,assemble,read
from customer_mixture import probabilities
from prepare_panel import core,independent_audit
from repair_priority import insert,positions,BASE

DEST=HERE/"results/forkcredit01"
ARMS=("distance","economic","physical","neural","shuffled","oracle")


def first_actions(base,group,mass,value,dist,dist_rank):
    load=np.array([sum(mass[v] for v in path) for path in base]+[0.])
    choices={}; economic=[];distance=[];routes=[];detail=[]
    for i in group:
        by=[]; best=None
        for j,path in enumerate([*base,()]):
            if load[j]+mass[i]>20000.+1.e-7:continue
            slots=[]
            for k in range(len(path)+1):
                rr=path[:k]+(i,)+path[k:]
                gain=value(rr)-value(path)
                if not np.isfinite(gain):continue
                a=path[k-1] if k else 0;b=path[k] if k<len(path) else 0
                rec=(gain,-j,-k,rr)
                if best is None or rec>best:best=rec
                slots.append((gain,dist_rank[i,a]+dist_rank[i,b],a,b))
            if slots:by.append((j,*(max(s[k] for s in slots) for k in (0,1))))
        if best is None:raise ValueError("Missing singleton")
        choices[i]=best;route=-best[1];routes.append(route)
        vals=np.array([r[1:] for r in by])
        g=np.sort(vals,axis=0)[::-1]
        gap=g[0]-g[1] if len(g)>1 else np.array([np.inf,np.inf])
        economic.append(gap[0]);distance.append(gap[1])
        top=np.max(vals[:,0])
        detail.append([np.log1p(min(10000.,gap[0])),min(12.,gap[1]),np.log1p(len(by)),
            np.log1p(max(0.,top-value((i,)))),sum(vals[:,0]>=top-1.)/len(by),sum(vals[:,0]>=top-10.)/len(by),
            load[route]/20000.,len(base[route])/20 if route<len(base) else 0.])
    x=[]
    for ix,(i,r) in enumerate(zip(group,routes)):
        others=[j for j,k in zip(group,routes) if k==r and j!=i]
        near=sorted(dist[i,j] for j in group if i!=j)
        x.append(detail[ix]+[len(others)/7.,sum(mass[j] for j in others)/20000.,
            sum(load[r]+mass[i]+mass[j]>20000.+1.e-7 for j in others)/7.,np.log1p(near[0]),np.log1p(near[3])])
    def best_idx(vals):return max(range(len(group)),key=lambda k:(vals[k],-group[k]))
    return choices,np.array(x),best_idx(economic),best_idx(distance)


def signals(logp,group,mask):
    scores=[]
    visible=np.setdiff1d(np.arange(1,101),mask)
    for i in group:
        rest=[j for j in group if j!=i]
        def pool(v):return [v.max(),v.mean(),np.sort(v)[-1]-np.sort(v)[-2]]
        scores.append([logp[i,0],logp[0,i],*pool(logp[i,rest]),*pool(logp[rest,i]),
                          *pool(logp[i,visible]),*pool(logp[visible,i])])
    return np.asarray(scores)


def collect():
    start=time.process_time();rows,_=assemble()
    reg=read(HERE/"heldout_sources.json");archive=read(HERE/"archived_states.json")
    grouped=defaultdict(list)
    for e in rows:grouped[e["case"]].append(e)
    records=[];ph=[];nh=[];emb=[];labels=[];audit_count=0;route_calls=0
    for c,examples in grouped.items():
        nodes=np.array(read(HERE/reg["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(nodes=nodes.tolist(),distance=d.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=120000)
        def value(route):
            if not route:return 0.
            v=model.route(route)
            return v.net_benefit if v.feasible else -np.inf
        for si,state in enumerate(("initial","terminal")):
            parent=[tuple(r) for r in archive[c][state]]
            orig=independent_audit(model,[model.route(r) for r in parent],True)
            for mi,e in enumerate(examples[6*si:6*si+6]):
                logits=e["h"]@e["h"].T*e["scale"];np.fill_diagonal(logits,-1.e4)
                lp=np.log(np.maximum(probabilities(logits),1.e-30));seen=set()
                for anchor in e["mask"][:4]:
                    group=sorted(map(int,e["mask"]),key=lambda i:(i!=anchor,d[i,anchor],i))[:8]
                    key=tuple(sorted(group))
                    if key in seen:continue
                    seen.add(key);group=list(key)
                    base=list(filter(None,(tuple(i for i in r if i not in group) for r in parent)))
                    chosen,attr,econ,spatial=first_actions(base,group,mass,value,d,e["dist"])
                    demand=sorted(group,key=lambda i:(-mass[i],i));gains=[]
                    for i in group:
                        rr=insert(base,chosen[i])
                        for j in demand:
                            if j!=i:rr=insert(rr,positions(rr,j,value,mass))
                        u=independent_audit(model,[model.route(r) for r in rr],True)
                        if abs(sum(map(value,rr))-u)>1.e-6:raise ValueError("Fleet benefit differs")
                        gains.append(max(0.,u-orig));audit_count+=1
                    ph.append(np.column_stack((attr,e["phys"][group])))
                    nh.append(signals(lp,group,e["mask"]));emb.append(e["h"][group]);labels.append(gains)
                    records.append(dict(case=c,src=e["src"],geometry=e["geo"],state=state,mask=mi,anchor=int(anchor),
                                        customers=group,economic=econ,distance=spatial))
        route_calls+=value.cache_info().hits+value.cache_info().misses;value.cache_clear()
    if len(records)*8!=audit_count or len(grouped)!=27:raise ValueError("Branch count")
    DEST.mkdir(exist_ok=True,parents=True)
    (DEST/"contexts.json").write_text(json.dumps(dict(contexts=records,audits=audit_count,requests=route_calls,cpu_s=time.process_time()-start)))
    np.savez_compressed(DEST/"choices.npz",physical=np.array(ph,np.float32),neural=np.array(nh,np.float32),
        embeddings=np.array(emb,np.float32),U=np.array(labels,np.float64))


def block_weights(rows):
    # Average contexts within each case, then cases within geometry and source.
    geos={s:{r["geometry"] for r in rows if r["src"]==s} for s in SOURCES}
    cases=defaultdict(set);count=Counter(r["case"] for r in rows)
    for r in rows:cases[r["geometry"]].add(r["case"])
    return np.array([1/(len(SOURCES)*len(geos[r["src"]])*len(cases[r["geometry"]])*count[r["case"]]) for r in rows])


def fit():
    start=time.process_time();archive=read(DEST/"contexts.json");rows=archive["contexts"]
    X=dict(np.load(DEST/"choices.npz"));U=X["U"];ph=X["physical"];nh=X["neural"];h=X["embeddings"]
    n=len(rows);folds=read(BASE/"fit.json")["folds"]
    selection={a:np.empty(n,dtype=int) for a in ARMS}
    for arm in ("distance","economic"):selection[arm][:]=[r[arm] for r in rows]
    selection["oracle"][:]=U.argmax(axis=1)
    # Retained gains often tie at zero, so use a common tolerance and midpoint ranks.
    rounded=U.round(6)
    diff=rounded[:,:,None]-rounded[:,None,:]
    y=((diff>0).sum(axis=2)+.5*(diff==0).sum(axis=2)-4.)/7.
    rng=np.random.default_rng(606012)
    perm=np.array([rng.permutation(8) for r in rows])
    for k in range(5):
        tr=np.array([folds[r["case"]]!=k for r in rows])
        w=block_weights([r for j,r in enumerate(rows) if tr[j]])
        mu=np.average(h[tr].mean(axis=1),axis=0,weights=w)
        hp=(h[tr]-mu)*np.sqrt(w/8)[:,None,None]; flat=hp.reshape(-1,h.shape[-1])
        eig,vec=np.linalg.eigh(flat.T@flat)
        pc=(h-mu)@vec[:,-8:]/np.sqrt(np.maximum(eig[-8:],1.e-5))
        signals=np.concatenate([nh,pc],axis=2)
        for arm in ("physical","neural","shuffled"):
            extra=signals if arm=="neural" else signals[np.arange(n)[:,None],perm]
            x=ph.copy() if arm=="physical" else np.concatenate([ph,extra],axis=2)
            x=x-x.mean(axis=1,keepdims=True)
            sd=np.sqrt(np.average((x[tr]**2).mean(axis=1),axis=0,weights=w))
            x=np.clip(x/np.maximum(sd,.005),-6,6)
            x=x-x.mean(axis=1,keepdims=True)
            design=(x[tr]*np.sqrt(w/8)[:,None,None]).reshape(-1,x.shape[-1])
            target=(y[tr]*np.sqrt(w/8)[:,None]).ravel()
            coef=np.linalg.solve(design.T@design+.05*np.eye(x.shape[-1]),design.T@target)
            selection[arm][~tr]=(x[~tr]@coef).argmax(axis=1)
    effects={a:U[np.arange(n),v] for a,v in selection.items()}
    src={};passed=True;win_sources=0
    lines=["# 以完整重构收益训练掩码客户决策","","每个数是在留出几何上执行一个首客决策后，与父解相比能保留的真实净效益 U。上界为遍历同组八种首客后的最大值。","",
      "| 来源 | 距离 | 经济 regret-2 | 学习物理 | 学习 MASKCO | 学习置乱 | 八选一上界 |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in SOURCES:
        gs={r["geometry"] for r in rows if r["src"]==s};per={}
        for g in gs:
            cases={r["case"] for r in rows if r["geometry"]==g}
            per[g]={a:mean(np.mean([u[j] for j,r in enumerate(rows) if r["case"]==c]) for c in cases) for a,u in effects.items()}
        av={a:mean(v[a] for v in per.values()) for a in ARMS}
        wins=sum(int(v["neural"]>v["physical"]+1.e-7) for v in per.values())
        win_sources+=int(2*wins>len(gs));passed &= av["neural"]>av["distance"]+1.e-7
        src[s]=dict(means=av,by_geometry=per,wins_vs_physical=wins)
        lines.append("| "+s+" | "+" | ".join(f"{av[a]:.3f}" for a in ARMS)+" |")
    passed=bool(passed and win_sources>=2 and all(mean(v["means"]["neural"]-v["means"][a] for v in src.values())>1.e-7 for a in ("physical","shuffled","economic")))
    np.savez_compressed(DEST/"out_of_fold_choices.npz",**selection)
    lines+=["",f"{len(rows):,} 组、{archive['audits']:,} 个完整车队终态经过独立审计；进入限时求解开发的门槛"+("通过。" if passed else "未通过。"),"","[配对规则与门槛](../../用真实分叉收益训练掩码首客.md)"]
    (DEST/"analysis.json").write_text(json.dumps(dict(sources=src,passed=passed,contexts=n,audits=archive["audits"],requests=archive["requests"],
        data_cpu_s=archive["cpu_s"],fit_cpu_s=time.process_time()-start),indent=2))
    (DEST/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":
    import sys
    if sys.argv[1]=="collect":collect()
    else:fit()
