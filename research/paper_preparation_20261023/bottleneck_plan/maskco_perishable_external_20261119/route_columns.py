"""Repartition a capacity-linked group using complete routes proposed by masked customer edges."""
from collections import defaultdict
from functools import lru_cache
from statistics import mean
import json
import time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp,linprog
from scipy.sparse import csc_matrix
from directed_decoder_fit import HERE,assemble,design,read
from directed_decoder_features import normalized_logits
from customer_mixture import combine
from prepare_panel import core,independent_audit

BASE=HERE/"results/headcapture01"
ARMS=("distance","physical","neural","shuffled")
DEST=HERE/"results/columns01"


def select(routes,mask,dist,mass,rep):
    anchor=next(k for k,r in enumerate(routes) if mask[rep] in r)
    selected=[anchor];ids=list(routes[anchor])
    spare=lambda route:1-mass[list(route)].sum()/20000.
    slack=spare(routes[anchor])
    while len(selected)<5 or slack<1.-1.e-8:
        can=[k for k,r in enumerate(routes) if k not in selected and len(ids)+len(r)<=28]
        if not can:break
        k=min(can,key=lambda j:(dist[np.ix_(ids,routes[j])].min()/(.05+spare(routes[j])),j))
        selected.append(k);ids+=list(routes[k]);slack+=spare(routes[k])
    return selected,slack


def ranked(logits):
    z=logits.copy();np.fill_diagonal(z,-np.inf)
    return -np.log(1+np.argsort(np.argsort(-z,axis=1,kind="stable"),axis=1,kind="stable"))


def matrices(e,forward,backward,rep):
    scales=[forward[k] for k in ("mu","sd","eh","pca")]
    pred={};distance=e["dist"]
    for side,weights in (("out",forward),("in",backward)):
        raw={}
        for m in ("physical","neural"):
            x=design(e,scales,m=="neural")
            raw[m]=distance+(x@weights[m])@x.T
        p,h=raw["physical"],raw["neural"]
        pred[side,"physical"]=ranked(p)
        pred[side,"neural"]=ranked(combine(p,h))
        pred[side,"shuffled"]=ranked(combine(p,h,distance,57019+rep+1000*(side=="in")))
    active=np.zeros(101,bool);active[e["mask"]]=True
    out={"distance":np.where(active[:,None]&~active[None,:],distance,
        np.where(active[None,:]&~active[:,None],distance.T,.5*(distance+distance.T)))}
    for m in ARMS[1:]:
        n=active[:,None].astype(int)+active[None,:]
        total=pred["out",m]*active[:,None]+pred["in",m].T*active[None,:]
        out[m]=np.where(n,total/np.maximum(1,n),out["distance"])
    return out


def proposals(nodes,score,mass):
    paths=set()
    for start in sorted(nodes):
        beam=[((start,),mass[start],float(score[0,start]))]
        for count in (4,6,8):
            pool=[]
            for seq,load,x in beam:
                prev=seq[-1]
                for v in nodes:
                    if v not in seq and load+mass[v]<=20000.+1.e-7:
                        y=x+score[prev,v]
                        pool.append((-(y+score[v,0]),seq+(v,),load+mass[v],y))
            pool.sort();beam=[(seq,load,x) for _,seq,load,x in pool[:count]]
            paths.update(seq for seq,_,_ in beam)
    return paths


def partition(parts,columns,value,report_bound=False):
    ids=sorted(i for r in parts for i in r);ids_map={i:k for k,i in enumerate(ids)}
    keep={}
    for r in sorted(set(parts)|set((i,) for i in ids)|columns):
        price=value(r)
        if not np.isfinite(price):continue
        key=tuple(sorted(r))
        if key not in keep or price>keep[key][0]:keep[key]=(price,r)
    pool=list(keep.values());a=[];j=[]
    for k,(v,r) in enumerate(pool):
        a.extend(ids_map[i] for i in r);j.extend([k]*len(r))
    matrix=csc_matrix((np.ones(len(a)),(a,j)),shape=(len(ids),len(pool)))
    c=np.array([sum(value((i,)) for i in r)-v for v,r in pool])
    if report_bound:
        lp=linprog(c,A_eq=matrix,b_eq=np.ones(len(ids)),bounds=(0,None),method="highs")
        if not lp.success:raise ValueError("Route-pool relaxation infeasible")
        incumbent=sum(sum(value((i,)) for i in r)-value(r) for r in parts)
        rc=c-np.asarray(matrix.T@lp.eqlin.marginals).ravel()
        admissible=lp.fun+rc<=incumbent+1.e-5
        matrix=matrix[:,admissible];c=c[admissible]
        pool=[p for p,keep_col in zip(pool,admissible) if keep_col]
    sol=milp(c,integrality=np.ones(len(pool)),bounds=Bounds(0.,1.),
        constraints=LinearConstraint(matrix,1.,1.),options={"mip_rel_gap":0.,"time_limit":30.})
    if not sol.success and (not report_bound or sol.status!=1):raise RuntimeError(f"Route pool did not reach proven optimum: {sol.message}")
    winner=[r for x,(_,r) in zip(sol.x,pool) if x>.5] if sol.x is not None else list(parts)
    if sum(map(value,winner))<sum(map(value,parts)):winner=list(parts)
    if sorted(i for r in winner for i in r)!=ids:raise ValueError("Wrong customer coverage")
    if report_bound:
        lo=sol.mip_dual_bound if sol.mip_dual_bound is not None else lp.fun
        bound=sum(value((i,)) for i in ids)-lo
        return winner,len(keep),max(0.,bound-sum(map(value,winner)))
    return winner,len(pool),float(sol.mip_gap)


def main():
    start=time.process_time();rows,_=assemble();record=[];geom=read(HERE/"heldout_sources.json")
    folds=read(BASE/"fit.json")["folds"];archive=read(HERE/"archived_states.json")
    for c in sorted(k for k in folds if k.startswith("dk_")):
        examples=[e for e in rows if e["case"]==c]
        nodes=np.array(read(HERE/geom["derived_instances"][c]["file"])["nodes"])
        dist=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(nodes=nodes.tolist(),distance=dist.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=None)
        def value(r):
            a=model.route(r)
            return a.net_benefit if a.feasible else -np.inf
        k=folds[c];forward=dict(np.load(BASE/f"readout_fold{k}.npz"));back=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        for si,state in enumerate(("initial","terminal")):
            routes=list(map(tuple,archive[c][state]));initial=independent_audit(model,[model.route(r) for r in routes],True)
            for rep in (0,1):
                e=examples[si*6+rep];masks=matrices(e,forward,back,rep)
                js,slack=select(routes,e["mask"],dist,mass,rep)
                parts=[routes[j] for j in js];community=sorted(i for r in parts for i in r)
                outside=[r for j,r in enumerate(routes) if j not in js]
                q={}
                for arm in ARMS:
                    columns=proposals(community,masks[arm],mass)
                    best,size,gap=partition(parts,columns,value)
                    U=independent_audit(model,[model.route(r) for r in outside+best],True)
                    if U<initial-1.e-7 or abs(U-sum(value(r) for r in outside+best))>1.e-6:raise ValueError("Economic partition audit")
                    q[arm]=dict(gain=U-initial,proposals=len(columns),pool=size,gap=gap,fleet=len(best)-len(parts),chosen=best)
                record.append(dict(case=c,state=state,rep=rep,original_U=initial,spare_vehicle_equivalents=slack,source_routes=parts,customers=len(community),arms=q))
    if len(record)!=20:raise ValueError("Missing groups")
    DEST.mkdir(exist_ok=True,parents=True);(DEST/"regions.json").write_text(json.dumps(record,indent=2))
    means={s:{a:mean(z["arms"][a]["gain"] for z in record if z["state"]==s) for a in ARMS} for s in ("initial","terminal")}
    geo={c:{a:mean(z["arms"][a]["gain"] for z in record if z["case"]==c) for a in ARMS} for c in sorted({v["case"] for v in record})}
    wins=sum(v["neural"]>v["distance"]+1.e-7 for v in geo.values())
    passed=wins>=3 and all(m["neural"]>m["distance"]+1.e-7 for m in means.values()) and all(mean(v["neural"]-v[a] for v in geo.values())>1.e-7 for a in ("physical","shuffled"))
    lines=["# 按剩余容量重组丹麦配送路线","","20 个预定路线区域，各方法在自己的候选池内求得最优可行配车，表中为相对共同原解的平均净效益增量 U。","",
        "| 起点 | 距离 | 纯物理 | MASKCO | 置乱 | MASKCO 减距离 |","|---|---:|---:|---:|---:|---:|"]
    for s,m in means.items():lines.append("| "+s+" | "+" | ".join(f"{m[a]:.3f}" for a in ARMS)+f" | {m['neural']-m['distance']:+.3f} |")
    lines+=["",f"五个几何中有 {wins} 个的 MASKCO 增量高于距离。扩大到三个来源的门槛"+("通过。" if passed else "未通过。"),"","[候选和门槛](../../按剩余载重选择跨路线重构.md)"]
    (DEST/"analysis.json").write_text(json.dumps(dict(means=means,by_geometry=geo,audits=80,wins=wins,passed=passed,cpu_s=time.process_time()-start),indent=2))
    (DEST/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
