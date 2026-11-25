"""Reconstruct a larger route region with three overlapping, ordinary MaskCO forwards."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from heat_placebo import remap

HERE=Path(__file__).resolve().parent
ARMS=("distance","real","shuffled","single")


def select_region(routes,dist,anchor):
    ids=sorted(range(len(routes)),key=lambda k:(k!=anchor,float(dist[np.ix_(routes[anchor],routes[k])].min()),k))
    picked=[];size=0
    for k in ids:
        if size+len(routes[k])<=60:
            picked.append(k);size+=len(routes[k])
    if not 40<size<=60:raise ValueError("A complete route region of 41–60 customers is required")
    return picked


def affinity(h):
    x=np.array(h,copy=True)
    np.fill_diagonal(x,-np.inf)
    ranks=1+np.argsort(np.argsort(-x,axis=1,kind="stable"),axis=1,kind="stable")
    scores=-np.log(ranks)
    return (scores+scores.T)/2.


def tile_edges(task,routes,ids,provider,rng):
    dist=np.asarray(task.ins["distance"])
    region=np.asarray(sorted(v for k in ids for v in routes[k]),dtype=int)
    pieces=np.array_split(rng.permutation(region),3)
    rest=np.setdiff1d(np.arange(1,101),region)
    distance=affinity(-dist)
    accum=np.zeros((2,101,101));counts=np.zeros((101,101),dtype=int)
    single=distance.copy()
    for index in range(3):
        pair=np.r_[pieces[index],pieces[(index+1)%3]]
        ids40=rng.permutation(np.r_[pair,rng.choice(rest,40-len(pair),replace=False)])
        if len(set(ids40))!=40:raise ValueError("Incomplete model mask")
        h=provider.heatmap(task,routes,ids40,"maskco_base",index)
        aligned,shuffled=remap(h,dist,routes,ids40,rng)
        where=np.ix_(pair,pair)
        actual=affinity(aligned)[where];null=affinity(shuffled)[where]
        accum[0][where]+=actual;accum[1][where]+=null;counts[where]+=1
        if index==0:single[where]=actual
    inside=counts[np.ix_(region,region)]
    if inside.min()!=1 or inside.max()!=2:raise ValueError("Co-masked edge coverage differs")
    real,shuffled=accum/np.maximum(counts,1)[None,:,:]
    return (distance,real,shuffled,single),int((inside>0).sum())


def search(fn,inputs,parts,ranking,initial_price,seed,limit):
    from block_split import split
    rng=np.random.default_rng(seed)
    ids=sorted(i for r in parts for i in r)
    preference={i:sorted((j for j in ids if j!=i),key=lambda j:(-ranking[i,j],j)) for i in ids}
    best=parts;value=initial_price;offers=0;calls=0
    for t in range(limit):
        # Keep full incumbent route directions. Randomizing route junctions is free for its partition.
        route_keys=rng.random(101)
        seq=[i for route in sorted(best,key=lambda r:route_keys[min(r)]) for i in route]
        for move in range(1+t%3):
            i=ids[int(rng.integers(len(ids)))];u=seq.index(i)
            valid=[j for j in preference[i] if abs(u-seq.index(j))>1]
            if not valid:continue
            draw,side,op=rng.random(3)
            j=valid[int(draw**2*min(12,len(valid)))];v=seq.index(j)
            if op<1./3.:
                seq[u],seq[v]=seq[v],seq[u]
            elif op<2./3.:
                seq.pop(u);seq.insert(seq.index(j)+int(side>=.5),i)
            else:
                a,b=sorted((u,v))
                seq[a+1:b+1]=reversed(seq[a+1:b+1])
        v,new,work=split(fn,inputs,seq);calls+=work
        if v>value+1.e-7:best,value,offers=new,v,offers+1
    return best,value,offers,calls


def main(ix):
    import external_common as harness
    import pool_search as pool
    from block_split import library
    cfg=json.loads((HERE/"phase.json").read_text());case=cfg["blocks"][ix]["case"]
    archive=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=harness.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("overlapping-region-development",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    rng=np.random.default_rng(4097001);fn=library();start=time.process_time();out=[]
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");inputs=pool.inputs(task);dist=inputs[1]
        provider=locked.RepairProvider("overlap-masks",policy,False)
        for state in ("initial","terminal"):
            routes=tuple(map(tuple,archive[state]))
            if state=="initial" and routes!=tuple(map(tuple,task.initial())):raise ValueError("Archived initial solution differs")
            full=task.evaluate(routes,audit=True)["net_benefit_U"]
            anchors=rng.choice(len(routes),cfg["regions_per_state"],replace=False)
            for z,anchor in enumerate(anchors):
                region=select_region(routes,dist,int(anchor))
                parts=tuple(routes[k] for k in region);count=sum(map(len,parts))
                base=sum(task.route(r).net_benefit for r in parts)
                ranks,ncovered=tile_edges(task,routes,region,provider,rng)
                outcomes={}
                for arm,weights in zip(ARMS,ranks):
                    chosen,v,offers,requests=search(fn,inputs,parts,weights,base,6301903+z,cfg["orders"])
                    combined=tuple(r for j,r in enumerate(routes) if j not in region)+tuple(chosen)
                    actual=task.evaluate(combined,audit=True)["net_benefit_U"]-full
                    if abs(v-base-actual)>1.e-6 or actual < -1.e-6:
                        raise ValueError("Regional partition differs from full economic audit")
                    outcomes[arm]=dict(gain=max(0.,actual),fleet_delta=len(chosen)-len(parts),offers=offers,requests=requests,
                                       orders=cfg["orders"])
                out.append(dict(state=state,region=z,customers=count,routes=len(parts),covered=ncovered,arms=outcomes))
    (HERE/"out"/f"{ix}.json").write_text(json.dumps(dict(case=case,rows=out,forwards=provider.forward_calls,
                        cpu_s=time.process_time()-start),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
