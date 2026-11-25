"""Compare insertion and route-order reconstruction from identical archived route states."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import external_common as adapter
import pool_search as pool
from block_split import bundle, library as split_lib, proposals, split
from calibration import residual_edges
HERE=Path(__file__).resolve().parent
def read(name):
    return json.loads((HERE/name).read_text())

def insertion(task,lib,inputs,routes,group,weights,seed,regret,count):
    state=pool.pack(routes)
    best=state.copy()
    stats=np.zeros(3,np.int64)
    lib.repair_batch(*inputs,weights,group,len(group),state,best,seed,20.,1.5,0,regret,count,stats)
    if stats[0]!=count:
        raise ValueError("Incomplete fixed-count insertion")
    improved=pool.unpack(best)
    score=task.evaluate(improved,audit=True)["net_benefit_U"]
    return score,int(stats[2]),improved

def order_fleet(task,fn,inputs,routes,ids,weights,seed,count=64):
    parts=[routes[k] for k in ids]
    old=sum(task.route(r).net_benefit for r in parts)
    best,winner,requests=old,parts,0
    for seq in proposals(parts,weights,seed,count):
        val,rs,work=split(fn,inputs,seq);requests+=work
        if val>best:best,winner=val,rs
    full=tuple(r for k,r in enumerate(routes) if k not in ids)+tuple(winner)
    u=task.evaluate(full,audit=True)["net_benefit_U"]
    if abs(best-sum(task.route(r).net_benefit for r in winner))>1.e-6:
        raise ValueError("Split benefit differs from independently scheduled routes")
    return u,requests,full

def orders(task,fn,inputs,routes,ids,weights,seed,count=64):
    u,requests,_=order_fleet(task,fn,inputs,routes,ids,weights,seed,count)
    return u,requests

def main(i):
    plan=read("phase.json");case=plan["blocks"][i]["case"];n_groups=plan["groups_per_state"]
    if case not in read("archived_states.json"):raise ValueError("Unregistered archived case")
    start=time.perf_counter();cpu=time.process_time()
    manifest=dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D")
    _,_,access,core,source,locked=adapter.install("ORDER_SPLIT",manifest,False)
    from timed_search import single_cpu_guard
    core_id=single_cpu_guard()
    policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("retrospective-operator-analysis",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    out=[]
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test")
        lib=pool.library();fn=split_lib();inputs=pool.inputs(task);dist=inputs[1]
        distance=np.ascontiguousarray(-dist)
        provider=locked.RepairProvider("operator-probe",policy,False)
        states=read("archived_states.json")[case]
        initial=tuple(map(tuple,states["initial"]))
        if initial!=tuple(map(tuple,task.initial())):raise ValueError("Archived initial solution differs")
        rng=np.random.default_rng(20372001)
        early_mask=rng.choice(np.arange(1,101),40,replace=False).astype(np.int32)
        _,_,earlier=insertion(task,lib,inputs,initial,early_mask,distance,4201,0,256)
        for name,raw in (("initial",initial),("early",earlier),("terminal",states["terminal"])):
            routes=tuple(map(tuple,raw))
            base=task.evaluate(routes,audit=True)["net_benefit_U"]
            bundles=list(dict.fromkeys(bundle(routes,dist,int(anchor)) for anchor in rng.permutation(len(routes))))
            bundles=[ids for ids in bundles if ids] or [()]
            for k in range(n_groups):
                ids=bundles[k%len(bundles)]
                members=[i for idx in ids for i in routes[idx]]
                others=rng.choice(np.setdiff1d(np.arange(1,101),members),40-len(members),replace=False)
                group=np.asarray(rng.permutation(np.concatenate([members,others])),dtype=np.int32)
                heat=np.ascontiguousarray(provider.heatmap(task,routes,group,"maskco_base",k),dtype=float)
                spatial,residual=residual_edges(dist,heat)
                used_heat=np.ascontiguousarray(spatial+.25*residual) if k%2 else heat
                first=2400+k;second=6400+k
                for op in ("insertion","order"):
                    options={}
                    for which,seed,ranking in (("common",first,distance),("distance",second,distance),("heat",second,used_heat if op=="insertion" else heat)):
                        if op=="order":
                            score,calls=orders(task,fn,inputs,routes,ids,ranking,seed) if ids else (base,0)
                        else:score,calls,_=insertion(task,lib,inputs,routes,group,ranking,seed,(k//2)%2,32)
                        if score<base-1.e-6:raise ValueError("Paired search lost its common start")
                        options[which]=dict(gain=max(0.,score-base),requests=calls)
                    out.append(dict(state=name,group=k,bundle=list(ids),operator=op,options=options))
    result=dict(case=case,rows=out,forwards=provider.forward_calls,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start,core=core_id)
    (HERE/"out"/f"{i}.json").write_text(json.dumps(result,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
