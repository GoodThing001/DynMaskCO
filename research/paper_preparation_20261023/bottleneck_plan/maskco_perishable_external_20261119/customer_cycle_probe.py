"""Measure matched economic opportunities for reassigning masked customers across three routes."""
from itertools import combinations
from pathlib import Path
import json
import sys
import time

import numpy as np

from heat_placebo import remap

HERE=Path(__file__).resolve().parent


def replacement_tables(task, routes, customers, weights):
    """For each outgoing/incoming customer, maximize edge score and strict profit separately."""
    count=len(customers)
    table=np.full((count,count,len(weights)),-np.inf)
    values=np.full((count,count),-np.inf)
    witness={}
    owners={v:k for k,r in enumerate(routes) for v in r}
    nodes=np.asarray(task.ins["nodes"])
    capacity=task.c["routing"]["capacity_kg"]
    for j,old in enumerate(customers):
        original=routes[owners[old]]
        remaining=tuple(v for v in original if v!=old)
        original_price=task.route(original).net_benefit
        mass=sum(nodes[v,3] for v in remaining)
        old_path=(0,)+original+(0,)
        old_score=weights[:,old_path[:-1],old_path[1:]].sum(axis=1)
        path=(0,)+remaining+(0,)
        left,right=np.array(path[:-1]),np.array(path[1:])
        remaining_score=weights[:,left,right].sum(axis=1) if remaining else np.zeros(len(weights))
        subtracted=weights[:,left,right] if remaining else np.zeros((len(weights),1))
        for k,incoming in enumerate(customers):
            if owners[old]==owners[incoming] or mass+nodes[incoming,3]>capacity+1.e-7:continue
            inserts=weights[:,left,incoming]+weights[:,incoming,right]-subtracted
            table[j,k]=remaining_score-old_score+inserts.max(axis=1)
            for pos in range(len(path)-1):
                proposed=remaining[:pos]+(incoming,)+remaining[pos:]
                result=task.route(proposed)
                gain=result.net_benefit-original_price if result.feasible else -np.inf
                if gain>values[j,k]:
                    values[j,k]=gain
                    witness[j,k]=proposed
    return table,values,witness,owners


def shortlist(scores, gains):
    if not len(gains):return {}
    distance_order=np.argsort(-scores[:,0],kind="stable")
    common=distance_order[:16]
    extra=np.ones(len(gains),dtype=bool)
    extra[common]=False
    eligible=np.flatnonzero(extra)
    winners={}
    for arm,name in enumerate(("distance","real","shuffled")):
        order=np.argsort(-scores[eligible,arm],kind="stable")
        chosen=np.r_[common,eligible[order[:16]]]
        winners[name]=int(chosen[np.argmax(gains[chosen])])
    winners["oracle"]=int(np.argmax(gains))
    return winners


def probe(task,routes,customers,weights,full):
    table,values,witness,owners=replacement_tables(task,routes,customers,weights)
    triples=np.array(list(combinations(range(len(customers)),3)),dtype=np.int32)
    cycles=np.row_stack((triples,triples[:,[0,2,1]]))
    i,j,k=cycles.T
    eligible=np.isfinite(table[i,j,0])&np.isfinite(table[j,k,0])&np.isfinite(table[k,i,0])
    cycles=cycles[eligible]
    i,j,k=cycles.T
    scores=table[i,j]+table[j,k]+table[k,i]
    gains=values[i,j]+values[j,k]+values[k,i]
    winners=shortlist(scores,gains)
    outputs={arm:0. for arm in ("distance","real","shuffled","oracle")}
    for arm,picked in winners.items():
        if gains[picked]<=1.e-7:continue
        changed=list(routes)
        a,b,c=cycles[picked]
        for old,new in ((a,b),(b,c),(c,a)):
            changed[owners[customers[old]]]=witness[old,new]
        independent=task.evaluate(tuple(changed),audit=True)["net_benefit_U"]-full
        if abs(independent-gains[picked])>1.e-6:
            raise ValueError("Three-route gain differs from independent whole-solution evaluation")
        outputs[arm]=independent
    return outputs,scores,gains,cycles


def main(ix):
    import external_common as harness
    import pool_search as pool
    plan=json.loads((HERE/"phase.json").read_text())
    case=plan["blocks"][ix]["case"]
    states=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=harness.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    single_cpu_guard()
    model,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("masked-three-customer-cycle",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    rng=np.random.default_rng(40681207)
    records=[];all_features=[];all_gains=[];all_ids=[]
    start=time.process_time()
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");dist=np.asarray(task.ins["distance"])
        provider=locked.RepairProvider("masked-rotation",model,False)
        for label in ("initial","terminal"):
            routes=tuple(map(tuple,states[label]))
            if label=="initial" and routes!=tuple(map(tuple,task.initial())):raise ValueError("Initial states differ")
            full=task.evaluate(routes,audit=True)["net_benefit_U"]
            for t in range(plan["masks_per_state"]):
                group=rng.choice(np.arange(1,101),40,replace=False)
                customers=tuple(map(int,group))
                raw=provider.heatmap(task,routes,group,"maskco_base",t)
                actual,null=remap(raw,dist,routes,group,rng)
                effects,x,y,cycles=probe(task,routes,customers,np.stack((-dist,actual,null)),full)
                records.append(dict(state=label,t=t,cycles=len(y),feasible=int(np.isfinite(y).sum()),positive=int((y>1.e-7).sum()),gains=effects))
                all_ids.append(group[cycles].astype(np.uint8));all_features.append(x.astype(np.float32));all_gains.append(y.astype(np.float32))
    target=HERE/"out"
    np.savez_compressed(target/f"{ix}.npz",scores=np.concatenate(all_features),gains=np.concatenate(all_gains),cycles=np.concatenate(all_ids))
    meta=dict(case=case,rows=records,forwards=provider.forward_calls,requests=sum(task.route.cache_info()[:2]),cpu_s=time.process_time()-start)
    (target/f"{ix}.json").write_text(json.dumps(meta,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
