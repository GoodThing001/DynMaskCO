"""Refine a completed masked reconstruction by conditioning on its visible high-confidence customers."""
import ctypes
import json
from pathlib import Path
import sys
import time

import numpy as np
from view_frame_probe import logprob, shuffle_probabilities

HERE = Path(__file__).resolve().parent
ARMS = ("distance","updated","stale","random","shuffled")


def low_confidence(routes, mask, scores):
    ranks = scores.copy()
    np.fill_diagonal(ranks,-np.inf)
    # A shared score receives the same competition rank, including colocated customers.
    incident = {}
    for r in routes:
        around = (0,)+tuple(r)+(0,)
        for p,i in enumerate(r,1):
            prev,nxt = around[p-1],around[p+1]
            a = np.count_nonzero(ranks[i] > ranks[i,prev])
            b = np.count_nonzero(ranks[i] > ranks[i,nxt])
            incident[i] = np.log1p(a) + np.log1p(b)
    return sorted(mask,key=lambda i:(-incident[i],i))[:20]


def main(index):
    import external_common as h
    import pool_search as pool
    cfg=json.loads((HERE/"phase.json").read_text())
    block=cfg["blocks"][index];case=block["case"]
    _,_,access,core,source,locked=h.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    import profit_neural
    core_id=single_cpu_guard()
    policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("two-pass-masked-repair",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    cpu=time.process_time();wall=time.perf_counter();rows=[]
    forwards=repairs=audits=0
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test")
        builder=BaseInferenceInputs(task);lib=pool.library();inputs=pool.inputs(task)
        lib.repair_fixed.argtypes=[pool.F64,pool.F64,pool.F64,pool.F64,pool.I32,ctypes.c_int,pool.I32,pool.I32]
        lib.repair_fixed.restype=None
        dist=np.ascontiguousarray(-inputs[1])

        def predict(routes,mask):
            nonlocal forwards
            forwards+=1
            return np.ascontiguousarray(logprob(profit_neural.predict(policy,builder.build(routes,np.array(mask,np.int32)))))

        def repair(routes,mask,score):
            nonlocal repairs,audits
            parents=pool.pack(routes);ans=parents.copy()
            lib.repair_fixed(*inputs,score,np.ascontiguousarray(mask,np.int32),len(mask),parents,ans)
            rebuilt=pool.unpack(ans)
            U=task.evaluate(rebuilt,audit=True)["net_benefit_U"]
            repairs+=1;audits+=1
            return rebuilt,U

        for j,seed in enumerate(cfg["seeds"]):
            for state in cfg["states"]:
                start=tuple(map(tuple,task.initial() if state=="initial" else block["distance64"][j]))
                orig=task.evaluate(start,audit=True)["net_benefit_U"]
                if state=="distance64" and abs(orig-block["U64"][j])>1.e-6:raise ValueError("Different intermediate parent")
                rng=np.random.default_rng([seed,index,state=="distance64",4891])
                order=rng.permutation(np.arange(1,101)).tolist()
                mask=order[:40];padding=order[40:60];rand=mask[:20]
                first=predict(start,mask)
                d,du=repair(start,mask,dist);h1,hu=repair(start,mask,first)
                if abs(hu-du)>1.e-6 and d==h1:raise ValueError("Fleet identity")
                dm=low_confidence(d,mask,dist);hm=low_confidence(h1,mask,first)
                second=predict(h1,hm+padding);random_heat=predict(h1,rand+padding)
                placebo=np.ascontiguousarray(shuffle_probabilities(second,inputs[1],h1,hm+padding,rng))
                methods={
                    "distance":(d,du,dm,dist), "updated":(h1,hu,hm,second),
                    "stale":(h1,hu,hm,first), "random":(h1,hu,rand,random_heat),
                    "shuffled":(h1,hu,hm,placebo)
                }
                results={}
                for arm,(parent,pU,hidden,score) in methods.items():
                    final,v=repair(parent,hidden,score)
                    # Visible customers retain their predecessors' relative order within each remaining route skeleton.
                    hidden=set(hidden)
                    skeleton=lambda rs:sorted(tuple(i for i in r if i not in hidden) for r in rs if any(i not in hidden for i in r))
                    if skeleton(final)!=skeleton(parent):raise ValueError("Visible customers changed routes")
                    results[arm]=dict(first=pU,second=v,best=max(orig,pU,v),routes=final)
                rows.append(dict(seed=seed,state=state,original=orig,mask=mask,padding=padding,
                                 first_fleets=dict(distance=d,neural=h1),
                                 distance_mask=dm,neural_mask=hm,random_mask=rand,arms=results))
    result=dict(case=case,rows=rows,forwards=forwards,repairs=repairs,audited_candidates=audits,core=core_id,
                cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    (HERE/f"out/{index}.json").write_text(json.dumps(result,allow_nan=False))

if __name__=="__main__":
    main(int(sys.argv[1]))
