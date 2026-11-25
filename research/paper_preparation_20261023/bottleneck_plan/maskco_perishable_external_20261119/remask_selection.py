"""Isolate the use of the masked neural prior for choosing customers to reopen."""
import ctypes
import json
import sys
import time
import numpy as np

from iterative_remask import HERE, low_confidence
from view_frame_probe import logprob, shuffle_probabilities


def main(ix):
    import external_common
    import pool_search as pool
    cfg=json.loads((HERE/"phase.json").read_text())
    block=cfg["blocks"][ix];case=block["case"]
    _,_,access,core,source,locked=external_common.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    import profit_neural
    core_id=single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("common-distance-second-insertion",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    wall=time.perf_counter();cpu=time.process_time();rows=[];n_forward=n_repair=0
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test")
        builder=BaseInferenceInputs(task);lib=pool.library();inputs=pool.inputs(task);dist=np.ascontiguousarray(-inputs[1])
        lib.repair_fixed.argtypes=[pool.F64,pool.F64,pool.F64,pool.F64,pool.I32,ctypes.c_int,pool.I32,pool.I32]
        lib.repair_fixed.restype=None
        def repair(rs,mask,mat):
            nonlocal n_repair
            before=pool.pack(rs);after=before.copy()
            lib.repair_fixed(*inputs,mat,np.ascontiguousarray(mask,np.int32),len(mask),before,after)
            new=pool.unpack(after)
            value=task.evaluate(new,audit=True)["net_benefit_U"]
            n_repair+=1
            return new,value
        for state in cfg["states"]:
            parent=tuple(map(tuple,task.initial() if state=="initial" else block["distance64"][0]))
            old=task.evaluate(parent,audit=True)["net_benefit_U"]
            if state=="distance64" and abs(old-block["U64"][0])>1.e-6:raise ValueError("Archived parent changed")
            for seed in cfg["seeds"]:
                rng=np.random.default_rng([seed,ix,state=="distance64",67501])
                m=rng.permutation(np.arange(1,101)).tolist()[:40]
                heat=np.ascontiguousarray(logprob(profit_neural.predict(policy,builder.build(parent,np.asarray(m,np.int32)))))
                n_forward+=1
                shuffled=shuffle_probabilities(heat,inputs[1],parent,m,rng)
                results={}
                for name,rank in (("distance",dist),("neural",heat)):
                    first,u=repair(parent,m,rank)
                    masks={a:low_confidence(first,m,r) for a,r in (("distance",dist),("neural",heat),("shuffled",shuffled))}
                    masks["random"]=m[:20]
                    proposals={}
                    for a,mask in masks.items():
                        routes,v=repair(first,mask,dist)
                        hidden=set(mask)
                        skeleton=lambda rs:sorted(tuple(i for i in r if i not in hidden) for r in rs if any(i not in hidden for i in r))
                        if skeleton(routes)!=skeleton(first):raise ValueError("Unmasked skeleton differs")
                        proposals[a]=dict(mask=mask,U=v,best=max(old,u,v),routes=routes)
                    results[name]=dict(first_U=u,first_routes=first,second=proposals)
                rows.append(dict(seed=seed,state=state,original_U=old,mask=m,arms=results))
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,rows=rows,forwards=n_forward,audited_candidates=n_repair,
            core=core_id,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall),allow_nan=False))


if __name__=="__main__":
    main(int(sys.argv[1]))
