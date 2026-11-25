"""Score the same 20 actual removals while varying only which other customers MASKCO observes."""
import ctypes
import json
import sys
import time
from pathlib import Path
import numpy as np
from view_frame_probe import logprob, shuffle_probabilities

HERE=Path(__file__).resolve().parent
ARMS=("distance","true20","pad40a","pad40b","shuffle20")


def main(ix):
    import external_common
    import pool_search as pool
    from paired_operator_probe import insertion
    cfg=json.loads((HERE/"phase.json").read_text());b=cfg["blocks"][ix];case=b["case"]
    _,_,access,core,source,locked=external_common.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    import profit_neural
    core_id=single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("same-movers-conditional-context",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    cpu=time.process_time();wall=time.perf_counter();rows=[];forwards=requests=audits=0
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test")
        builder=BaseInferenceInputs(task);lib=pool.library();inputs=pool.inputs(task);dist=np.ascontiguousarray(-inputs[1])
        lib.repair_fixed.argtypes=[pool.F64,pool.F64,pool.F64,pool.F64,pool.I32,ctypes.c_int,pool.I32,pool.I32]
        lib.repair_fixed.restype=None
        for state in cfg["states"]:
            parent=tuple(map(tuple,task.initial() if state=="initial" else b["distance64"][1]))
            old=task.evaluate(parent,audit=True)["net_benefit_U"]
            if state=="distance64" and abs(old-b["U64"][1])>1.e-6:raise ValueError("Archived common parent differs")
            for seed in cfg["seeds"]:
                rng=np.random.default_rng([seed,ix,state=="distance64",1889]);perm=rng.permutation(np.arange(1,101)).tolist()
                mask=perm[:20];pa=perm[20:40];pb=perm[40:60];data={};heat={}
                for name,ids in (("true20",mask),("pad40a",mask+pa),("pad40b",mask+pb)):
                    batch=builder.build(parent,ids)
                    if set(np.flatnonzero(batch["features"][0,:,10]==0))!=set(ids):
                        raise ValueError("Neural mask and requested context differ")
                    adjacency=batch["adjacency"][0]
                    if np.any(adjacency[ids,:]) or np.any(adjacency[:,ids]):raise ValueError("Hidden links remain")
                    data[name]=dict(timestep=float(batch["timestep"][0]),visible_customer_half_edges=int(adjacency[1:].sum()))
                    heat[name]=np.ascontiguousarray(logprob(profit_neural.predict(policy,batch)));forwards+=1
                if any(data[k]["timestep"]>data["true20"]["timestep"]+1.e-6 for k in ("pad40a","pad40b")):
                    raise ValueError("Extra occlusion increased observed context")
                heat["distance"]=dist
                heat["shuffle20"]=np.ascontiguousarray(shuffle_probabilities(heat["true20"],inputs[1],parent,mask,rng))
                top={}
                for k in ("true20","pad40a","pad40b"):
                    t=heat[k].copy();np.fill_diagonal(t,-np.inf)
                    top[k]=np.argsort(-t[mask,:],axis=1,kind="stable")[:,:8]
                overlap={k:sum(len(set(x)&set(y)) for x,y in zip(top["true20"],top[k]))/(20*8) for k in ("pad40a","pad40b")}
                selected=np.ascontiguousarray(mask,np.int32);out={}
                for name in ARMS:
                    oldpack=pool.pack(parent);newpack=oldpack.copy();score=heat[name]
                    lib.repair_fixed(*inputs,score,selected,len(selected),oldpack,newpack)
                    fleet=pool.unpack(newpack);u=task.evaluate(fleet,audit=True)["net_benefit_U"]
                    final,n,rs=insertion(task,lib,inputs,fleet,selected,score,seed+9321,1,32)
                    audits+=2;requests+=n
                    if final<u-1.e-6:raise ValueError("32-step search lost initial candidate")
                    out[name]=dict(first_U=u,refined_U=final,best_U=max(old,final),requests32=n,
                                   winning_routes=parent if old>=final else rs)
                rows.append(dict(state=state,seed=seed,original_U=old,mask=mask,padding_a=pa,padding_b=pb,context=data,
                                 top8_overlap=overlap,arms=out))
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,rows=rows,forwards=forwards,audited_candidates=audits,
            refinement_route_requests=requests,core=core_id,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall),allow_nan=False))


if __name__=="__main__":
    main(int(sys.argv[1]))
