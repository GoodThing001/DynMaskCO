"""Keep the destruction stream fixed while following different masked neural warm starts."""
from pathlib import Path
import ctypes,sys,time,json
import numpy as np
from view_frame_probe import view,logprob,blend,shuffle_probabilities
HERE=Path(__file__).resolve().parent

def suffix_stream(seed,steps):
    rng=np.random.default_rng([seed,94183])
    return [np.ascontiguousarray(rng.choice(np.arange(1,101),6+(i%7),replace=False),dtype=np.int32) for i in range(steps)]

def main(ix):
    import external_common as h
    import pool_search as pool
    from block_split import bundle,library
    from paired_operator_probe import insertion,order_fleet
    cfg=json.loads((HERE/"phase.json").read_text());b=cfg["blocks"][ix];c=b["case"]
    _,_,access,core,source,locked=h.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from base_inputs import BaseInferenceInputs
    from timed_search import single_cpu_guard
    import profit_neural
    core_id=single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("common-distance-continuation",frozenset([(0,"external_test",c)]),frozenset(["initialize","infer","search"]))
    wall=time.perf_counter();cpu=time.process_time();records=[];suffix_calls=forwards=prefix_calls=0
    with access.restricted_scope(scope):
        task=core.ProfitTask(c,fold=0,role="external_test");builder=BaseInferenceInputs(task)
        lib=pool.library();fn=library();inputs=pool.inputs(task);d=inputs[1];dist=np.ascontiguousarray(-d)
        lib.repair_fixed.argtypes=[pool.F64,pool.F64,pool.F64,pool.F64,pool.I32,ctypes.c_int,pool.I32,pool.I32];lib.repair_fixed.restype=None
        start=tuple(map(tuple,task.initial()));original=task.evaluate(start,audit=True)["net_benefit_U"]
        for seed in cfg["seeds"]:
            rng=np.random.default_rng([seed,662973]);ids=next((v for j in rng.permutation(len(start)) if (v:=bundle(start,d,int(j)))),())
            members=[i for j in ids for i in start[j]]
            more=rng.choice(sorted(set(range(1,101))-set(members)),40-len(members),replace=False)
            mask=np.array(rng.permutation(members+more.tolist()),dtype=np.int32)
            batch=builder.build(start,mask)
            frames=[logprob(profit_neural.predict(policy,view(batch,j))) for j in range(8)];forwards+=8
            ensemble=blend(frames);shuffled=shuffle_probabilities(ensemble,d,start,mask,rng)
            stream=suffix_stream(seed,cfg["steps"]);out={}
            for a,mat in (("distance",dist),("original",frames[0]),("eight",ensemble),("shuffled",shuffled)):
                weights=np.ascontiguousarray(mat,dtype=float)
                u,n,rs=insertion(task,lib,inputs,start,mask,weights,seed,seed%2,32)
                v,m,split=order_fleet(task,fn,inputs,start,ids,weights,seed+10000,24) if ids else (original,0,start)
                prefix_calls+=n+m;best,win=(u,rs) if u>=v else (v,split)
                if abs(task.evaluate(win,audit=True)["net_benefit_U"]-best)>1.e-6:raise ValueError("Prefix audit")
                initial=pool.pack(win);timeline={"0":dict(U=best,routes=win)};accept=0
                for count,removed in enumerate(stream,1):
                    answer=initial.copy()
                    lib.repair_fixed(*inputs,dist,removed,len(removed),initial,answer);suffix_calls+=1
                    move=pool.unpack(answer);score=task.evaluate(move)["net_benefit_U"]
                    if score>best+1.e-7:
                        best,win,initial=score,move,answer;accept+=1
                    if count in (64,cfg["steps"]):
                        if abs(task.evaluate(win,audit=True)["net_benefit_U"]-best)>1.e-6:raise ValueError("Continuation audit")
                        timeline[str(count)]=dict(U=best,routes=win)
                out[a]=dict(checkpoints=timeline,accepted=accept)
            records.append(dict(seed=seed,orig_U=original,mask=mask.tolist(),group=ids,arms=out))
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=c,seeds=records,forwards=forwards,strict_suffix_calls=suffix_calls,prefix_route_requests=prefix_calls,
          cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,core=core_id),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
