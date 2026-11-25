"""Compare complete model outputs and resource usage on shared training masks."""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import cpu_maskco
import pool_search as pool

HERE=Path(__file__).resolve().parent


def neighbors(a,b):
    def top(h):
        order=np.argsort(-h,axis=-1,kind="stable")
        return order[:,:min(8,order.shape[-1])]
    x,y=top(a),top(b)
    return [len(set(i)&set(j))/len(i) for i,j in zip(x,y)]


def position_agreement(task,routes,group,first,second):
    active=group[:10];hidden=set(active)
    orders=[tuple(i for i in r if i not in hidden) for r in routes]
    out=[]
    for i in active:
        slots=[(a,b) for r in orders if r and sum(task.ins["nodes"][j][3] for j in r)+task.ins["nodes"][i][3]<=20000.+1e-7
               for a,b in zip((0,)+r,r+(0,))]
        if not slots:continue
        a,b=np.asarray(slots).T
        values=[(h[a,i]+h[i,b]-h[a,b])[None] for h in (first,second)]
        out.extend(neighbors(*values))
    return out


def main(index):
    from contextlib import nullcontext
    cfg=json.loads((HERE/"phase.json").read_text())
    case=cfg["blocks"][index]["case"]
    study=pool.study;access,core,source,locked=study.setup("SLOT")
    from base_inputs import BaseInferenceInputs
    import profit_neural
    from timed_search import single_cpu_guard
    core_ids=single_cpu_guard()
    if case not in core.fold_definition(0)["train"]:raise ValueError("Expected training-only case")
    scope=access.Scope("maskco-cpu-comparison",frozenset([(0,"train",case)]),frozenset(["initialize","infer","search"]))
    try:
        from threadpoolctl import threadpool_info,threadpool_limits
        limits=threadpool_limits(limits=1)
    except ImportError:
        if os.environ.get("OPENBLAS_NUM_THREADS")!="1" or os.environ.get("MKL_NUM_THREADS")!="1":raise
        limits=nullcontext()
        threadpool_info=lambda:[]
    with limits:
        backends=[(r["internal_api"],r["num_threads"]) for r in threadpool_info()]
        pair=cpu_maskco.load_pair(study,access,core,source,locked)
        with access.restricted_scope(scope):
            task=core.ProfitTask(case,fold=0,role="train")
            routes=task.initial()
            second_root=pool.solve(pool.library(),core,locked,None,case,"train",1148,"POOL_D",2)["routes"]
            builder=BaseInferenceInputs(task);rng=np.random.default_rng(31542)
            groups=[rng.choice(np.arange(1,101),40,replace=False) for _ in range(cfg["mask_count"])]
            times={m:[] for m in ("jax","blas")}
            outputs=[lambda b:profit_neural.predict(pair,b),pair.blas.logits]
            warm=builder.build(routes,groups[0])
            for fn in outputs:fn(warm)
            max_err=0.;row_overlap=[];slot_overlap=[]
            for state_no,state in enumerate((routes,second_root)):
                for j,g in enumerate(groups):
                    batch=builder.build(state,g)
                    results={}
                    for idx in ((0,1) if (state_no+j)%2==0 else (1,0)):
                        m=("jax","blas")[idx]
                        start=time.perf_counter();cpu=time.process_time()
                        h=outputs[idx](batch)
                        times[m].append([time.perf_counter()-start,time.process_time()-cpu])
                        if not np.isfinite(h).all():raise ValueError("Nonfinite model output")
                        h=h-h.max(axis=1,keepdims=True)
                        results[m]=h-np.log(np.exp(h).sum(axis=1,keepdims=True))
                    a,b=results["jax"],results["blas"];off=~np.eye(len(a),dtype=bool)
                    max_err=max(max_err,float(abs(a[off]-b[off]).max()))
                    row_overlap.extend(neighbors(a,b))
                    slot_overlap.extend(position_agreement(task,state,g,a,b))
            tw={m:np.sum(v,axis=0).tolist() for m,v in times.items()}
            wall,cpu=1-np.asarray(tw["blas"])/tw["jax"]
            row,slot=float(np.mean(row_overlap)),float(np.mean(slot_overlap))
            changes=set(map(tuple,routes))!=set(map(tuple,second_root))
            good=bool(changes and max_err<=.002 and row>=.995 and slot>=.99 and min(wall,cpu)>=.2)
            result=dict(case=case,affinity=core_ids,backend_threads=backends,max_error=max_err,neighbor_overlap=row,
                   slot_overlap=slot,changed_state=changes,masks=cfg["mask_count"],totals=tw,wall_saved=wall,cpu_saved=cpu,gate=good,times=times)
    (HERE/"out"/f"{index}.json").write_text(json.dumps(result,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
