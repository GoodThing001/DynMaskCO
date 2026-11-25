"""Give a second fixed-budget reconstruction a separate seed and a prescribed coordinate frame."""
from pathlib import Path
import sys,time,json
import numpy as np
from view_frame_probe import view,logprob,shuffle_probabilities
HERE=Path(__file__).resolve().parent

def main(ix):
    import external_common as h
    import pool_search as pool
    from block_split import library
    from paired_operator_probe import insertion,orders
    cfg=json.loads((HERE/"phase.json").read_text());b=cfg["blocks"][ix];case=b["case"]
    archive=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=h.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from base_inputs import BaseInferenceInputs
    from timed_search import single_cpu_guard
    import profit_neural
    core_id=single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("independent-second-view",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    tick=time.process_time();wall=time.perf_counter();out=[];n_forward=0
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");builder=BaseInferenceInputs(task)
        lib=pool.library();fn=library();inputs=pool.inputs(task);d=inputs[1]
        for z in b["contexts"]:
            state=z["state"];k=z["rep"];ids=z["group"];mask=np.array(z["mask"],dtype=np.int32);routes=tuple(map(tuple,archive[state]))
            original=task.evaluate(routes,audit=True)["net_benefit_U"]
            if abs(original-z["original_U"])>1.e-6 or any(i not in mask for j in ids for i in routes[j]):raise ValueError("First-trial identity")
            batch=builder.build(routes,mask);first=logprob(profit_neural.predict(policy,batch));flip=logprob(profit_neural.predict(policy,view(batch,3)));n_forward+=2
            seed=74351+b["first_index"]*97+5*(state=="terminal")+k
            shuffled=shuffle_probabilities(flip,d,routes,mask,np.random.default_rng(seed+765))
            arms={}
            for a,scores in (("distance",-d),("v0",first),("v3",flip),("shuffled",shuffled)):
                scores=np.ascontiguousarray(scores,dtype=float)
                v,n,_=insertion(task,lib,inputs,routes,mask,scores,seed,k%2,32)
                w,m=orders(task,fn,inputs,routes,ids,scores,seed+10000,24) if ids else (original,0)
                if min(v,w)<original-1.e-6:raise ValueError("Feasible parent lost")
                arms[a]=dict(gain=max(v,w)-original,requests=n+m)
            out.append(dict(state=state,rep=k,group=ids,mask=mask.tolist(),original_U=original,arms=arms))
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,rows=out,forwards=n_forward,cpu_s=time.process_time()-tick,wall_s=time.perf_counter()-wall,core=core_id),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
