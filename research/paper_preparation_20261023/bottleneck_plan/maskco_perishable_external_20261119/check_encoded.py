"""Measure full and split inference on matched masks and different training routes."""
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import encoded_mask as enc
import pool_search as pool

HERE=Path(__file__).resolve().parent
def main(index):
    import jax
    plan=json.loads((HERE/"phase.json").read_text())
    b=plan["blocks"][index]
    if b["case"]!="r101" or b["role"]!="train":raise ValueError("Qualification uses the old training case")
    enc.check_mask_family()
    study=pool.study
    access,core,source,locked=study.setup("SLOT")
    import profit_neural
    from base_inputs import BaseInferenceInputs
    from timed_search import single_cpu_guard
    affinity=single_cpu_guard()
    original=profit_neural.CompiledPredictor
    def factory(model):
        with patch.object(profit_neural,"CompiledPredictor",original):
            return enc.with_split_inference(model)
    with patch.object(profit_neural,"CompiledPredictor",factory):
        policy,_=study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("encoded-mask-qualification",frozenset([(0,"train","r101")]),frozenset(["infer","initialize","search"]))
    with access.restricted_scope(scope):
        task=core.ProfitTask("r101",fold=0,role="train")
        routes=task.initial()
        alternative=pool.solve(pool.library(),core,locked,None,"r101","train",7193,"POOL_D",2)["routes"]
        if {tuple(x) for x in alternative}==set(map(tuple,routes)):raise ValueError("Need two different route adjacencies")
        masks=enc.covering_masks(b["seed"])
        builder=BaseInferenceInputs(task)
        batches=[(j,builder.build(route,g)) for route in (routes,alternative,routes,alternative) for j,g in enumerate(masks)]
        enc.split_batch(policy,batches[0][1],0,{})
        reference=[];split=[];seconds={"full":0.,"split":0.}
        cache={}
        for i,(j,batch) in enumerate(batches):
            for mode in (("full","split") if i%2==0 else ("split","full")):
                start=time.process_time()
                if mode=="full":
                    output=enc.log_heat(profit_neural.predict(policy,batch))
                    reference.append(output)
                else:
                    output=enc.log_heat(enc.split_batch(policy,batch,j,cache))
                    split.append(output)
                seconds[mode]+=time.process_time()-start
        err=0.;overlaps=[]
        offdiag=~np.eye(101,dtype=bool)
        for ref,new in zip(reference,split):
            err=max(err,float(np.max(np.abs(ref[offdiag]-new[offdiag]))))
            r=np.argsort(-ref,axis=1,kind="stable")[:,:8];q=np.argsort(-new,axis=1,kind="stable")[:,:8]
            overlaps.extend(len(set(a)&set(c))/8 for a,c in zip(r,q))
        changed=sum(not np.allclose(reference[k][offdiag],reference[k+10][offdiag],atol=1e-4) for k in range(10))
        fraction=1-seconds["split"]/seconds["full"]
        result=dict(cpu=affinity,train_case="r101",forwards=len(batches),unique_encodes=len(cache),changed_adjacency_responses=changed,
                    full_cpu_s=seconds["full"],reuse_cpu_s=seconds["split"],saved_fraction=fraction,
                    max_abs_log_heat_error=err,mean_top8_overlap=float(np.mean(overlaps)))
        result["gate"]=bool(len(cache)==10 and changed==10 and err<=1e-3 and np.mean(overlaps)>=.995 and fraction>=.10)
        (HERE/"out"/f"{index}.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main(int(sys.argv[1]))
