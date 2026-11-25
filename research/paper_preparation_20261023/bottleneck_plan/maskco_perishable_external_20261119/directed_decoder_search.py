"""Run the cross-fitted directed readout inside the complete 60-second ORDER_SPLIT search."""
from pathlib import Path
import json
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=("HEAD","PHYSICAL","ORDER_SPLIT","ORDER_SPLIT_D","DIST","DIST_SLOT")


def read(p):return json.loads(p.read_text(encoding="utf8"))


class Provider:
    def __init__(self,model,head,neural,customer_mix):
        self.model,self.neural=model,neural
        self.scales=[head[k] for k in ("mu","sd","eh","pca")]
        self.W=head["neural" if neural else "physical"]
        self.physical_W=head["physical"] if customer_mix and neural else None
        self.base=self.inputs=None
        self.forward_calls=0;self.forward_seconds=0.;self.updates=0

    def heatmap(self,task,routes,removed,variant,step):
        from base_inputs import BaseInferenceInputs
        from directed_decoder_features import embedding,physical,normalized_logits
        from directed_decoder_fit import design,distance_score
        start=time.perf_counter()
        if self.inputs is None:
            self.inputs=BaseInferenceInputs(task)
            self.base=distance_score(task.ins["distance"])
        batch=self.inputs.build(routes,removed)
        row=dict(phys=physical(task.ins["nodes"],batch))
        if self.neural:
            row["h"]=embedding(self.model,batch);self.forward_calls+=1
        x=design(row,self.scales,self.neural)
        logits=self.base+(x@self.W)@x.T
        if self.physical_W is not None:
            from customer_mixture import combine
            ph=design(row,self.scales,False)
            logits=combine(self.base+(ph@self.physical_W)@ph.T,logits)
        else:
            logits=normalized_logits(logits)
        self.forward_seconds+=time.perf_counter()-start;self.updates+=1
        return logits


def cell(arm,case,seed,cfg):
    import external_common as api
    if arm in ("DIST","DIST_SLOT"):
        return api.endpoint(arm,case,seed,cfg)
    if arm not in ARMS:raise ValueError(arm)
    from directed_decoder_features import attach_embeddings,embedding
    import pool_search as pool
    _,_,access,core,src,locked=api.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    core_id=single_cpu_guard()
    begin=time.perf_counter()
    policy=None
    if arm=="HEAD":
        policy=attach_embeddings(pool.study,access,core,src,locked)
        warm=access.Scope("readout-warmup",frozenset([(0,"train","r101")]),frozenset(["initialize","infer","warmup"]))
        with access.restricted_scope(warm):
            task=core.ProfitTask("r101",fold=0,role="train")
            embedding(policy,BaseInferenceInputs(task).build(task.initial(),np.arange(1,41)))
    elif arm=="ORDER_SPLIT":
        policy,_=pool.study.load("SLOT",access,core,src,locked,"c205")
    fold=read(HERE/"readout_folds.json")[case]
    novel=arm in ("HEAD","PHYSICAL")
    if novel:
        with np.load(HERE/f"readout_fold{fold}.npz") as f:
            provider=Provider(policy,dict(f),arm=="HEAD",cfg.get("customer_mix",False))
    elapsed=time.perf_counter()-begin
    scope=access.Scope("cross-fit-masked-readout",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    with access.restricted_scope(scope):
        lib=pool.library()
        inner=arm if not novel else "ORDER_SPLIT"
        if novel:
            with patch.object(locked,"RepairProvider",lambda *args:provider):
                result=pool.solve(lib,core,locked,policy,case,"external_test",seed,inner,cfg["seconds"])
            result["readouts"]=provider.updates
        else:
            result=pool.solve(lib,core,locked,policy,case,"external_test",seed,inner,cfg["seconds"])
    result.update(method=arm,development_fold=fold,cpu=core_id,setup_s=elapsed)
    return result


def main():
    ix=int(sys.argv[1]);cfg=read(HERE/"phase.json");b=cfg["blocks"][ix];case,seed=b["case"],b["seed"]
    if len(sys.argv)==3:
        arm=sys.argv[2]
        if arm not in b["arms"]:raise ValueError("Unregistered arm")
        q=cell(arm,case,seed,cfg)
        (HERE/f"out/{ix}_{arm}.cell.json").write_text(json.dumps(q,allow_nan=False))
        return
    results={}
    for arm in b["arms"]:
        subprocess.run([sys.executable,"-B",str(HERE/"directed_decoder_search.py"),str(ix),arm],check=True)
        results[arm]=read(HERE/f"out/{ix}_{arm}.cell.json")
    if len({json.dumps(r["initial_routes"]) for r in results.values()})!=1:raise ValueError("Initial solutions differ")
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,seed=seed,arms=results),allow_nan=False))
if __name__=="__main__":main()
