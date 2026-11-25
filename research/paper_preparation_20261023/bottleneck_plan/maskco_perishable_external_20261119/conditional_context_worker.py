"""Evaluate exact frozen model ranks under two mask boundaries on the same parent."""
import json,sys,time
from pathlib import Path
import numpy as np
from directed_decoder_features import attach_embeddings,embedding,normalized_logits
HERE=Path(__file__).resolve().parent

def main(ix):
    import external_common as harness
    import pool_search as pool
    _,_,access,core,source,locked=harness.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    import profit_neural
    single_cpu_guard();cfg=json.loads((HERE/"phase.json").read_text());b=cfg["blocks"][ix];case=b["case"]
    archive=json.loads((HERE/"archived_states.json").read_text())[case]
    policy=attach_embeddings(pool.study,access,core,source,locked)
    scope=access.Scope("paired-frozen-mask",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    data={};err=0.;start=time.process_time()
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");builder=BaseInferenceInputs(task)
        if tuple(map(tuple,task.initial()))!=tuple(map(tuple,archive["initial"])):raise ValueError("Common start")
        for routes in archive.values():
            if isinstance(routes,list):task.evaluate(tuple(map(tuple,routes)),audit=True)
        for ri,z in enumerate(b["regions"]):
            parent=tuple(map(tuple,archive[z["state"]]))
            for boundary in ("broad","narrow"):
                mask=z[boundary]
                if len(mask)!=len(set(mask)) or len(mask)!=40:raise ValueError("Mask")
                if sorted(set(z["ids"])&set(mask))!=sorted(z["ids"] if boundary=="broad" else z["hidden"]):raise ValueError("Unexpected regional visibility")
                batch=builder.build(parent,np.array(mask,dtype=int));h=embedding(policy,batch);raw=(h@h.T)*policy.edge_scale
                if ri==0:
                    off=~np.eye(len(h),dtype=bool)
                    err=max(err,float(np.max(np.abs(normalized_logits(raw)[off]-normalized_logits(profit_neural.predict(policy,batch))[off]))))
                if not np.isfinite(raw).all() or raw.shape!=(101,101):raise ValueError("Frozen edge matrix")
                data[f"{ri}_{boundary}"]=raw.astype(np.float32)
    if err>.002:raise ValueError("Frozen logit reconstruction mismatch")
    np.savez_compressed(HERE/f"out/{ix}.npz",**data)
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,contexts=len(b["regions"]),forwards=len(data),max_error=err,cpu_s=time.process_time()-start)))
if __name__=="__main__":main(int(sys.argv[1]))
