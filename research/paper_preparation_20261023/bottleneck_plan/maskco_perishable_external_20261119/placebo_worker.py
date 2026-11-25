"""Run the locked order-and-split solver with aligned, stratified random or distance-only scores."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pool_search as pool
from heat_placebo import PairedRanking
HERE=Path(__file__).resolve().parent

def main(i):
    plan=json.loads((HERE/"phase.json").read_text());block=plan["blocks"][i]
    case,seed,role=block["case"],block["seed"],block.get("role","validation")
    access,core,src,locked=pool.study.setup("SLOT")
    from timed_search import single_cpu_guard
    affinity=single_cpu_guard()
    policy,scope=pool.study.load("SLOT",access,core,src,locked,"c205" if role=="train" else case)
    if role=="train":scope=access.Scope("placebo-qualification",frozenset([(0,role,case)]),frozenset(["initialize","infer","search"]))
    lib=pool.library();records={}
    with access.restricted_scope(scope):
        for name in block["arms"]:
            native="ORDER_SPLIT_D" if name=="ALIGNED_D" else "ORDER_SPLIT"
            cls=locked.RepairProvider;created=[]
            def provider(arm,model,distance_only):
                base=cls(arm,model,distance_only)
                if distance_only:return base
                wrapped=PairedRanking(base,seed,name=="SHUFFLED");created.append(wrapped)
                return wrapped
            with patch.object(locked,"RepairProvider",provider):
                record=pool.solve(lib,core,locked,policy,case,role,seed,native,plan["seconds"])
            record.update(cpu=affinity,method=name)
            if created:
                w,=created
                record.update(changed_top8=float(np.mean(w.changed_top8)),directional_logit_recovery_error=w.reconstruction_error)
                if w.reconstruction_error>0.002 or not .001<record["changed_top8"]<=1:raise ValueError("Placebo failed to isolate edge alignment")
            records[name]=record
    if len({json.dumps(v["initial_routes"]) for v in records.values()})!=1:raise ValueError("Common initializer differs")
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,seed=seed,arms=records),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
