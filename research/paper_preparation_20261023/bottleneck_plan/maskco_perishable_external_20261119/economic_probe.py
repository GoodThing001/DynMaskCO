"""Cross-fit economic-prior heat calibration on training insertion opportunities."""
import json
from pathlib import Path
import sys
import numpy as np
import pool_search as pool
from static_probe import losses
from calibration import residual_edges
from economic_edges import ALPHAS,economic_prior,mask_residual
HERE=Path(__file__).resolve().parent
def main(i):
    case=json.loads((HERE/"phase.json").read_text())["blocks"][i]["case"]
    study=pool.study;access,core,src,locked=study.setup("SLOT")
    if case not in core.fold_definition(0)["train"]:raise ValueError("Training cases only")
    policy,_=study.load("SLOT",access,core,src,locked,"c205")
    scope=access.Scope("two-customer-economic-prior",frozenset([(0,"train",case)]),frozenset(["infer","initialize","search"]))
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="train")
        alternate=pool.solve(pool.library(),core,locked,None,case,"train",2746,"POOL_D",3)["routes"]
        prior,allowed=economic_prior(task)
        d=np.asarray(task.ins["distance"]);provider=locked.RepairProvider("economics",policy,False)
        rng=np.random.default_rng(261119362);sums={};n=0
        for state in (task.initial(),alternate):
            routes=tuple(tuple(r) for r in state)
            for rep in range(5):
                group=rng.choice(np.arange(1,101),40,replace=False)
                h=provider.heatmap(task,routes,group,"maskco_base",rep)
                space,res=residual_edges(d,h);conditional=mask_residual(prior,allowed,h)
                rankers={str(a):prior+a*conditional for a in ALPHAS}
                rankers.update(distance=-d,legacy=space+.25*res)
                mask=tuple(map(int,rng.choice(group,12,replace=False)))
                for k,v in losses(task,routes,mask,d,rankers).items():sums[k]=sums.get(k,0.)+v
                n+=12
    doc=dict(case=case,customers=n,forwards=provider.forward_calls,feasible_pairs=int(allowed.sum()),loss={k:v/n for k,v in sums.items()})
    (HERE/"out"/f"{i}.json").write_text(json.dumps(doc,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
