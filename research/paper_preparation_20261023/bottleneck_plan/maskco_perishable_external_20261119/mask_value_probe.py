"""Score matched neural and distance ruin selections by audited economic repair value."""
from pathlib import Path
import ctypes
import json
import sys
import numpy as np
import pool_search as pool
from encoded_mask import covering_masks
from mask_value import moves,merit,order
HERE=Path(__file__).resolve().parent

def main(i):
    cfg=json.loads((HERE/"phase.json").read_text())
    case=cfg["blocks"][i]["case"]
    archived=json.loads((HERE/"old_states.json").read_text())[case]
    study=pool.study
    lib=pool.library()
    lib.repair_fixed.argtypes=[pool.F64,pool.F64,pool.F64,pool.F64,pool.I32,ctypes.c_int,pool.I32,pool.I32]
    lib.repair_fixed.restype=None
    access,core,src,locked=study.setup("SLOT")
    policy,scope=study.load("SLOT",access,core,src,locked,case)
    report={}
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="validation")
        nodes,dist,params=pool.inputs(task);dscore=np.ascontiguousarray(-dist)
        provider=locked.RepairProvider("ruin-value",policy,False)
        for column,raw in archived.items():
            routes=tuple(map(tuple,raw));initial=float(task.evaluate(routes)["net_benefit_U"]);packed=pool.pack(routes)
            rows=[]
            for rep,group in enumerate(covering_masks(1000)):
                heat=provider.heatmap(task,routes,group,"maskco_base",rep)
                choices=moves(group,dist,np.random.default_rng([29471,rep]))
                chosen={a:order(merit(routes,v,group,choices)) for a,v in (("masked",heat),("distance",dscore))}
                gains=[];best_value=initial;best_routes=routes
                for mask in choices:
                    answer=packed.copy()
                    lib.repair_fixed(nodes,dist,params,dscore,np.ascontiguousarray(mask,dtype=np.int32),len(mask),packed,answer)
                    proposal=pool.unpack(answer)
                    value=float(task.evaluate(proposal)["net_benefit_U"])
                    gains.append(max(0.,value-initial))
                    if value>best_value:best_value,best_routes=value,proposal
                checked=task.evaluate(best_routes,audit=True)["net_benefit_U"]
                if abs(checked-best_value)>1e-6:raise ValueError("Selected repair value differs")
                rows.append(dict(oracle=max(gains),common=int(len(set(chosen["masked"])&set(chosen["distance"]))),
                                 **{a:max(gains[k] for k in ix) for a,ix in chosen.items()}))
            report[column]=rows
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,anchors=report),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
