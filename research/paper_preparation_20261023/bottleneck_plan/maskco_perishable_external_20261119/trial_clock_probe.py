"""Quantify repeatability of identical timed and count-limited MASKCO operator trials on training data."""
import json
from pathlib import Path
from statistics import median
import sys
import numpy as np
import pool_search as pool
from calibration import residual_edges
HERE=Path(__file__).resolve().parent
def main(i):
    case=json.loads((HERE/"phase.json").read_text())["blocks"][i]["case"]
    study=pool.study;access,core,src,locked=study.setup("SLOT")
    if case not in core.fold_definition(0)["train"]:raise ValueError("Training case only")
    policy,_=study.load("SLOT",access,core,src,locked,"c205");lib=pool.library()
    scope=access.Scope("operator-trial-clock",frozenset([(0,"train",case)]),frozenset(["initialize","infer","search"]))
    rng=np.random.default_rng(275509);records=[]
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="train");inputs=pool.inputs(task);distance=inputs[1]
        distance_heat=np.ascontiguousarray(-distance)
        provider=locked.RepairProvider("trial-clock",policy,False)
        later=pool.solve(lib,core,locked,None,case,"train",2746,"POOL_D",3)["routes"]
        for state in (task.initial(),later):
            routes=tuple(tuple(r) for r in state);start=pool.pack(routes)
            for j in range(8):
                group=rng.choice(np.arange(1,101),40,replace=False).astype(np.int32)
                h=np.ascontiguousarray(provider.heatmap(task,routes,group,"maskco_base",j),dtype=np.float64)
                spatial,residual=residual_edges(distance,h)
                options=[distance_heat,h,np.ascontiguousarray(spatial+.25*residual)]
                seed=int(rng.integers(2**32));modes=[(g,n) for g in range(2) for n in range(3)]
                def run(g,n,cap):
                    st=start.copy();best=start.copy();stats=np.zeros(3,dtype=np.int64)
                    lib.repair_batch(*inputs,options[n],group,len(group),st,best,seed,2. if cap else .04,1.5,j%2,g,cap,stats)
                    if cap and stats[0]!=cap:raise ValueError("Fixed-work trial hit its clock")
                    return int(stats[0]),float(task.evaluate(pool.unpack(best))["net_benefit_U"]),best.tolist(),st.tolist()
                wall=[[run(g,n,0) for g,n in modes] for _ in range(2)]
                cap=[min(wall[0][k][0],wall[1][k][0]) for k in range(6)]
                fixed=[[run(g,n,cap[k]) for k,(g,n) in enumerate(modes)] for _ in range(2)]
                if fixed[0]!=fixed[1]:raise ValueError("Fixed-work repeated trajectories differ")
                def pick(arr):return max(range(6),key=lambda k:(arr[k][1],-((k-j)%6)))
                winners=[pick(a) for a in wall]
                equal=wall[0][winners[0]][2:]==wall[1][winners[1]][2:]
                records.append(dict(winners=winners,chosen_state_equal=equal,individual_work=[[[v[0] for v in a][:3],[v[0] for v in a][3:]] for a in wall],
                                    fixed_winner=pick(fixed[0]),fixed_equal=True))
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,states=records),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
