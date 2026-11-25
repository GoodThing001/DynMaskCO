"""Measure the cold-chain economic effect of instance-only encoding on frozen training geometries."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pool_search as pool
from static_encoder import StaticEncoderHeat
from encoded_mask import load_split
from calibration import residual_edges
HERE=Path(__file__).resolve().parent
def losses(task,routes,mask,dist,rankers):
    from profit_search import remove_customers
    partial=remove_customers(routes,mask)
    out={m:0. for m in rankers}
    for i in mask:
        places=[]
        for r in partial:
            if sum(task.ins["nodes"][j][3] for j in r)+task.ins["nodes"][i][3]>20000+1e-7:continue
            for pos,(a,b) in enumerate(zip((0,)+r,r+(0,))):
                candidate=task.route(r[:pos]+(i,)+r[pos:])
                gain=candidate.net_benefit-task.route(r).net_benefit if candidate.feasible else -np.inf
                places.append((a,b,gain))
        if not places:continue
        a,b=np.array([t[:2] for t in places],dtype=int).T
        gains=np.asarray([t[2] for t in places])
        single=task.route((i,)).net_benefit
        oracle=max(single,gains.max())
        order=np.argsort(dist[a,i]+dist[i,b]-dist[a,b],kind="stable")
        for m,h in rankers.items():
            other=order[6:];merit=h[a,i]+h[i,b]-h[a,b]
            kept=np.concatenate((order[:6],other[np.argsort(-merit[other],kind="stable")[:6]]))
            out[m]+=oracle-max(single,gains[kept].max())
    return out

def main(i):
    b=json.loads((HERE/"phase.json").read_text())["blocks"][i];case=b["case"]
    study=pool.study
    access,core,src,locked=study.setup("SLOT")
    if case not in core.fold_definition(0)["train"]:raise ValueError("Training geometry only")
    model,_=load_split(study,access,core,src,locked,"c205")
    scope=access.Scope("static-encoder-training",frozenset([(0,"train",case)]),frozenset(["infer","initialize","search"]))
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="train")
        alternate=pool.solve(pool.library(),core,locked,None,case,"train",2746,"POOL_D",3)["routes"]
        full=locked.RepairProvider("full",model,False);static=StaticEncoderHeat(model)
        dist=np.asarray(task.ins["distance"])
        rng=np.random.default_rng(261119362)
        clock={"full":0.,"static":0.}
        sums={a:0. for a in ("distance","full","static")};n=0;matches=[]
        for state in (task.initial(),alternate):
            routes=tuple(tuple(r) for r in state)
            for rep in range(5):
                group=rng.choice(np.arange(1,101),40,replace=False)
                rankers={"distance":-dist};tops={}
                order=(("full",full),("static",static))
                for name,provider in (order if rep%2==0 else reversed(order)):
                    start=time.process_time()
                    h=provider.heatmap(task,routes,group,"maskco_base",rep)
                    clock[name]+=time.process_time()-start
                    s,r=residual_edges(dist,h)
                    rankers[name]=s+.25*r
                    tops[name]=np.argsort(-h[group],axis=1,kind="stable")[:,:8]
                matches.extend(len(set(a)&set(b))/8 for a,b in zip(tops["full"],tops["static"]))
                mask=tuple(map(int,rng.choice(group,12,replace=False)))
                vals=losses(task,routes,mask,dist,rankers)
                for name,value in vals.items():sums[name]+=value
                n+=12
    rec=dict(case=case,customers=n,forwards=full.forward_calls,encodes=len(static.cache),cpu=clock,
             shortlist_regret_U={name:v/n for name,v in sums.items()},top8_overlap=float(np.mean(matches)))
    (HERE/"out"/f"{i}.json").write_text(json.dumps(rec,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
