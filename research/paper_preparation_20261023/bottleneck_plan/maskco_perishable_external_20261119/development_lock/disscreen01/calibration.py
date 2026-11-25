"""Distance-conditional residual of masked edge ranks."""
import numpy as np

ALPHAS=(0.,.25,.5,1.,2.,4.,8.,16.)

def residual_edges(distance, heat):
    d=np.asarray(distance,dtype=float)
    learned=np.array(heat,dtype=float,copy=True)
    spatial=-d.copy()
    np.fill_diagonal(learned,-np.inf)
    np.fill_diagonal(spatial,-np.inf)
    def ranks(weights):
        return 1+np.argsort(np.argsort(-weights,axis=1,kind="stable"),axis=1,kind="stable")
    delta=np.clip(np.log(ranks(spatial)/ranks(learned)),-3.,3.)
    scale=float(np.median(np.partition(d,8,axis=1)[:,8]))
    if scale<=0:
        scale=float(np.median(d[d>0]))
    return np.ascontiguousarray(-d/scale),np.ascontiguousarray(delta)

def loss_by_alpha(task,routes,mask,spatial,residual):
    from profit_search import remove_customers
    partial=remove_customers(routes,mask)
    d=np.asarray(task.ins["distance"])
    edges=[(r,pos,a,b) for r in partial for pos,(a,b) in enumerate(zip((0,)+r,r+(0,)))]
    losses=[]
    for i in mask:
        singleton=task.route((i,)).net_benefit
        slots=[];values=[]
        for route,pos,a,b in edges:
            if sum(task.ins["nodes"][j][3] for j in route)+task.ins["nodes"][i][3]>20000+1e-7:
                continue
            value=task.route(route[:pos]+(i,)+route[pos:])
            values.append(value.net_benefit-task.route(route).net_benefit if value.feasible else -np.inf)
            slots.append((a,b))
        if not slots:
            losses.append([0.]*len(ALPHAS))
            continue
        a,b=np.asarray(slots).T
        mileage=d[a,i]+d[i,b]-d[a,b]
        order=np.argsort(mileage,kind="stable")
        protected,remaining=order[:6],order[6:]
        base=spatial[a,i]+spatial[i,b]-spatial[a,b]
        change=residual[a,i]+residual[i,b]-residual[a,b]
        values=np.asarray(values)
        oracle=max(singleton,np.max(values))
        row=[]
        for alpha in ALPHAS:
            ranked=remaining[np.argsort(-(base+alpha*change)[remaining],kind="stable")[:6]]
            v=max(singleton,np.max(values[np.concatenate((protected,ranked))]))
            row.append(oracle-v)
        losses.append(row)
    return np.mean(losses,axis=0)

def collect(i):
    import json
    from pathlib import Path
    import sys
    from statistics import mean
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    import pool_search
    study=pool_search.study
    block=json.loads((here/"phase.json").read_text())["blocks"][i]
    case=block["case"]
    access,core,source,locked=study.setup("SLOT")
    model,_=study.load("SLOT",access,core,source,locked,"c205")
    if case not in core.fold_definition(0)["train"]:raise ValueError("Training geometry only")
    scope=access.Scope("economic-calibration-train",frozenset([(0,"train",case)]),frozenset(["infer","initialize","search"]))
    lib=pool_search.library()
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="train");init=task.initial()
        improved=pool_search.solve(lib,core,locked,None,case,"train",2746,"POOL_D",3)
        provider=locked.RepairProvider("calibration",model,False)
        loss=[];rng=np.random.default_rng(202611194)
        for state in (init,tuple(tuple(r) for r in improved["routes"])):
            anchor=[]
            for j in range(5):
                hidden=rng.choice(np.arange(1,101),size=40,replace=False)
                heat=provider.heatmap(task,state,hidden,"maskco_base",j)
                spatial,residual=residual_edges(task.ins["distance"],heat)
                for _ in range(3):
                    mask=tuple(map(int,rng.choice(hidden,size=12,replace=False)))
                    anchor.append(loss_by_alpha(task,state,mask,spatial,residual))
            loss.append(np.mean(anchor,axis=0).tolist())
    obj=dict(case=case,forward_calls=provider.forward_calls,forward_s=provider.forward_seconds,
             initial=loss[0],improved=loss[1],loss=np.mean(loss,axis=0).tolist(),customers=360,alphas=ALPHAS)
    (here/"out"/f"{i}.json").write_text(json.dumps(obj,allow_nan=False))
if __name__=="__main__":
    import sys
    collect(int(sys.argv[1]))
