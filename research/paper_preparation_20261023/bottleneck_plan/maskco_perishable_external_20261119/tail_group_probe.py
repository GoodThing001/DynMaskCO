"""Compare learned and distance selection of masked route pairs for a two-route tail exchange."""
from itertools import combinations
from pathlib import Path
import json
import sys
import time
import numpy as np
import external_common as adapter
import pool_search as pool
from heat_placebo import remap

HERE=Path(__file__).resolve().parent


def route_pairs(task,routes,group,weights):
    nodes=np.asarray(task.ins["nodes"])
    output=[]
    for i,j in combinations(group,2):
        left,right=routes[i],routes[j]
        scores=np.full(len(weights),-np.inf)
        sums=[np.r_[0.,np.cumsum(nodes[list(r),3])] for r in (left,right)]
        baseline=sum(task.route(r).net_benefit for r in (left,right))
        winner=0.;solution=None;cuts=0
        for u in range(len(left)+1):
            for v in range(len(right)+1):
                if ((u==0 and v==0) or (u==len(left) and v==len(right))
                        or u+len(right)-v==0 or v+len(left)-u==0):continue
                lm,rm=sums
                if max(lm[u]+rm[-1]-rm[v],rm[v]+lm[-1]-lm[u])>20000.+1e-7:continue
                a=left[u-1] if u else 0
                b=left[u] if u<len(left) else 0
                c=right[v-1] if v else 0
                d=right[v] if v<len(right) else 0
                for k,h in enumerate(weights):
                    scores[k]=max(scores[k],h[a,d]+h[c,b]-h[a,b]-h[c,d])
                cuts+=1
                first,second=left[:u]+right[v:],right[:v]+left[u:]
                x,y=task.route(first),task.route(second)
                if x.feasible and y.feasible:
                    gain=x.net_benefit+y.net_benefit-baseline
                    if gain>winner+1.e-7:winner,solution=gain,(first,second)
        if cuts:
            output.append(dict(pair=(i,j),rank=scores.tolist(),gain=winner,cuts=cuts,solution=solution))
    return output


def main(index):
    plan=json.loads((HERE/"phase.json").read_text());case=plan["blocks"][index]["case"]
    archived=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=adapter.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    single_cpu_guard()
    model,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("masked-route-pair-development",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    started=time.process_time(); result=[]
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");dist=np.asarray(task.ins["distance"])
        provider=locked.RepairProvider("masked-routes",model,False)
        rng=np.random.default_rng(203102125)
        for state in ("initial","terminal"):
            routes=tuple(map(tuple,archived[state]))
            if state=="initial" and routes!=tuple(map(tuple,task.initial())):raise ValueError("Common initializer differs")
            full=task.evaluate(routes,audit=True)["net_benefit_U"]
            for t in range(plan["masks_per_state"]):
                anchor=int(rng.integers(len(routes)))
                ids=sorted(range(len(routes)),key=lambda j:(j!=anchor,float(np.min(dist[np.ix_(routes[anchor],routes[j])])),j))
                selected=[];hidden=[]
                for j in ids:
                    if len(routes[j])+len(hidden)<=40:
                        selected.append(j);hidden.extend(routes[j])
                rest=np.setdiff1d(np.arange(1,101),hidden)
                group=rng.permutation(np.r_[np.asarray(hidden,dtype=int),rng.choice(rest,40-len(hidden),replace=False)])
                raw=provider.heatmap(task,routes,group,"maskco_base",t)
                actual,null=remap(raw,dist,routes,group,rng)
                pairs=route_pairs(task,routes,selected,(-dist,actual,null))
                base=sorted(range(len(pairs)),key=lambda k:-pairs[k]["rank"][0])[:2]
                gains={}
                for rank,name in enumerate(("distance","real","shuffled")):
                    remaining=sorted((k for k in range(len(pairs)) if k not in base),key=lambda k:-pairs[k]["rank"][rank])
                    picked=base+remaining[:2]
                    gains[name]=max((pairs[k]["gain"] for k in picked),default=0.)
                gains["oracle"]=max((row["gain"] for row in pairs),default=0.)
                if gains["oracle"]>0:
                    best=max(pairs,key=lambda r:r["gain"])
                    proposal=[r for j,r in enumerate(routes) if j not in best["pair"]]+list(best["solution"])
                    if abs(task.evaluate(proposal,audit=True)["net_benefit_U"]-full-gains["oracle"])>1e-6:
                        raise ValueError("Exchange differs from independent route scheduling")
                result.append(dict(state=state,t=t,pairs=len(pairs),cuts=sum(r["cuts"] for r in pairs),gains=gains))
    (HERE/"out"/f"{index}.json").write_text(json.dumps(dict(case=case,rows=result,forwards=provider.forward_calls,
        requests=sum(task.route.cache_info()[:2]),cpu_s=time.process_time()-started),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
