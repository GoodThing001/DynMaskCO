"""Matched order budgets on old distance endpoints with a route-by-route native split audit."""
import json
from pathlib import Path
import sys
import numpy as np
import pool_search as pool
from block_split import bundle,library,proposals,split
HERE=Path(__file__).resolve().parent
def check(fn,inputs,task,route):
    seq=tuple(route[:8])
    score,parts,_=split(fn,inputs,seq)
    optimum=-np.inf
    for cuts in range(1<<(len(seq)-1)):
        idx=[0]+[i for i in range(1,len(seq)) if cuts & (1<<(i-1))]+[len(seq)]
        rs=[task.route(seq[l:r]) for l,r in zip(idx,idx[1:])]
        if all(v.feasible for v in rs):optimum=max(optimum,sum(v.net_benefit for v in rs))
    if abs(optimum-score)>1e-6 or tuple(i for r in parts for i in r)!=seq:raise ValueError("Native split differs from exhaustive partitions")
def main(i):
    b=json.loads((HERE/"phase.json").read_text())["blocks"][i]
    case=b["case"];study=pool.study;fn=library()
    access,core,src,locked=study.setup("SLOT")
    policy,scope=study.load("SLOT",access,core,src,locked,case)
    states=json.loads((HERE/"old_states.json").read_text())[case]
    out={}
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="validation");inputs=pool.inputs(task);dist=inputs[1]
        provider=locked.RepairProvider("split-probe",policy,False)
        for name,raw in states.items():
            tours=tuple(tuple(r) for r in raw);tests=[]
            check(fn,inputs,task,sum(tours,())[:8])
            seen=set()
            for anchor in np.random.default_rng(7339).permutation(len(tours)):
                ids=bundle(tours,dist,int(anchor))
                if not ids or ids in seen:continue
                seen.add(ids)
                parts=[tours[k] for k in ids];group=tuple(i for r in parts for i in r)
                old=sum(task.route(r).net_benefit for r in parts)
                heat=provider.heatmap(task,tours,group,"maskco_base",len(tests))
                obs={}
                for method,weights in (("learned",heat),("distance",-dist)):
                    best=old;winner=parts;calls=0
                    for seq in proposals(parts,weights,9000+len(tests),128):
                        val,routes,work=split(fn,inputs,seq);calls+=work
                        if val>best+1e-7:best=val;winner=routes
                    if sorted(group)!=sorted(i for r in winner for i in r) or abs(best-sum(task.route(r).net_benefit for r in winner))>1e-6:
                        raise ValueError("Fixed group or exact objective differs")
                    obs[method]=dict(gain_U=best-old,route_requests=calls)
                tests.append(obs)
                if len(tests)==8:break
            if not tests:raise ValueError("No complete eligible route bundle")
            out[name]=tests
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,anchors=out,forwards=provider.forward_calls),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
