"""Collect insertion opportunities from frozen MASKCO on the two original training geometries."""
from pathlib import Path
import json
import sys
import numpy as np
import pool_search as pool
from economic_edges import economic_prior
from directional_head import edge_features,prior_scores
HERE=Path(__file__).resolve().parent

def main(index):
    case=json.loads((HERE/"phase.json").read_text())["blocks"][index]["case"]
    study=pool.study;access,core,src,locked=study.setup("SLOT")
    if case not in core.fold_definition(0)["train"]:raise ValueError("Only original training geometries")
    model,_=study.load("SLOT",access,core,src,locked,"c205")
    scope=access.Scope("directed-readout-training",frozenset([(0,"train",case)]),frozenset(["infer","initialize","search"]))
    lengths=[];offset=[];neural=[];spatial=[];gains=[];anchors=[]
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="train");nodes=np.asarray(task.ins["nodes"]);dist=np.asarray(task.ins["distance"])
        prior,allowed=economic_prior(task);distance_features=edge_features(nodes,dist,-dist)
        rng=np.random.default_rng(261119815)
        alternate=pool.solve(pool.library(),core,locked,None,case,"train",2746,"POOL_D",3)["routes"]
        provider=locked.RepairProvider("directional-readout",model,False)
        for raw in (task.initial(),alternate):
            routes=tuple(map(tuple,raw))
            for rep in range(5):
                group=rng.choice(np.arange(1,101),40,replace=False)
                hidden=set(map(int,rng.choice(group,12,replace=False)))
                heat=provider.heatmap(task,routes,group,"maskco_base",rep)
                hf=edge_features(nodes,dist,heat)
                partial=[tuple(v for v in r if v not in hidden) for r in routes];partial=[r for r in partial if r]
                for i in sorted(hidden):
                    ab=[];values=[]
                    for r in partial:
                        if nodes[list(r),3].sum()+nodes[i,3]>20000+1e-7:continue
                        base=task.route(r).net_benefit
                        for j,(a,b) in enumerate(zip((0,)+r,r+(0,))):
                            tr=task.route(r[:j]+(i,)+r[j:])
                            ab.append([a,b]);values.append(tr.net_benefit-base if tr.feasible else -np.inf)
                    if len(ab)<=6:
                        lengths.append(0);continue
                    a,b=np.array(ab).T;v=np.array(values)
                    order=np.argsort(dist[a,i]+dist[i,b]-dist[a,b],kind="stable");keep=order[6:]
                    anchor=max(task.route((i,)).net_benefit,v[order[:6]].max())
                    improvements=np.maximum(v[keep]-anchor,0.)
                    # The six protected distance choices and the singleton have already been accounted for.
                    lengths.append(len(keep));gains.extend(improvements);offset.extend(prior_scores(a[keep],b[keep],i,prior))
                    for f,record in ((hf,neural),(distance_features,spatial)):
                        record.extend(f[a[keep],i]+f[i,b[keep]]-f[a[keep],b[keep]])
    if len(lengths)!=120 or provider.forward_calls!=10:raise ValueError("Training opportunity count differs")
    target=HERE/"out"
    np.savez_compressed(target/f"{index}.npz",lengths=np.array(lengths,np.int32),offset=np.array(offset,np.float32),
                        gains=np.array(gains,np.float32),neural=np.asarray(neural,np.float32),spatial=np.asarray(spatial,np.float32))
    (target/f"{index}.json").write_text(json.dumps(dict(case=case,opportunities=len(lengths),slots=sum(lengths),forwards=provider.forward_calls)))
if __name__=="__main__":main(int(sys.argv[1]))
