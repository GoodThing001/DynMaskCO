"""Separate pair savings, directional feasibility and directed value in a masked insertion shortlist."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import numpy as np
from domain_profile import core
from preflight_publication_model_v1 import nearest_neighbour_witness
from economic_edges import economic_prior
HERE=Path(__file__).resolve().parent

def pair_scores(task):
    prior,valid=economic_prior(task)
    symmetric=np.where(valid & valid.T,(prior+prior.T)/2,np.maximum(prior,prior.T))
    directed_allowed=np.where(valid,symmetric,prior)
    return {"distance":-np.asarray(task.ins["distance"]),"undirected_savings":symmetric,
            "directed_feasibility":directed_allowed,"directed_savings":prior}

def ablate(task,initial,seed):
    scores=pair_scores(task);dist=-scores["distance"]
    samples=defaultdict(list);rng=np.random.default_rng(seed)
    for rep in range(10):
        mask=rng.choice(np.arange(1,101),12,replace=False)
        hidden=set(mask);partial=[tuple(i for i in r if i not in hidden) for r in initial]
        partial=[r for r in partial if r]
        for i in mask:
            i=int(i)
            slots=[];values=[]
            for r in partial:
                if sum(task.ins["nodes"][j][3] for j in r)+task.ins["nodes"][i][3]>20000+1e-7:continue
                base=task.route(r).net_benefit
                for pos,(a,b) in enumerate(zip((0,)+r,r+(0,))):
                    trial=task.route(r[:pos]+(i,)+r[pos:])
                    slots.append((a,b));values.append(trial.net_benefit-base if trial.feasible else -np.inf)
            single=task.route((i,)).net_benefit
            if not slots:
                for key in scores:samples[key].append(0.)
                continue
            a,b=np.asarray(slots,dtype=int).T;u=np.asarray(values)
            best=max(single,float(max(u)));rank=np.argsort(dist[a,i]+dist[i,b]-dist[a,b],kind="stable")
            for key,h in scores.items():
                val=h[a,i]+h[i,b]-h[a,b]
                other=rank[6:];order=other[np.argsort(-val[other],kind="stable")[:6]]
                picked=np.r_[rank[:6],order]
                samples[key].append(best-max(single,float(max(u[picked]))))
    if any(len(ds)!=120 for ds in samples.values()):raise ValueError("Missing insertion opportunities")
    return {m:mean(ds) for m,ds in samples.items()}

def main():
    cfg=core.load_config();out={}
    source=json.loads((HERE/"economicprobe01.json").read_text());allowed=set(core.fold_definition(0)["train"])
    for b in source["blocks"]:
        case=b["case"]
        if case not in allowed:raise ValueError("Only fold-zero training data are admitted")
        ins=core.load_original(case,cfg);model=core.PublicationModel(ins,cfg,"main_Product1",.8)
        routes=tuple(tuple(r.route) for r in nearest_neighbour_witness(model,True))
        task=SimpleNamespace(route=lru_cache(maxsize=30000)(model.route),ins=ins)
        out[case]=ablate(task,routes,261119503)
    data={}
    for key in ("R","RC"):
        vals=[v for c,v in out.items() if c.startswith("rc")==(key=="RC")]
        data[key]={m:mean(v[m] for v in vals) for m in vals[0]}
    path=HERE/"orientation_ablation.json"
    path.write_text(json.dumps(dict(examples=12*120,anchor="audited common initializer",per_case=out,training_geometries=data),indent=2))
    print(data)
if __name__=="__main__":main()
