"""Compare matched geometric and masked-neural partner proposals on prior development solutions."""
import json
from pathlib import Path
import sys
import numpy as np
import pool_search as pool
from edge_moves import evaluate_pairs,disjoint_gain
from encoded_mask import covering_masks
HERE=Path(__file__).resolve().parent

def pairs(ranking,group,existing):
    selected=set()
    for i in map(int,group):
        available=[int(j) for j in group if j!=i and tuple(sorted((i,int(j)))) not in existing]
        if available:
            best=min(available,key=lambda j:(ranking[i,j]+ranking[j,i],j))
            selected.add(tuple(sorted((i,best))))
    return selected

def rank(heat):
    return np.argsort(np.argsort(-heat,axis=1,kind="stable"),axis=1,kind="stable")

def main(i):
    b=json.loads((HERE/"phase.json").read_text())["blocks"][i]
    case=b["case"]
    study=pool.study
    access,core,src,locked=study.setup("SLOT")
    policy,scope=study.load("SLOT",access,core,src,locked,case)
    archived=json.loads((HERE/"old_states.json").read_text())[case]
    res={}
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="validation")
        dist=rank(-np.asarray(task.ins["distance"]))
        for anchor,routes in archived.items():
            existing={tuple(sorted((u,v))) for r in routes for u,v in zip(r,r[1:])}
            gain=evaluate_pairs(task,routes)
            provider=locked.RepairProvider("edge-probe",policy,False)
            readings=[]
            for k,group in enumerate(covering_masks(1000)):
                heat=rank(provider.heatmap(task,routes,group,"maskco_base",k))
                row={}
                for name,ranks in (("maskco",heat),("distance",dist)):
                    selected=pairs(ranks,group,existing)
                    moves=[gain(u,v) for u,v in sorted(selected)]
                    total,count=disjoint_gain(moves)
                    row[name]=dict(sum_disjoint_U=total,positive_disjoint_moves=count,best_U=max(v for v,ids in moves),pairs=len(selected))
                readings.append(row)
            res[anchor]=readings
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,anchors=res),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
