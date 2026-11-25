"""Register whole-region inputs on five previously evaluated Danish geometries."""
import json
import numpy as np
from route_columns import HERE,BASE,DEST,assemble,read,select
from full_column_opportunity import spare_group

def main():
    examples,_=assemble()
    archive=read(HERE/"archived_states.json");meta=read(HERE/"heldout_sources.json")
    old={(r["case"],r["state"],r["rep"]):r["source_routes"] for r in read(DEST/"regions.json")}
    cfg=read(HERE/"headcapture01.json");blocks=[]
    for ci,c in enumerate(sorted(k for k in archive if k.startswith("dk_"))):
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
        ex=[e for e in examples if e["case"]==c];series=[]
        for si,state in enumerate(("initial","terminal")):
            routes=list(map(tuple,archive[c][state]))
            for rep in range(4):
                m=ex[si*6+rep]["mask"]
                ids,_=select(routes,m,d,nodes[:,3],rep)
                groups=[("neighbour",[routes[i] for i in ids])]
                if rep<2 and [list(r) for r in groups[0][1]]!=old[c,state,rep]:raise ValueError("Earlier region mismatch")
                if state=="terminal" and rep<2:
                    a=next(j for j,r in enumerate(routes) if m[rep] in r)
                    groups.append(("max_spare",spare_group(routes,a,d,nodes[:,3])))
                for kind,parts in groups:
                    clients=sorted(i for r in parts for i in r)
                    rng=np.random.default_rng(771580+ci*32+len(series))
                    mask=clients+rng.choice(sorted(set(range(1,101))-set(clients)),40-len(clients),replace=False).tolist()
                    series.append(dict(state=state,rep=rep,group=kind,parts=parts,mask=mask))
        if len(series)!=10:raise ValueError(c)
        blocks.append(dict(case=c,regions=series))
    if len(blocks)!=5:raise ValueError("Five development geometries required")
    cfg.pop("masks_per_state")
    cfg.update(purpose="whole-region MASKCO embeddings on development geometries",blocks=blocks)
    for name,items in (("jointcapture01",blocks),("jointqual01",[dict(case=blocks[0]["case"],regions=[blocks[0]["regions"][0],blocks[0]["regions"][-1]])])):
        (HERE/(name+".json")).write_text(json.dumps(dict(cfg,blocks=items),indent=2),encoding="utf8")

if __name__=="__main__":main()
