"""Register two new neighbourhoods on a different archived distance seed."""
import json
import numpy as np
from route_columns import HERE,read,select

def main():
    original=read(HERE/"archived_states.json");meta=read(HERE/"heldout_sources.json");archives={}
    sources=read(HERE/"replicate01.json")["blocks"]
    cfg=read(HERE/"jointcapture01.json")
    for ci,b in enumerate(cfg["blocks"]):
        c=b["case"];idx,= (k for k,r in enumerate(sources) if r["case"]==c and r["seed"]==32003)
        v=read(HERE/f"results/replicate01/out/{idx}.json")["arms"]["HYBRID_D"]
        if (v["status"],v["method"],v["case"],v["seed"])!=("AUDITED","HYBRID_D",c,32003):raise ValueError(c)
        routes=list(map(tuple,v["routes"]));archives[c]=dict(initial=original[c]["initial"],terminal=v["routes"])
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2)
        regs=[]
        for rep in (0,1):
            rng=np.random.default_rng(341116+ci*32+rep)
            mask=rng.choice(np.arange(1,101),40,replace=False)
            js,_=select(routes,mask,d,nodes[:,3],rep);parts=[routes[j] for j in js]
            ids=sorted(i for r in parts for i in r)
            full=ids+rng.choice(sorted(set(range(1,101))-set(ids)),40-len(ids),replace=False).tolist()
            regs.append(dict(state="terminal",rep=rep,group="neighbour",parts=parts,mask=full))
        b["regions"]=regs
    if len(archives)!=5:raise ValueError("Missing Danish source")
    (HERE/"group_parent_states.json").write_text(json.dumps(archives),encoding="utf8")
    cfg["file_sources"]["archived_states.json"]="group_parent_states.json"
    cfg["purpose"]="whole-route groups on another archived development seed"
    (HERE/"groupcapture01.json").write_text(json.dumps(cfg,indent=2),encoding="utf8")

if __name__=="__main__":main()
