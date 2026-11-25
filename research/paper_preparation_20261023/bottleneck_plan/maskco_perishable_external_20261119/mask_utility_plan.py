"""Select the previously unused nearby anchors for whole-mask utility development."""
import json
import numpy as np
from route_columns import HERE,assemble,read,select

def main():
    prior=read(HERE/"jointcapture01.json");ex,_=assemble()
    meta=read(HERE/"heldout_sources.json");archives=read(HERE/"archived_states.json")
    for ci,b in enumerate(prior["blocks"]):
        c=b["case"];examples=[e for e in ex if e["case"]==c]
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);series=[]
        for si,state in enumerate(("initial","terminal")):
            routes=list(map(tuple,archives[c][state]))
            for rep in (4,5):
                ids,_=select(routes,examples[si*6+rep]["mask"],d,nodes[:,3],rep)
                parts=[routes[i] for i in ids];inside=sorted(i for r in parts for i in r)
                rng=np.random.default_rng(760041+16*ci+len(series))
                mask=inside+rng.choice(sorted(set(range(1,101))-set(inside)),40-len(inside),replace=False).tolist()
                series.append(dict(state=state,rep=rep,group="neighbour",parts=parts,mask=mask))
        b["regions"]=series
    prior["purpose"]="prospective whole-mask utility development anchors"
    (HERE/"utilitycapture01.json").write_text(json.dumps(prior,indent=2),encoding="utf8")
if __name__=="__main__":main()
