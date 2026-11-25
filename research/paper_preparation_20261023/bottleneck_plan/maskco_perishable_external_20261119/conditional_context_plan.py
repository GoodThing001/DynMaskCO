"""Define the same regional candidate fleets under a coarse and a precise input mask."""
from pathlib import Path
import json
import numpy as np
from conditional_route_library import HERE,OUT,read,gather,signature

def main():
    prior=read(HERE/"jointcapture01.json");blocks=[];all_libs=[]
    for ci in range(5):
        c,libs,queries=gather(ci);case_regions=[]
        for si,state in enumerate(("initial","terminal")):
            options=[]
            for (s,parts),lib in libs.items():
                if s!=state:continue
                for m in sorted(queries[s,parts]):
                    sk=signature(parts,set(m));feasible=[(q,v) for q,v in sorted(lib.items()) if signature(q,set(m))==sk]
                    if len({round(v,6) for _,v in feasible if v>1.e-6})>=2:options.append((parts,m,feasible))
            rng=np.random.default_rng(642118+ci*4+si)
            for j in rng.permutation(len(options))[:24]:
                parts,m,feasible=options[j];ids=sorted(i for r in parts for i in r);uid=len(case_regions)
                outside=rng.permutation(sorted(set(range(1,101))-set(ids))).tolist()
                if min(len(m),len(ids))<1 or len(ids)>40:raise ValueError("Region mask")
                case_regions.append(dict(state=state,ids=ids,hidden=list(m),broad=ids+outside[:40-len(ids)],narrow=list(m)+outside[:40-len(m)]))
                all_libs.append(dict(case=c,context=uid,state=state,parts=parts,hidden=m,choices=[dict(routes=q,gain=v) for q,v in feasible]))
        blocks.append(dict(case=c,regions=case_regions))
    (HERE/"conditional_choice_library.json").write_text(json.dumps(all_libs),encoding="utf8")
    prior.update(blocks=blocks,worker="conditional_context_worker.py",purpose="conditional skeleton paired representation",block_timeout_s=420)
    prior["files"].append("conditional_context_worker.py")
    for phase,b in (("skeletonqual01",[dict(case=blocks[0]["case"],regions=blocks[0]["regions"][:1])]),("skeletoncontext01",blocks)):
        (HERE/(phase+".json")).write_text(json.dumps(dict(prior,blocks=b),indent=2),encoding="utf8")
    print([len(b["regions"]) for b in blocks])
if __name__=="__main__":main()
