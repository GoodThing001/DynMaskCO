"""Measure whether a candidate mask admits alternative complete regional repairs from other masks."""
from collections import defaultdict
from statistics import mean
import json
from mask_utility_data import HERE,OUT,read

def signature(routes,hidden):
    return tuple(sorted(t for r in routes if (t:=tuple(i for i in r if i not in hidden))))

def gather(i):
    study=read(OUT/f"utility_{i}.json");libraries=defaultdict(dict);queries=defaultdict(set)
    for r in study["rows"]:
        parts=tuple(sorted(map(tuple,r["parts"])))
        key=(r["state"],parts);pool=libraries[key];pool[parts]=0.
        for z in r["candidates"]:
            sol=tuple(sorted(map(tuple,z["chosen"])))
            if sol in pool and abs(pool[sol]-z["gain"])>1.e-6:raise ValueError("Inconsistent joint value")
            pool[sol]=z["gain"]
            queries[key].add(tuple(z["mask"]))
    return study["case"],libraries,queries

def main():
    rows=[]
    for i in range(5):
        case,libs,queries=gather(i)
        for (state,parts),lib in libs.items():
            for m in sorted(queries[state,parts]):
                hidden=set(m);sk=signature(parts,hidden)
                other=[v for q,v in lib.items() if q!=parts and signature(q,hidden)==sk and v>1.e-6]
                rows.append(dict(case=case,state=state,accepted=len(other),best=max([0.]+other)))
    info={}
    for state in ("initial","terminal"):
        per_case=[]
        for c in sorted({r["case"] for r in rows}):
            q=[r for r in rows if r["case"]==c and r["state"]==state]
            per_case.append(dict(case=c,contexts=len(q),one=sum(r["accepted"]>=1 for r in q),two=sum(r["accepted"]>=2 for r in q),best=mean(r["best"] for r in q)))
        info[state]=per_case
    (OUT/"conditional_opportunity.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(info))
if __name__=="__main__":main()
