"""Quantify the cost of route orientation on the twelve registered training instances."""
from itertools import combinations
import json
from pathlib import Path
from statistics import mean,median
from domain_profile import core
HERE=Path(__file__).resolve().parent
def main():
    cfg=core.load_config()
    reg=json.loads((HERE/"economicprobe01.json").read_text())
    allowed=set(core.fold_definition(0)["train"]);by={}
    for row in reg["blocks"]:
        case=row["case"]
        if case not in allowed:raise ValueError("Source must belong to the first fold training set")
        instance=core.load_original(case,cfg)
        model=core.PublicationModel(instance,cfg,"main_Product1",.8)
        dif=[];one=0;neither=0
        for i,j in combinations(range(1,101),2):
            a,b=model.route((i,j)),model.route((j,i))
            if a.feasible and b.feasible:dif.append(abs(a.net_benefit-b.net_benefit))
            elif a.feasible or b.feasible:one+=1
            else:neither+=1
        by[case]=dict(both_feasible=len(dif),one_direction=one,neither=neither,
                      mean_directional_gap_U=mean(dif) if dif else None,median_directional_gap_U=median(dif) if dif else None)
    groups={}
    for group in ("R","RC"):
        cases=[r for c,r in by.items() if c.startswith("rc")== (group=="RC")]
        groups[group]={k:mean(r[k] for r in cases if r[k] is not None) for k in cases[0]}
        groups[group]["cases_with_both"]=sum(r["both_feasible"]>0 for r in cases)
        groups[group]["share_one_given_any"]=sum(r["one_direction"] for r in cases)/sum(r["one_direction"]+r["both_feasible"] for r in cases)
    (HERE/"directional_evidence.json").write_text(json.dumps(dict(training_cases=by,summary=groups),indent=2))
    print(groups)
if __name__=="__main__":main()
