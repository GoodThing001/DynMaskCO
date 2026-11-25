"""Prepare and analyze the matched destroy-only experiment on archived development solutions."""
from pathlib import Path
from statistics import mean
import json
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save
from mechanism_summary import source_means

def plan():
    q=read(HERE/"edgeprobe01.json")
    q["files"]=[n for n in q["files"] if n not in ("edge_probe.py","edge_moves.py")]
    q["files"]+=["mask_value.py","mask_value_probe.py","masked_repair.cpp"]
    q["file_sources"]["masked_repair.cpp"]="../"+PREV.name+"/lock/masked_repair.cpp"
    q.update(worker="mask_value_probe.py",purpose="Paired strictly distance repair after neural or distance ruin selection",block_timeout_s=480)
    save(HERE/"maskvalue01.json",q)

def report():
    q=read(HERE/"maskvalue01.json")
    folder=HERE/"results/maskvalue01"
    t=read(folder/"status.json")
    if not t["end"] or t["failed"] or t["complete"]!=8:raise ValueError("Incomplete paired diagnosis")
    meta=read(HERE.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    summary={}
    for anchor in ("initial_routes","routes"):
        cases={}
        for i,b in enumerate(q["blocks"]):
            r=read(folder/f"out/{i}.json");rows=r["anchors"][anchor]
            if b["case"]!=r["case"] or len(rows)!=10:raise ValueError("Study grid differs")
            x={m:mean(v[m] for v in rows) for m in ("masked","distance","oracle","common")}
            x["delta_U"]=x["masked"]-x["distance"]
            cases[b["case"]]=x
        summary[anchor]=source_means(cases,meta)
    diff=[v["delta_U"] for v in summary["routes"].values()]
    summary["advance"]=sum(x>1e-6 for x in diff)>=2 and mean(diff)>1e-6
    (folder/"analysis.json").write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=="__main__":plan() if sys.argv[1]=="plan" else report()
