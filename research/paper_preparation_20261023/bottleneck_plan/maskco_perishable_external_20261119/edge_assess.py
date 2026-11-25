"""Aggregate matched old development states by source and customer geometry."""
import json
from pathlib import Path
from statistics import mean
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"maskco_heatmap_calibration_20261119"))
from assess import describe
def read(f):return json.loads(f.read_text(encoding="utf8"))
def main():
    m=read(HERE/"edgeprobe01.json")
    out=HERE/"results/edgeprobe01"
    status=read(out/"status.json")
    if status["failed"] or status["complete"]!=len(m["blocks"]) or not status["end"]:raise ValueError("Incomplete edge diagnosis")
    meta=read(HERE.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    summary={}
    for anchor in ("initial_routes","routes"):
        result={}
        for i,b in enumerate(m["blocks"]):
            data=read(out/f"out/{i}.json")
            case=b["case"]
            if case!=data["case"] or len(data["anchors"][anchor])!=10:raise ValueError("Masked queries missing")
            result[case]={m:mean(x[m]["sum_disjoint_U"] for x in data["anchors"][anchor]) for m in ("maskco","distance")}
        summary[anchor]={}
        for src in sorted({meta[c]["dataset"] for c in result}):
            a={m:describe({(c,1000):vals[m] for c,vals in result.items() if meta[c]["dataset"]==src},meta)["mean"] for m in ("maskco","distance")}
            a["delta_U"]=a["maskco"]-a["distance"]
            summary[anchor][src]=a
    ds=[v["delta_U"] for v in summary["routes"].values()]
    summary["advance"]=sum(x>1.e-6 for x in ds)>=2 and mean(ds)>1.e-6
    (out/"analysis.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(summary)
if __name__=="__main__":main()
