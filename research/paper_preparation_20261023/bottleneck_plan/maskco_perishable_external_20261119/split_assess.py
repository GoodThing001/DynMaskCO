"""Aggregate true block gains with equal weight for geometries and source families."""
from pathlib import Path
from statistics import mean
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"maskco_heatmap_calibration_20261119"))
from phase_plan import read
from mechanism_summary import source_means
import json
def main():
    stage="splitprobe01";cfg=read(HERE/(stage+".json"));folder=HERE/"results"/stage
    state=read(folder/"status.json")
    if not state["end"] or state["failed"] or state["complete"]!=len(cfg["blocks"]):raise ValueError("Incomplete panel")
    meta=read(HERE.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    out={}
    for anchor in ("initial_routes","routes"):
        cases={}
        for i,b in enumerate(cfg["blocks"]):
            r=read(folder/f"out/{i}.json");rows=r["anchors"][anchor]
            if r["case"]!=b["case"] or not rows:raise ValueError("Incomplete anchor")
            vals={m:mean(v[m]["gain_U"] for v in rows) for m in ("learned","distance")}
            vals["delta_U"]=vals["learned"]-vals["distance"]
            cases[r["case"]]=vals
        out[anchor]=source_means(cases,meta)
    ds=[v["delta_U"] for v in out["routes"].values()]
    out["advance"]=sum(x>1e-6 for x in ds)>=2 and mean(ds)>1e-6
    (folder/"analysis.json").write_text(json.dumps(out,indent=2));print(out)
if __name__=="__main__":main()
