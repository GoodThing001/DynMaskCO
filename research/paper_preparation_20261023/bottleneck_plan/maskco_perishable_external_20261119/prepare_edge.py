"""Select the eight prior screening cases and their registered distance endpoints."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
def read(f):return json.loads(f.read_text(encoding="utf8"))
def write(f,r):
    with f.open("x",encoding="utf8") as out:json.dump(r,out,indent=2)
def main():
    cases=sorted({b["case"] for b in read(PREV/"screen07.json")["blocks"]})
    source=read(PREV/"dev03.json")
    states={}
    for i,b in enumerate(source["blocks"]):
        if b["case"] in cases and b["seed"]==1000:
            c=read(PREV/f"results/dev03/out/{i}.json")["arms"]["HYBRID_D"]
            if c["status"]!="AUDITED":raise ValueError("Missing audited historical route")
            states[b["case"]]={k:c[k] for k in ("initial_routes","routes")}
    if set(states)!=set(cases):raise ValueError("Incomplete source states")
    write(HERE/"old_states.json",states)
    paths={"runner.py":"development_runner.py","pool_search.py":"../"+PREV.name+"/lock/pool_search.py"}
    write(HERE/"edgeprobe01.json",dict(purpose="development-only paired direct-edge opportunity diagnosis",
          worker="edge_probe.py",files=["runner.py","pool_search.py","edge_probe.py","edge_moves.py","encoded_mask.py","old_states.json"],
          file_sources=paths,blocks=[dict(case=c) for c in cases],block_timeout_s=420))
if __name__=="__main__":main()
