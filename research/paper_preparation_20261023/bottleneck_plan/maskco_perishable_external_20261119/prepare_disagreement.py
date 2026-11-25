"""Prepare training acceptance and the registered validation subset for disagreement masks."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
def main():
    info=read(HERE/"parent_disagreement.json")["sources"]
    if len(info)!=3 or min(v["cover_improvement_U"] for v in info.values())<=0:raise ValueError("Missing cross-parent economic opportunity")
    fs=["runner.py","cached_search.py","encoded_mask.py","disagreement.py","masked_repair.cpp","calibration.py","calibrated.json"]
    paths={n:"../"+PREV.name+"/lock/"+n for n in ("masked_repair.cpp","calibration.py","calibrated.json")}
    paths["runner.py"]="development_runner.py"
    arms=["DISAGREE","DISAGREE_D","HYBRID"]
    def make(name,purpose,seconds,blocks,timeout):
        save(HERE/(name+".json"),dict(purpose=purpose,worker="cached_search.py",files=fs,file_sources=paths,seconds=seconds,blocks=blocks,block_timeout_s=timeout))
    small={b["case"] for b in read(PREV/"screen07.json")["blocks"]}
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    make("disqual01","training-only solver acceptance",15,[dict(case="r101",role="train",seed=1000,arms=arms)],300)
    make("disscreen01","development screening",60,paired(small,[1000,1001],arms),400)
    make("disdev01","full development",60,paired(full,[1000,1001,1002],arms[:2]+["DIST","DIST_SLOT"]),500)
if __name__=="__main__":main()
