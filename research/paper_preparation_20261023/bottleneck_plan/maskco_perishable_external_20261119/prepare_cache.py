"""Prepare the fixed technical and development panels on historical cases."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import paired,read,save

def main():
    common=["runner.py","cached_search.py","masked_repair.cpp","calibration.py","calibrated.json","encoded_mask.py"]
    sources={n:"../"+PREV.name+"/lock/"+n for n in ("masked_repair.cpp","calibrated.json","calibration.py")}
    sources["runner.py"]="development_runner.py"
    def write(name,purpose,seconds,blocks,timeout):
        save(HERE/(name+".json"),dict(worker="cached_search.py",files=common,file_sources=sources,purpose=purpose,seconds=seconds,blocks=blocks,block_timeout_s=timeout))
    arms=["CACHEHYBRID","CACHEHYBRID_D","CYCLE_FULL"]
    cases={b["case"] for b in read(PREV/"screen07.json")["blocks"]}
    write("cachequal01","training-only software qualification",24,[dict(case="r101",role="train",seed=1000,arms=arms)],420)
    write("cachescreen01","development screening",60,paired(cases,[1000,1001],arms),450)
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    write("cachedev01","full development",60,paired(full,[1000,1001,1002],arms[:2]+["DIST","DIST_SLOT"]),500)
if __name__=="__main__":main()
