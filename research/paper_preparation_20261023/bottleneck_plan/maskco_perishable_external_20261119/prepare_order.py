"""Prespecify the masked visit-order reconstruction on historical development cases."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
def main():
    if not read(HERE/"results/splitprobe01/analysis.json")["advance"]:raise ValueError("No matched route-level opportunity")
    fs=["runner.py","block_search.py","block_split.py","split_order.cpp","masked_repair.cpp","calibration.py","calibrated.json"]
    paths={n:"../"+PREV.name+"/lock/"+n for n in ("masked_repair.cpp","calibration.py","calibrated.json")}
    paths["runner.py"]="development_runner.py"
    arms=["ORDER_SPLIT","ORDER_SPLIT_D","HYBRID"]
    def make(name,purpose,seconds,blocks,timeout):
        save(HERE/(name+".json"),dict(purpose=purpose,worker="block_search.py",files=fs,file_sources=paths,seconds=seconds,blocks=blocks,block_timeout_s=timeout))
    cases={b["case"] for b in read(PREV/"screen07.json")["blocks"]}
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    make("orderqual01","training-only solver acceptance",12,[dict(case="r101",role="train",seed=1000,arms=arms)],275)
    make("orderscreen01","development screening",60,paired(cases,[1000,1001],arms),450)
    make("orderdev01","full development",60,paired(full,[1000,1001,1002],arms[:2]+["DIST","DIST_SLOT"]),500)
if __name__=="__main__":main()
