"""Allocate the frozen instance-encoding diagnosis and its conditional development comparisons."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
def main():
    common={"runner.py":"development_runner.py","pool_search.py":"../"+PREV.name+"/lock/pool_search.py",
            "masked_repair.cpp":"../"+PREV.name+"/lock/masked_repair.cpp","calibration.py":"../"+PREV.name+"/lock/calibration.py"}
    basic=["pool_search.py","encoded_mask.py","static_encoder.py","masked_repair.cpp","runner.py","calibration.py"]
    cfg=dict(purpose="train-only instance encoder economic and compute diagnosis",worker="static_probe.py",files=basic+["static_probe.py"],
             file_sources=common,blocks=read(PREV/"traincal01.json")["blocks"],block_timeout_s=360)
    save(HERE/"staticprobe01.json",cfg)
    common=dict(common,**{"calibrated.json":"../"+PREV.name+"/lock/calibrated.json","static_search.py":"cached_search.py"})
    arms=["STATIC","STATIC_D","HYBRID"]
    cases=sorted({b["case"] for b in read(PREV/"screen07.json")["blocks"]})
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    def plan(name,purpose,seconds,blocks,timeout):
        save(HERE/(name+".json"),dict(purpose=purpose,worker="static_search.py",files=[x for x in basic if x!="pool_search.py"]+["static_search.py","calibrated.json"],
                                     file_sources=common,seconds=seconds,blocks=blocks,block_timeout_s=timeout))
    plan("staticqual01","training-only solver acceptance",12,[dict(case="r101",seed=1000,role="train",arms=arms)],320)
    plan("staticscreen01","development screening",60,paired(cases,[1000,1001],arms),450)
    plan("staticdev01","full development",60,paired(full,[1000,1001,1002],arms[:2]+["DIST","DIST_SLOT"]),500)
if __name__=="__main__":main()
