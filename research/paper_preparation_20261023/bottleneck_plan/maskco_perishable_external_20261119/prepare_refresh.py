"""Build the fixed prospective test of economic-triggered MASKCO refresh on old development cases."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired

def main():
    old=read(HERE/"workscreen01.json")
    cases={b["case"] for b in old["blocks"]}
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    shared=["masked_repair.cpp","calibration.py","calibrated.json"]
    files=["runner.py","refresh_search.py",*shared]
    paths={s:"../"+PREV.name+"/lock/"+s for s in shared}
    paths["runner.py"]="development_runner.py"
    arms=["REFRESH","REFRESH_D","HYBRID"]
    def make(phase, purpose, seconds, blocks):
        save(HERE/(phase+".json"),dict(worker="refresh_search.py",purpose=purpose,files=files,file_sources=paths,
                                        seconds=seconds,blocks=blocks,block_timeout_s=500))
    make("refreshqual01","training-only solver acceptance",24,[dict(case="r101",role="train",seed=2003,arms=arms)])
    make("refreshscreen01","development screening",60,paired(cases,[2003,2004],arms))
    make("refreshdev01","full development",60,paired(full,[3003,3004,3005],arms[:2]+["DIST","DIST_SLOT"]))
if __name__=="__main__":main()
