"""Freeze a work-count repair schedule calibrated only on training profiles."""
from pathlib import Path
import sys
import tarfile
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
from phase_plan import read,save,paired
def main():
    timing=read(HERE/"results/trialclock01/analysis.json")
    if not timing["advance"]:raise ValueError("Training time-slicing diagnosis missing")
    save(HERE/"work_caps.json",timing)
    directory=HERE/"work_reference";directory.mkdir(exist_ok=True)
    with tarfile.open(HERE/"factorial01_inputs.tar.gz") as tar:
        for n in ("ablation_search.py","encoded_mask.py","disagreement.py","calibrated.json","calibration.py","masked_repair.cpp"):
            (directory/("pool_common.py" if n=="ablation_search.py" else n)).write_bytes(tar.extractfile(n).read())
    fs=[p.name for p in directory.iterdir()]+["runner.py","work_search.py","work_caps.json"]
    paths={f:"work_reference/"+f for f in fs if (directory/f).exists()};paths["runner.py"]="development_runner.py"
    arms=["WORKMASK","WORKMASK_D","MASK_ONLY","MASK_ONLY_D"]
    def make(name,why,seconds,blocks,timeout):
        save(HERE/(name+".json"),dict(purpose=why,worker="work_search.py",files=fs,file_sources=paths,seconds=seconds,blocks=blocks,block_timeout_s=timeout))
    cases={b["case"] for b in read(PREV/"screen07.json")["blocks"]}
    full=read(PREV.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    make("workqual01","training-only fixed-work repeated-prefix acceptance",14,[dict(case="r101",role="train",seed=1000,arms=arms)],355)
    make("workscreen01","development screening",60,paired(cases,[1000,1001],arms),490)
    make("workdev01","full development",60,paired(full,[1000,1001,1002],arms[:2]+["DIST","DIST_SLOT"]),510)
if __name__=="__main__":main()
