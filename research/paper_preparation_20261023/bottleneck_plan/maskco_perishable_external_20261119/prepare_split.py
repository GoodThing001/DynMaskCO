"""Prepare the fixed small development diagnostic without access to external endpoints."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def main():
    cfg=json.loads((HERE/"edgeprobe01.json").read_text())
    cfg.update(purpose="development masked bundle order and partition diagnosis",worker="split_probe.py",block_timeout_s=300)
    cfg["files"]=["runner.py","pool_search.py","masked_repair.cpp","split_order.cpp","block_split.py","split_probe.py","old_states.json"]
    cfg["file_sources"]["masked_repair.cpp"]="../maskco_heatmap_calibration_20261119/lock/masked_repair.cpp"
    with (HERE/"splitprobe01.json").open("x",encoding="utf8") as f:json.dump(cfg,f,indent=2)
if __name__=="__main__":main()
