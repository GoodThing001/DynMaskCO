"""Audit the historical development matrix and evaluate the declared screening contrast."""
import json
from pathlib import Path
from statistics import mean
import sys
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import assess
import audit_delivery
def read(f):return json.loads(f.read_text(encoding="utf8"))
def main(phase):
    assess.HERE=audit_delivery.HERE=HERE
    audit_delivery.audit(phase)
    assess.report(phase)
    if phase=="cachescreen01":
        out=HERE/"results"/phase
        sources=read(out/"analysis.json")["sources"]
        scores={s:v["scores"] for s,v in sources.items()}
        heat={s:v["CACHEHYBRID"]-v["CACHEHYBRID_D"] for s,v in scores.items()}
        cache={s:v["CACHEHYBRID"]-v["CYCLE_FULL"] for s,v in scores.items()}
        passed=len(scores)==3 and min(heat.values())>1.e-6 and mean(cache.values())>1.e-6
        gate=dict(passed=passed,heat_minus_distance=heat,reuse_minus_full=cache)
        (out/"screen_gate.json").write_text(json.dumps(gate,indent=2))
        print(gate)
if __name__=="__main__":main(sys.argv[1])
