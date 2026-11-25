"""Calibrate operation counts only on the original training cases."""
from pathlib import Path
from statistics import median
import json
import sys
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))
def plan():
    cfg=read(HERE/"staticprobe01.json")
    cfg["worker"]="trial_clock_probe.py"
    cfg["files"]=[x for x in cfg["files"] if x not in ("static_probe.py","encoded_mask.py","static_encoder.py")]+["trial_clock_probe.py"]
    cfg["purpose"]="training-only paired repeatability diagnosis"
    cfg["blocks"]=[b for b in cfg["blocks"] if b["case"] in ("r101","r108","rc101","rc108")]
    with (HERE/"trialclock01.json").open("x",encoding="utf8") as f:json.dump(cfg,f,indent=2)
def report():
    d=HERE/"results/trialclock01";stat=read(d/"status.json")
    if stat["failed"] or stat["complete"]!=4 or not stat["end"]:raise ValueError("Training clock matrix missing")
    records=[t for i in range(4) for t in read(d/f"out/{i}.json")["states"]]
    winners=sum(z["winners"][0]!=z["winners"][1] for z in records)
    changed=sum(not z["chosen_state_equal"] for z in records)
    caps=[int(median(v for z in records for rep in z["individual_work"] for v in rep[g])) for g in range(2)]
    result=dict(contexts=len(records),changed_winner=winners,changed_chosen_state=changed,greedy_regret_caps=caps,
                advance=all(z["fixed_equal"] for z in records) and (winners>=.10*len(records) or changed>=.25*len(records)))
    (d/"analysis.json").write_text(json.dumps(result,indent=2));print(result)
if __name__=="__main__":(plan if sys.argv[1]=="plan" else report)()
