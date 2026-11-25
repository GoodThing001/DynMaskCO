"""Reuse the deployed models and inputs; isolate cores from active comparison runs."""
from pathlib import Path
import json
import shutil
import time


def main():
    here=Path(__file__).resolve().parent
    source=here.parent/"benchmark01"
    phase=here/"phase.json"
    config=json.loads(phase.read_text())
    base=json.loads((source/"phase.json").read_text())
    def wait_for_benchmark():
        peer=json.loads((source/"status.json").read_text())
        return peer["end"] is not None
    print("Waiting for benchmark01 to release its cores",flush=True)
    while not wait_for_benchmark():
        time.sleep(30)
    excluded=set()
    for stage in ("benchmark01","prospective01"):
        p=here.parent/stage/"status.json"
        if p.exists():
            peer=json.loads(p.read_text())
            if not peer["end"]:excluded.update(peer["cores"])
    config["excluded_cores"]=sorted(excluded)
    from unified_runner import available
    idle,_,_=available(48,excluded)
    if len(idle)<=4:
        raise RuntimeError("Not enough unoccupied cores to retain four for other work")
    config["max_cores"]=min(config["max_cores"],len(idle)-4)
    phase.write_text(json.dumps(config))
    for name in set(base["reused_recent_files"]) | {"future_dispatch.py","locked_method.json","heldout_admission.json","heldout_sources.json"}:
        shutil.copy2(source/name,here/name)
    meta=json.loads((source/"heldout_sources.json").read_text())
    for case in sorted({b["case"] for b in config["blocks"]}):
        name=meta["derived_instances"][case]["file"]
        (here/name).parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(source/name,here/name)
    print("Staged nine known cases. Excluded cores:",sorted(excluded),flush=True)


if __name__=="__main__":main()
