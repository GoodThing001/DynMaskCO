"""Stage frozen solver files and reserve cores disjoint from the paired benchmark."""
from pathlib import Path
import json
import shutil
from unified_runner import available

HERE=Path(__file__).resolve().parent


def main():
    cfg=json.loads((HERE/"phase.json").read_text())
    previous=HERE.parent/"benchmark01"
    prior=json.loads((previous/"phase.json").read_text())
    state=json.loads((previous/"status.json").read_text())
    if cfg["methods"]!=prior["methods"] or cfg["interpreters"]!=prior["interpreters"] or cfg["seconds"]!=60:
        raise ValueError("Frozen comparative method matrix differs")
    for name in cfg["reused_code"]:shutil.copy2(previous/name,HERE/name)
    for binary in cfg["interpreters"].values():
        if not Path(binary).is_file():raise FileNotFoundError(binary)
    rules=json.loads((HERE/"locked_method.json").read_text())
    registry=json.loads((HERE/"heldout_sources.json").read_text())
    admitted=json.loads((HERE/"heldout_admission.json").read_text())
    if rules["candidate"]!="ORDER_SPLIT" or rules["development_decision"]!="COMPLETE_PASS" or admitted["status"]!="PASS":
        raise ValueError("Frozen solver or panel not qualified")
    pairs={(b["case"],b["seed"]) for b in cfg["blocks"]}
    if pairs!={(c,s) for c in registry["cases"] for s in (65001,65002,65003)} or set(admitted["cases"])!=set(registry["cases"]):
        raise ValueError("Different panel or seeds")
    # The primary lane reservation is held until its final status.json is durable.
    excluded=set(state["cores"]) if state["end"] is None else set()
    candidates,physical,ram=available(200,excluded)
    cap=min(cfg["max_cores"],len(candidates)-cfg["keep_free"])
    if cap<1:raise RuntimeError(f"Insufficient disjoint spare cores or RAM: {candidates} / {ram} GiB")
    cfg.update(excluded_cores=sorted(excluded),max_cores=cap)
    (HERE/"phase.json").write_text(json.dumps(cfg))
    info=dict(physical=physical,peer_reserved=sorted(excluded),eligible=candidates,cap=cap,
              ram_gib=ram,free_reserved=cfg["keep_free"])
    (HERE/"resource_admission.json").write_text(json.dumps(info))
    print(json.dumps(info))


if __name__=="__main__":main()
