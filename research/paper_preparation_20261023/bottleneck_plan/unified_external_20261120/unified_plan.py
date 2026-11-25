"""Freeze the 16-arm benchmark and qualify wiring with the original audited endpoint receipts."""
from collections import Counter
import ast
import io
import json
from pathlib import Path
import random
import sys
import tarfile

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
PREVIOUS = STUDIES / "maskco_perishable_external_20261119"
RECENT = STUDIES / "recent5_20261120"
FUTURE_CODE = PREVIOUS / "future_validation"
sys.path.insert(0, str(FUTURE_CODE))
from future_dispatch import check
from prepare_future import interpreters

EXTERNAL = ["ALNS", "VNS", "NDS-Econ", "CaR", "DeepACO", "POMO", "AttentionModel",
            "RouteFinder", "CaDA", "MVMoE", "RRNCO", "PyVRP"]
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "DIST", "DIST_SLOT", *EXTERNAL]
RECENT_METHODS = {"RouteFinder", "CaDA", "MVMoE", "RRNCO", "PyVRP"}
SHARED = ["external_common.py", "pool_search.py", "block_split.py", "masked_repair.cpp",
          "split_order.cpp", "calibrated.json", "calibration.py"]
EXTRA = ["recent_or.py", "recent_worker.py", "neural_methods.py", "other_methods.py", "neural_mask_oracle.py"]

def read(p):
    return json.loads(p.read_text(encoding="utf8"))

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")

def main():
    old = read(RECENT/"matrix.json")
    meta = read(RECENT/"metadata.json")
    assert len(set(meta["cases"])) == 30 and len(set(meta["primary"])) == 27
    assert len(set(meta["primary"]) & set(meta["low_overlap"])) == 7 and len(meta["low_overlap"]) == 10
    assert read(RECENT/"results/benchmark01/independent_audit.json")["endpoints"] == 540
    assert read(PREVIOUS/"reserved_validation/results/reserve02/independent_audit.json")["endpoints"] == 1053
    assert read(PREVIOUS/"xtw_low_overlap/results/xtwsupp01/independent_audit.json")["endpoints"] == 117
    expected={(c,s) for c in meta["cases"] for s in [33001,33002,33003]}
    assert len(old["blocks"]) == 90 and {(x["case"],x["seed"]) for x in old["blocks"]} == expected
    dirs = [PREVIOUS/"reserved_validation",PREVIOUS/"xtw_low_overlap"]
    previous={}
    for d,phase in zip(dirs,["reserve02","xtwsupp01"]):
        p=read(d/(phase+".json"))
        for i,b in enumerate(p["blocks"]):
            previous[b["case"],b["seed"]] = d/"results"/phase/"out"/(str(i)+".json")
    order=list(METHODS)
    random.Random(20261120).shuffle(order)
    blocks=[]
    arm_samples=Counter()
    for i,b in enumerate(old["blocks"]):
        rotation=i % len(order)
        block=dict(case=b["case"],seed=b["seed"],methods=order[rotation:]+order[:rotation])
        blocks.append(block)
        if not arm_samples or i == 89:
            archived=read(previous[b["case"],b["seed"]])["arms"]
            first=archived["ORDER_SPLIT"]["initial_routes"],archived["ORDER_SPLIT"]["initial_U"]
            for arm in METHODS:
                r=read(RECENT/"results/benchmark01/out"/f"{i}_{arm}.json") if arm in RECENT_METHODS else archived[arm]
                check(r,arm,block,r["cpu"],first)
                arm_samples[arm]+=1
    assert all(max(sum(b["methods"][j]==m for b in blocks) for j in range(16)) -
                   min(sum(b["methods"][j]==m for b in blocks) for j in range(16)) <= 1 for m in METHODS)
    plan=dict(name="benchmark01",training_only=False,seconds=60,methods=METHODS,external_methods=EXTERNAL,
              blocks=blocks,interpreters=interpreters(METHODS),max_cores=24,
              reused_recent_files=SHARED+EXTRA+["heldout_sources.json","heldout_admission.json"])
    write(HERE/"matrix.json",plan)
    write(HERE/"metadata.json",meta)
    with tarfile.open(RECENT/"benchmark01.tar.gz") as fresh, tarfile.open(PREVIOUS/"reserve02_inputs.tar.gz") as hist:
        for name in SHARED:
            if fresh.extractfile(name).read() != hist.extractfile(name).read():raise ValueError(name)
        registry=json.load(fresh.extractfile("heldout_sources.json"))
        assert registry["cases"]==meta["cases"]
        for case,entry in registry["derived_instances"].items():
            assert entry["file"] in fresh.getnames() and json.load(fresh.extractfile(entry["file"]))["case"]==case
        locked=json.load(hist.extractfile("locked_method.json"))
        assert (locked["candidate"],locked["matched"],locked["development_decision"]) == ("ORDER_SPLIT","ORDER_SPLIT_D","COMPLETE_PASS")
    files={f:HERE/f for f in ("unified_worker.py","unified_runner.py","stage_existing.py")}
    files.update({"future_dispatch.py": FUTURE_CODE/"future_dispatch.py","phase.json":HERE/"matrix.json"})
    for n,p in files.items():
        if n.endswith(".py"):ast.parse(p.read_text(encoding="utf8"))
    with tarfile.open(HERE/"benchmark01.tar.gz","w:gz") as package:
        for n,p in files.items():package.add(p,arcname=n)
    write(HERE/"qualification.json",dict(status="PASS",type="archived-receipt dispatcher and frozen source qualification",
          verified_cases=30,archived_dispatch_checks=dict(arm_samples),numeric_launches=0,
          compared_shared_sources=SHARED,per_position_appearances=[5,6],
          phase="benchmark01",cases=30,blocks=90,endpoints=1440,allocated_search_core_s=86400))
    print("PASS 30 cases, 90 blocks, 1440 endpoints; 32 dispatcher checks on archived receipts")

if __name__=="__main__":main()

