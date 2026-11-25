"""Stage the locked single-core methods when the neighbouring experiments have reserved their cores."""
from pathlib import Path
import json
import os
import shutil
import time


PEERS=("benchmark01","prospective01","trajectory01")


def reservation(here, available, affinity, max_cores, keep_free):
    reserved=set()
    peer_status={}
    for name in PEERS:
        path=here.parent/name/"status.json"
        if not path.is_file():
            return None
        state=json.loads(path.read_text())
        if state["end"] is None:
            if not state["cores"]:
                return None
            reserved.update(state["cores"])
        peer_status[name]=[] if state["end"] is not None else state["cores"]
    idle,physical,ram=available(len(affinity),reserved)
    count=min(max_cores,len(idle)-keep_free)
    if count<1:
        return None
    selected=idle[:count]
    if set(selected)&reserved or len(idle)-count<keep_free:
        raise ValueError("Physical-core exclusion mismatch")
    return dict(peers=peer_status,physical=physical,ram_gib=ram,selected=selected,
                available_reserved_for_other_work=idle[count:],excluded_cores=sorted(set(affinity)-set(selected)))


def main():
    from unified_runner import available
    here=Path(__file__).resolve().parent
    cfg=json.loads((here/"phase.json").read_text())
    src=here.parent/"benchmark01"
    prior=json.loads((src/"phase.json").read_text())
    registered=json.loads((here/"heldout_sources.json").read_text())
    admitted=json.loads((here/"heldout_admission.json").read_text())
    methods=cfg["methods"]
    cases=set(registered["cases"])
    if not (cfg["seconds"]==60 and methods==prior["methods"] and len(methods)==16 and
            cfg["interpreters"]==prior["interpreters"] and len(cases)==8 and
            admitted["status"]=="PASS" and cases==set(admitted["cases"]) and
            {(b["case"],b["seed"]) for b in cfg["blocks"]}=={(c,s) for c in cases for s in (65001,65002,65003)}):
        raise ValueError("Unregistered comparison or panel")
    for name in cfg["reused_code"]:
        shutil.copy2(src/name,here/name)
    locked=json.loads((here/"locked_method.json").read_text())
    if locked["candidate"]!="ORDER_SPLIT" or locked["development_decision"]!="COMPLETE_PASS":
        raise ValueError("Frozen solver failed identification")
    for binary in set(cfg["interpreters"].values()):
        if not Path(binary).is_file():
            raise FileNotFoundError(binary)
    print("Waiting for the three adjacent stages to publish their core reservations.",flush=True)
    affinity=os.sched_getaffinity(0)
    while True:
        decision=reservation(here,available,affinity,cfg["max_cores"],cfg["keep_free"])
        if decision is not None:
            break
        time.sleep(30)
    # Reserve four whole physical lanes by excluding their representative sibling,
    # as understood by unified_runner.available.
    representatives={}
    for cpu in affinity:
        topology=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        ident=tuple((topology/key).read_text().strip() for key in ("physical_package_id","core_id"))
        representatives[ident]=min(cpu,representatives.get(ident,cpu))
    cfg["excluded_cores"]=sorted(set(representatives.values())-set(decision["selected"]))
    cfg["max_cores"]=len(decision["selected"])
    decision["excluded_cores"]=cfg["excluded_cores"]
    (here/"phase.json").write_text(json.dumps(cfg))
    (here/"resource_admission.json").write_text(json.dumps(decision))
    print("Admitted disjoint physical cores",decision["selected"],flush=True)


if __name__=="__main__":
    main()
