"""Prepare and retrieve the registered four-method, eight-case original-model comparison."""
import ast
import json
from pathlib import Path
import random
import shlex
import sys
import tarfile
import time
from origin_worker import ARMS, check

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/"maskco_cross_dataset_20261119"
MOD = HERE.parent/"maskco_perishable_external_20261119"
NAME = "origin01"

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def prepare():
    sample = read(MOD/"orderscreen01.json")["blocks"]
    cfg = read(HERE/"matrix.json")
    pairs = sorted((r["case"],r["seed"]) for r in sample)
    assert len(set(pairs)) == len(pairs) == 16 and {s for _,s in pairs} == {1000,1001}
    assert len({c for c,_ in pairs}) == 8
    rng = random.Random(65201)
    rng.shuffle(pairs)
    order = list(ARMS)
    rng.shuffle(order)
    blocks = [dict(case=c,seed=s,methods=order[i%4:]+order[:i%4]) for i,(c,s) in enumerate(pairs)]
    phase = dict(name=NAME,seconds=60,max_cores=8,methods=ARMS,
                 interpreters={m:cfg["interpreters"]["ORDER_SPLIT"] for m in ARMS},blocks=blocks)
    selected = set(pairs)
    a = read(MOD/"results/orderdev01/independent_audit.json")
    assert a["status"] == "PASS" and a["endpoints"] == 312
    match = 0
    for p in (MOD/"results/orderdev01/out").glob("*.json"):
        old = read(p)
        case_seed = old["case"],old["seed"]
        if case_seed not in selected:
            continue
        ref = None
        for m in ARMS:
            if m in ("MASK","SLOT"):
                row = read(OLD/"deliveries"/"crossmain01"/"cells"/f"{old['case']}_{old['seed']}_{m}.json")
                row["role"] = "validation"
            else:
                row = old["arms"][m]
            check(row,m,old,row["cpu"],ref)
            ref = row["initial_routes"],row["initial_U"]
            match += 1
    assert match == 64
    for oldname,version in frozen_files():
        if oldname.startswith("orderdev01/"):
            src=MOD/"development_lock/orderscreen01"/oldname.split("/")[-1]
            assert src.read_bytes()==version.read_bytes()
    (HERE/"origin_matrix.json").write_text(json.dumps(phase,indent=2),encoding="utf8")
    with tarfile.open(HERE/(NAME+".tar.gz"),"w:gz") as tar:
        for source,name in [("origin_matrix.json","phase.json"),("origin_worker.py","unified_worker.py"),("origin_worker.py","future_dispatch.py"),("stage_origin.py","stage_origin.py"),("unified_runner.py","unified_runner.py")]:
            if source.endswith(".py"):
                ast.parse((HERE/source).read_text(encoding="utf8"))
            tar.add(HERE/source,arcname=name)
    print("16 blocks; 64 previous endpoint schema and common initial fleets verified; 3,840 allocated core-seconds.")

def frozen_files():
    frozen=MOD/"reserved_validation/solver"
    return [("cross/compare.py",OLD/"compare.py"),("slot/worker.py",HERE.parent/"maskco_distance_slot_20261119"/"worker.py"),
            ("slot/operators.py",HERE.parent/"maskco_distance_slot_20261119"/"operators.py")]+[
            ("orderdev01/"+("block_search.py" if p.name=="pool_search.py" else p.name),p) for p in sorted(frozen.iterdir()) if p.is_file()]

def remote(action):
    from unified_transport import connect,CAMPAIGN,PYTHON,PROJECT,transport
    target=CAMPAIGN+"/"+NAME
    paths=dict(cross=PROJECT+"/_maskco_cross_dataset_20261119",slot=PROJECT+"/_maskco_distance_slot_20261119",
               orderdev01=PROJECT+"/_maskco_perishable_external_20261119/orderdev01")
    with connect() as c,c.open_sftp() as s:
        if action=="submit":
            for key,local in frozen_files():
                base,file=key.split("/")
                with s.open(paths[base]+"/"+file,"rb") as old:
                    if old.read()!=local.read_bytes():
                        raise ValueError(f"Frozen solver differs at {key}")
            s.mkdir(target)
            s.put(str(HERE/(NAME+".tar.gz")),target+"/dispatch.tar.gz")
            command=f"cd {target} && ({PYTHON} -B stage_origin.py && {PYTHON} -B unified_runner.py) > controller.log 2>&1; echo $? > exit_code"
            transport.remote_exec(c,f"cd {target} && tar -xzf dispatch.tar.gz && tmux new-session -d -s maskco_unified_origin01 {shlex.quote(command)}")
            print("Single submission origin01 after nine frozen-file identity checks; waiting for neighbouring reservations.", flush=True)
            return
        while True:
            try:
                with s.open(target+"/exit_code") as f:code=int(f.read())
                break
            except FileNotFoundError:
                try:
                    with s.open(target+"/status.json") as f:state=json.load(f)
                    print({k:state[k] for k in ("complete","cells","total_cells","failed")},flush=True)
                except FileNotFoundError:
                    print("waiting for neighbouring core allocations", flush=True)
                if action=="status":return
                time.sleep(30)
        print("Exit",code,flush=True)
        if action=="status":return
        out=HERE/"results"/NAME
        out.mkdir(parents=True,exist_ok=True)
        transport.remote_exec(c,"cd "+target+" && tar -czf receipts.tar.gz --ignore-failed-read out status.json phase.json controller.log resources.json exit_code")
        s.get(target+"/receipts.tar.gz",str(out/"receipts.tar.gz"))
        with tarfile.open(out/"receipts.tar.gz") as tar:
            for item in tar:
                dest=(out/item.name).resolve()
                if not dest.is_relative_to(out.resolve()):raise ValueError(item.name)
                if item.isdir():dest.mkdir(parents=True,exist_ok=True)
                if item.isfile():dest.write_bytes(tar.extractfile(item).read())
        if code:raise RuntimeError(f"Controller exit {code}; recovered logs")
        print("Recovered",out,flush=True)

if __name__=="__main__":
    if sys.argv[1]=="prepare":prepare()
    elif sys.argv[1] in ("submit","status","follow"):remote(sys.argv[1])

