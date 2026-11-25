"""Assign an entire randomized block to one physical core on the shared host."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import threading
import traceback
import psutil

HERE=Path(__file__).resolve().parent
OUT=HERE/"out"
now=lambda: datetime.now(timezone.utc).isoformat()

def available(limit, reserved):
    use=psutil.cpu_percent(interval=1,percpu=True)
    siblings={}
    for logical in os.sched_getaffinity(0):
        topology=Path(f"/sys/devices/system/cpu/cpu{logical}/topology")
        identity=tuple((topology/n).read_text().strip() for n in ("physical_package_id","core_id"))
        siblings.setdefault(identity,[]).append(logical)
    groups=sorted(siblings.values(),key=lambda g:(max(use[x] for x in g),min(g)))
    cores=[min(g) for g in groups if not reserved.intersection(g) and max(use[x] for x in g)<40]
    mem=psutil.virtual_memory().available
    return cores[:min(limit,int(mem//(5*2**30)))],len(siblings),round(mem/2**30,2)

def main():
    cfg=json.loads((HERE/"phase.json").read_text())
    OUT.mkdir()
    reserved=set(cfg.get("excluded_cores", []))
    for stage in cfg.get("exclude_active_stages", []):
        peer=json.loads((HERE.parent/stage/"status.json").read_text())
        if peer["end"] is None:
            reserved.update(peer["cores"])
    cores,physical,ram=available(cfg["max_cores"],reserved)
    if not cores:raise RuntimeError("No available physical core and memory slice")
    jobs=iter(enumerate(cfg["blocks"]))
    lock=threading.Lock()
    state=dict(start=now(),end=None,cores=cores,physical=physical,available_ram_gib=ram,total=len(cfg["blocks"]),
               total_cells=sum(len(b["methods"]) for b in cfg["blocks"]),active={},failed=[],complete=0,cells=0)
    def save():
        stage=HERE/"status.tmp"
        stage.write_text(json.dumps(state))
        stage.replace(HERE/"status.json")
    save()
    def lane(core):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="1",
                 MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
        env.pop("STATIC_PROFIT_SCOPE_FILE",None)
        while True:
            with lock:
                item=None if state["failed"] else next(jobs,None)
                if item is None:return
            idx,b=item
            first=None
            for m in b["methods"]:
                with lock:
                    state["active"][core]=dict(block=idx,case=b["case"],seed=b["seed"],method=m)
                    save()
                try:
                    with (OUT/f"{idx}_{m}.log").open("x") as log:
                        try:
                            subprocess.run(["taskset","-c",str(core),cfg["interpreters"][m],"-B",str(HERE/"unified_worker.py"),str(idx),m],
                                           env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
                            r=json.loads((OUT/f"{idx}_{m}.json").read_text())
                            from future_dispatch import check
                            check(r,m,b,[core],first)
                            if first is None:first=r["initial_routes"],r["initial_U"]
                        except Exception:
                            log.write(traceback.format_exc());raise
                except Exception as e:
                    with lock:
                        state["failed"].append(dict(block=idx,case=b["case"],arm=m,error=str(e)))
                        state["active"].pop(core,None);save()
                    return
                with lock:
                    state["cells"]+=1;save()
            with lock:
                state["complete"]+=1;state["active"].pop(core,None);save()
    with ThreadPoolExecutor(max_workers=len(cores)) as pool:list(pool.map(lane,cores))
    state["end"]=now();save()

if __name__=="__main__":main()

