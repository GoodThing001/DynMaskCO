"""Run each predeclared block on a separate, available single physical core."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import psutil

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/"phase.json").read_text())
out=HERE/"out"
out.mkdir()
lock=threading.Lock()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    load=psutil.cpu_percent(interval=1,percpu=True)
    siblings={}
    for i in os.sched_getaffinity(0):
        p=Path(f"/sys/devices/system/cpu/cpu{i}/topology")
        siblings.setdefault(tuple((p/x).read_text() for x in ("physical_package_id","core_id")),[]).append(i)
    active_test=json.loads((HERE.parent/"replicate01/status.json").read_text())
    reserved=set(active_test["cores"]) if active_test["end"] is None else set()
    cores=[min(v) for v in sorted(siblings.values(),key=lambda v:max(load[i] for i in v)) if not reserved.intersection(v) and max(load[i] for i in v)<40]
    cores=cores[:min(12,psutil.virtual_memory().available//(5*2**30))]
    if not cores:raise RuntimeError("Insufficient available cores or RAM")
    status=dict(start=now(),active={},failed=[],complete=0,cores=cores,total=len(plan["blocks"]),end=None)
    def save():
        tmp=HERE/"status.tmp";tmp.write_text(json.dumps(status));tmp.replace(HERE/"status.json")
    save()
    blocks=iter(enumerate(plan["blocks"]))
    def lane(core):
        while True:
            with lock:
                if status["failed"]:return
                item=next(blocks,None)
                if item is None:return
                i,b=item;status["active"][core]=i;save()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            env.pop("STATIC_PROFIT_SCOPE_FILE",None)
            with (out/f"{i}.log").open("w") as log:
                try:
                    rc=subprocess.run(["taskset","-c",str(core),b.get("python",sys.executable),"-B",str(HERE/plan["worker"]),str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=plan["block_timeout_s"]).returncode
                except (OSError,subprocess.TimeoutExpired):
                    log.write(traceback.format_exc());rc=1
            with lock:
                if rc or not (out/f"{i}.json").is_file():status["failed"].append(i)
                else:status["complete"]+=1
                status["active"].pop(core,None);save()
    with ThreadPoolExecutor(max_workers=len(cores)) as pool:list(pool.map(lane,cores))
    status["end"]=now();save()
if __name__=="__main__":main()
