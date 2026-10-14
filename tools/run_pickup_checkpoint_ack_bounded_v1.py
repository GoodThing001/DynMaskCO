"""Exclusive bounded invocation, actual owned descendants only."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'checkpoint_ack_root_v1'
if OUT.exists():raise ValueError('exclusive tree already exists; no retry')
started=time.perf_counter();timed_out=False
with (P/'checkpoint_ack_root_v1.production.stdout.txt').open('x',encoding='utf-8') as f:
    child=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/qualify_pickup_checkpoint_ack_v1.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    try:code=child.wait(timeout=180)
    except subprocess.TimeoutExpired:
        timed_out=True;owned=psutil.Process(child.pid).children(recursive=True)
        for p in owned:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        child.kill();child.wait(timeout=10);psutil.wait_procs(owned,timeout=10);code=1
with (OUT/'SUPERVISION.json').open('x',encoding='utf-8') as f:json.dump(dict(timeout=timed_out,exit_code=code,pid=child.pid,wall_s=time.perf_counter()-started,cap_s=180.),f,indent=2)
print(json.dumps(dict(exit_code=code,timeout=timed_out,wall_s=time.perf_counter()-started)));raise SystemExit(code)
