"""One 60-second stage; only terminate exact launched process descendants."""
import json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'research/review_3080_20261010/delivery/phase_cleanup_boundary_engineering_root_v1'
stage=sys.argv[1]
if stage not in ('producer','audit'):raise ValueError('Fixed trace-only stage')
script=ROOT/'tools'/('run_phase_cleanup_boundary_v1.py' if stage=='producer' else 'audit_phase_cleanup_boundary_v1.py')
def write(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
if (O/(stage+'_SUPERVISOR_STARTED.json')).exists():raise RuntimeError('No repeat stage')
with (O/(stage+'_stdout.txt')).open('xb') as out,(O/(stage+'_stderr.txt')).open('xb') as err:
    start=time.perf_counter();child=subprocess.Popen([sys.executable,'-B',str(script)],stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    proc=psutil.Process(child.pid);identity=dict(pid=child.pid,create_time=proc.create_time(),cmdline=proc.cmdline(),username=proc.username())
    write(stage+'_SUPERVISOR_STARTED.json',dict(clock=start,identity=identity,limit_seconds=60))
    timeout=False;killed=[]
    try:code=child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        timeout=True
        if proc.create_time()!=identity['create_time']:raise RuntimeError('Process identity changed')
        for p in reversed(proc.children(recursive=True)):
            try:
                if p.username()!=identity['username']:raise RuntimeError('Unowned descendant')
                killed.append(dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline()));p.kill()
            except psutil.NoSuchProcess:pass
        child.kill();code=child.wait(timeout=2)
    write(stage+'_SUPERVISOR_EXIT.json',dict(actual_returncode=code,timeout=timeout,killed_descendants=killed,
        end_clock=time.perf_counter(),natural_exit=not timeout,identity=identity,model_executed=False))
print(stage+' actual exit '+str(code))
raise SystemExit(code)
