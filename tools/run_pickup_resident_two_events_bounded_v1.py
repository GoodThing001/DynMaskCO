"""Exclusive 300s supervisor; owned descendants only, no experiment retry."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'resident_two_events_root_v1';phase=sys.argv[1]
if phase not in ('production','audit'):raise ValueError('phase')
if phase=='production' and OUT.exists():raise ValueError('exclusive tree exists')
if phase=='audit' and ((OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json').exists() or json.loads((OUT/'RUN_EXIT.json').read_text(encoding='utf-8'))['exit_code']):raise ValueError('audit barrier')
program=ROOT/'tools'/('qualify_pickup_resident_two_events_v1.py' if phase=='production' else 'audit_pickup_resident_two_events_v1.py');started=time.perf_counter();timed_out=False
with (P/('resident_two_events_root_v1.'+phase+'.stdout.txt')).open('x',encoding='utf-8') as f:
    child=subprocess.Popen([sys.executable,'-B',str(program)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    try:code=child.wait(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out=True;owned=psutil.Process(child.pid).children(recursive=True)
        for p in owned:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        child.kill();child.wait(timeout=10);psutil.wait_procs(owned,timeout=10);code=1
record=dict(phase=phase,timeout=timed_out,exit_code=code,child_pid=child.pid,supervisor_pid=os.getpid(),wall_s=time.perf_counter()-started,cap_s=300.)
with (OUT/('SUPERVISION_'+phase.upper()+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record));raise SystemExit(code)
