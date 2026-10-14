"""Bound one exclusive production or independent extension execution to 900 seconds."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'observed_day_extension_root_v1'
phase=sys.argv[1]
if phase not in ('production','audit'):raise ValueError('phase')
program=ROOT/'tools'/('qualify_pickup_observed_day_extension_v1.py' if phase=='production' else 'audit_pickup_observed_day_extension_v1.py')
if phase=='production' and OUT.exists():raise ValueError('existing exclusive production tree')
if phase=='audit' and ((OUT/'INDEPENDENT_OBSERVED_DAY_EXTENSION.json').exists() or json.loads((OUT/'RUN_EXIT.json').read_text())['exit_code']!=0):raise ValueError('audit launch barrier')
started=time.perf_counter();timed_out=False
with (P/('observed_day_extension_root_v1.'+phase+'.stdout.txt')).open('x',encoding='utf-8') as stream:
    process=subprocess.Popen([sys.executable,'-B',str(program)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    try:code=process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out=True;owned=psutil.Process(process.pid).children(recursive=True)
        for child in owned:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        process.kill();process.wait(timeout=10);psutil.wait_procs(owned,timeout=10);code=1
record=dict(phase=phase,supervisor_pid=os.getpid(),child_pid=process.pid,timeout=timed_out,exit_code=code,wall_s=time.perf_counter()-started,cap_s=900.)
target=OUT/('SUPERVISION_'+phase.upper()+'.json') if OUT.exists() else P/('observed_day_extension_root_v1.'+phase+'.failure.json')
with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record));raise SystemExit(code)
