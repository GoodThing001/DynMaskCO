"""Bound the new startup-only stage and reap only its launched descendants."""
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v3'


def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def main():
    stage=sys.argv[1]
    if stage not in ('producer','audit'):raise ValueError('Fixed startup-only stage')
    reg=json.loads((OUT/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    if (OUT/(stage+'_SUPERVISOR_STARTED.json')).exists():raise RuntimeError('No repeat stage')
    child=None;owned=None;identity=None;timed_out=False;killed=[]
    start=time.perf_counter()
    with (OUT/(stage+'_stdout.txt')).open('xb') as out,(OUT/(stage+'_stderr.txt')).open('xb') as err:
        try:
            child=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/run_delivery_early_launch_v3.py'),stage],
                stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            owned=psutil.Process(child.pid)
            identity=dict(pid=child.pid,create_time=owned.create_time(),cmdline=owned.cmdline(),username=owned.username())
            write(stage+'_SUPERVISOR_STARTED.json',dict(clock=start,identity=identity,limit_seconds=reg['outer_supervisor_seconds']))
            try:code=child.wait(timeout=reg['outer_supervisor_seconds'])
            except subprocess.TimeoutExpired:timed_out=True;code=None
        finally:
            if child is not None and child.poll() is None:
                if owned is not None and identity is not None and owned.create_time()==identity['create_time']:
                    descendants=owned.children(recursive=True)
                    for p in reversed(descendants):
                        try:
                            if p.username()!=identity['username']:raise RuntimeError('Unexpected descendant owner')
                            killed.append(dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline()))
                            p.kill()
                        except psutil.NoSuchProcess:pass
                    psutil.wait_procs(descendants,timeout=2)
                child.kill();code=child.wait(timeout=2)
    write(stage+'_SUPERVISOR_EXIT.json',dict(actual_returncode=code,timeout=timed_out,killed_descendants=killed,
        start_clock=start,end_clock=time.perf_counter(),natural_exit=not timed_out,identity=identity,
        model_executed=False))
    print(stage+' supervised actual exit '+str(code))
    raise SystemExit(code)


if __name__=='__main__':main()
