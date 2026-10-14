"""Exactly one bounded owned Popen for the requested local stage."""
import sys,subprocess,json,time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
def main():
    kind=sys.argv[1]
    if kind not in ('run','audit'):raise ValueError('Unknown stage')
    active=[]
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            if any(s.lower()==str(HERE/(name+'.py')).lower() for s in cmd for name in ('run','audit')):active.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if active:raise RuntimeError('Exact owned workload still active')
    started=time.perf_counter();timed_out=False
    with (HERE/(kind+'.stdout.txt')).open('x',encoding='utf-8') as out,(HERE/(kind+'.stderr.txt')).open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-B',str(HERE/(kind+'.py'))],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            owned=psutil.Process(p.pid);identity=dict(pid=p.pid,create_time=owned.create_time(),cmdline=owned.cmdline(),owner=owned.username())
            p.wait(timeout=600.)
        except subprocess.TimeoutExpired:timed_out=True;p.kill();p.wait(timeout=2.)
    receipt=dict(started=started,finished=time.perf_counter(),returncode=p.returncode,timed_out=timed_out,identity=identity,reaped=p.poll() is not None,cap_s=600,normal_stage_requested=kind,offline_component_not_online_budget=True)
    with (HERE/(kind.upper()+'_SUPERVISION_RECEIPT.json')).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt));return p.returncode
if __name__=='__main__':raise SystemExit(main())
