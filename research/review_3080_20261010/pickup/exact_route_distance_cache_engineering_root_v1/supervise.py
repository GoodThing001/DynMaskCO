"""Bounded local component/audit supervision, not online request qualification."""
import sys,subprocess,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    kind=sys.argv[1]
    if kind not in ('run','audit'):raise ValueError(kind)
    start=time.perf_counter();timeout=False
    with (HERE/(kind+'.stdout.txt')).open('x',encoding='utf-8') as out,(HERE/(kind+'.stderr.txt')).open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-B',str(HERE/(kind+'.py'))],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        try:p.wait(timeout=60)
        except subprocess.TimeoutExpired:timeout=True;p.kill();p.wait(timeout=2)
    r=dict(pid=p.pid,started=start,finished=time.perf_counter(),returncode=p.returncode,timed_out=timeout,reaped=p.poll() is not None,cap_s=60,offline_component_not_online_budget=True)
    with (HERE/(kind.upper()+'_SUPERVISION_RECEIPT.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
    print(json.dumps(r));return p.returncode
if __name__=='__main__':raise SystemExit(main())
