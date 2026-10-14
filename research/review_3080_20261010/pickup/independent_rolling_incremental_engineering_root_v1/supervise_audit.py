"""Independent reference has bounded120s offline supervision, not online budget."""
import subprocess,sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    started=time.perf_counter();p=None;timed_out=False
    with (HERE/'audit.stdout.txt').open('x',encoding='utf-8') as out,(HERE/'audit.stderr.txt').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-B',str(HERE/'audit.py')],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        try:p.wait(timeout=120.)
        except subprocess.TimeoutExpired:timed_out=True;p.kill();p.wait(timeout=2.)
    receipt=dict(started=started,finished=time.perf_counter(),pid=p.pid,returncode=p.returncode,timed_out=timed_out,reaped=p.poll() is not None,cap_s=120.,offline_not_online_budget=True)
    with (HERE/'AUDIT_SUPERVISION_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt));return p.returncode
if __name__=='__main__':raise SystemExit(main())
