"""Read current actual ownership and small receipts, no experiment relaunch."""
import json,os,time
from pathlib import Path
import psutil
BASE=Path(__file__).resolve().parent

def main():
    processes=[]
    for p in psutil.process_iter(['pid','ppid','name','cmdline','create_time','username']):
        try:
            argv=p.info['cmdline'] or []
            if p.pid!=os.getpid() and any(str(BASE).lower() in s.lower() and s.endswith('.py') for s in argv):
                processes.append(dict(**p.info,affinity=p.cpu_affinity(),rss=p.memory_info().rss,children=[q.pid for q in p.children(recursive=True)]))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    receipts={}
    for n in ['LAUNCH_ATTEMPT.json','LAUNCH.json','SUPERVISOR_STARTED.json','RUN_EXIT.json','AUDIT_RUN_EXIT.json','CHAIN_EXIT.json','FAILURE.json','AUDIT_FAILURE.json','SUPERVISOR_FAILURE.json','RESULT.json','INDEPENDENT_SEMANTICS_AUDIT.json']:
        p=BASE/n
        if p.is_file():receipts[n]=json.loads(p.read_text(encoding='utf-8'))
    files=[dict(name=p.relative_to(BASE).as_posix(),size=p.stat().st_size) for p in BASE.rglob('*') if p.is_file()]
    print(json.dumps(dict(observed_epoch=time.time(),base=str(BASE),owned_processes=processes,receipts=receipts,files=files,complete=(BASE/'CHAIN_EXIT.json').is_file() and not processes)))

if __name__=='__main__':main()
