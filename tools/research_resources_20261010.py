import json, sys
from pathlib import Path
from _srv import run

code = r'''import json, time, psutil, subprocess
before=psutil.cpu_times(percpu=True); time.sleep(2); after=psutil.cpu_times(percpu=True)
cpus=[]
for i,(a,b) in enumerate(zip(before,after)):
    total=sum(b)-sum(a); idle=b.idle-a.idle
    cpus.append({'cpu':i,'busy_percent':round(100*(1-idle/total),1) if total>0 else None})
processes=[]
for p in psutil.process_iter(['pid','ppid','name','cmdline','username']):
    try:
        cmd=p.info['cmdline'] or []
        if p.info['username']=='hzeng' and ('python' in p.info['name'] or 'tmux' in p.info['name']):
            processes.append({'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'cmd':cmd,'affinity':p.cpu_affinity()})
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps({'server_time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'cpus':cpus,'memory_available':psutil.virtual_memory().available,'disk_free':psutil.disk_usage('/home/hzeng/project').free,'gpu':gpu.stdout,'processes':processes},ensure_ascii=False))'''
import base64
payload=base64.b64encode(code.encode()).decode()
out=run("/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import base64;exec(base64.b64decode('"+payload+"'))\"",120)
d=json.loads(out)
dest=Path(sys.argv[1]);dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print(json.dumps({'saved':str(dest),'server_time':d['server_time'],'least_busy':sorted(d['cpus'],key=lambda x:x['busy_percent'])[:16],'gpu':d['gpu'],'n_processes':len(d['processes']),'disk_free_gb':d['disk_free']/1e9},ensure_ascii=False))
