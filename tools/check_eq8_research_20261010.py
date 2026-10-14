"""Read actual QA/pilot state and recursively inspect only our process tree."""
import base64,json,sys,hashlib,paramiko
from pathlib import Path
from _srv import run,_cfg
R='/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_d_eq8_v1'
code=r'''from pathlib import Path
import json,hashlib,psutil,datetime
R=Path("REMOTE")
def read(p):
    if not p.is_file():return None
    try:return json.loads(p.read_text())
    except Exception:return p.read_text()[-1500:]
status={'server_time':datetime.datetime.now().isoformat(),'qa_gate':read(R/'QA_GATE.json'),'qa_exit':read(R/'qa.exit.json'),'pilot_exit':read(R/'pilot.exit.json'),'completed':read(R/'COMPLETED.json'),'failure':read(R/'pipeline_failure.json'),'pipeline_exit':read(R/'pipeline.exit')}
for stage in ('qa','pilot'):
    progress=R/stage/'progress.jsonl';n=0;last=None
    if progress.is_file():
        for line in progress.read_text().splitlines():
            try:last=json.loads(line);n+=1
            except Exception:pass
    status[stage+'_progress']={'completed':n,'last_instance':last['instance_id'] if last else None}
roots=[]
for proc in psutil.process_iter(['pid','cmdline']):
    try:
        cmd=proc.info['cmdline'] or []
        if 'pipeline.py' in cmd and proc.cwd()==str(R):roots.append(proc)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
tree=[]
for parent in roots:
    for proc in [parent]+parent.children(recursive=True):
        try:tree.append({'pid':proc.pid,'ppid':proc.ppid(),'cmd':proc.cmdline(),'affinity':proc.cpu_affinity(),'cpu_time_s':sum(proc.cpu_times()[:2])})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
status['own_process_tree']=tree
for p in ('qa.log','pilot.log','pipeline.log'):
    file=R/p
    if file.is_file():status[p+'_tail']=file.read_text(errors='replace')[-1600:]
if (R/'SOURCE_END.json').is_file():
    start=read(R/'SOURCE_START.json');end=read(R/'SOURCE_END.json');actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in start}
    status['actual_end_identity']={'match':start==end==actual,'count':len(start),'mismatch':[p for p in start if start[p]!=actual[p]]}
print(json.dumps(status))'''.replace('REMOTE',R)
payload=base64.b64encode(code.encode()).decode();d=json.loads(run("/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import base64;exec(base64.b64decode('"+payload+"'))\"",120))
base=Path('C-VRP_Cold-chainVehicleRoutingProblem/results/research_20261010');dest=base/sys.argv[1]
with dest.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in d.items() if not k.endswith('_tail')},ensure_ascii=False))
if d['failure'] or d['pipeline_exit'] not in (None,0):print(d.get('pipeline.log_tail',''),d.get('qa.log_tail',''))
if '--download' in sys.argv:
    if not d.get('actual_end_identity',{}).get('match'):raise RuntimeError('Only complete end-verified batch can enter independent audit')
    target=base/'delivery_raw_v1';target.mkdir(exist_ok=False)
    host,user,pw=_cfg();c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=pw,timeout=30);s=c.open_sftp()
    names=['PRE_DATA_PROTOCOL.json','PILOT_PROTOCOL.json','SOURCE_MANIFEST.json','SOURCE_START.json','SOURCE_END.json','QA_GATE.json','qa.exit.json','pilot.exit.json','COMPLETED.json','pipeline.exit','qa.log','pilot.log','pipeline.log']
    names += [stage+'/'+p for stage in ('qa','pilot') for p in ('PROTOCOL.json','SOURCE_MANIFEST.json','identity_start.json','identity_end.json','progress.jsonl','result.json')]
    for name in names:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        with s.open(R+'/'+name,'rb') as src,p.open('xb') as dst:dst.write(src.read())
    with (target/'pilot/INDEPENDENT_ACTUAL_END_IDENTITY.json').open('x') as f:json.dump(d['actual_end_identity'],f,indent=2)
    manifest=json.loads((target/'SOURCE_MANIFEST.json').read_text());srcroot=target/'frozen_src'
    for name,h in manifest.items():
        p=srcroot/name;p.parent.mkdir(parents=True,exist_ok=True)
        with s.open(R+'/src/'+name,'rb') as src,p.open('xb') as dst:dst.write(src.read())
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Downloaded frozen source mismatch '+name)
    s.close();c.close();print('Downloaded full raw batch and frozen source to',target)
