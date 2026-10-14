"""Exclusive independent full-gradient engineering replay; no scientific rerun."""
import hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
from deploy_delivery_multistep_full_gradient_root_v2 import SUPERVISOR
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
SOURCE=D/'multistep_optimizer_root_v1';OUT=D/'optimizer_arithmetic_diagnosis_root_v1'
REL='research/review_3080_20261010/optimizer_arithmetic_diagnosis_root_v1'
WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    assert json.loads((SOURCE/'downloaded/RUN_EXIT.json').read_text())['exit_code']==1
    assert json.loads((SOURCE/'downloaded/RESULT.json').read_text())['failed']==2
    OUT.mkdir(exist_ok=False)
    code="import psutil,json,os,pathlib; me={os.getpid(),os.getppid()}; print(json.dumps({'available_ram_bytes':psutil.virtual_memory().available,'allowed_cpu':psutil.Process().cpu_affinity(),'python_processes':[{'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'argv_names':[pathlib.Path(s).name for s in p.cmdline() if s.endswith('.py')],'owned':any('review_3080_20261010' in s and not s.startswith('exec(') and s.endswith('.py') for s in p.cmdline())} for p in psutil.process_iter() if 'python' in p.name().lower() and p.pid not in me]}))"
    pre=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),30,CONFIG);write(OUT/'RESOURCE_PREFLIGHT.json',pre)
    if pre['exit_code']!=0:raise RuntimeError('Read-only process/resource check failed')
    actual=json.loads(pre['stdout'])
    if actual['available_ram_bytes']<8*1024**3 or 15 not in actual['allowed_cpu'] or any(p['owned'] for p in actual['python_processes']):raise RuntimeError('Actual resource or owned-job barrier; no dispatch')
    program=(ROOT/'tools/diagnose_delivery_optimizer_arithmetic_program_root_v1.py').read_bytes()
    inputs={}
    for name in ('ENGINEERING_REGISTRATION.json','train.py','RUN_EXIT.json','RESULT.json','INITIAL_PARAMETERS.json','pretrained_CE_STEPS.jsonl','pretrained_paired_STEPS.jsonl','random_task_paired_STEPS.jsonl'):
        p=SOURCE/'downloaded'/name
        if not p.exists():p=SOURCE/name
        inputs[name]=sha(p.read_bytes())
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Two failed first-step OLD-state float32 arithmetic diagnosis; no optimizer update','program_sha256':sha(program),'inputs':inputs,'cpu':[15],'minimum_ram_gib':8,'max_s':1200,'objective_decomposition_absolute_tolerance':1e-6,'gradient_hash_equality':'exact','optimizer_steps':0,'fresh_data':False,'new_science_family':False,'scientific_training_qualified':False,'shared_MD_uploaded':False}
    write(OUT/'AUDIT_REGISTRATION.json',reg)
    files={'audit.py':program,'AUDIT_REGISTRATION.json':(OUT/'AUDIT_REGISTRATION.json').read_bytes(),'supervisor.py':SUPERVISOR.replace("'full_gradient.py'","'audit.py'").encode(),'launcher.py':LAUNCHER.replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'engineering_scope':'independent full gradient replay only'").encode(),'status.py':STATUS.encode()}
    for n,b in files.items():
        if n!='AUDIT_REGISTRATION.json':(OUT/n).write_bytes(b)
    write(OUT/'UPLOAD_MANIFEST.json',{n:sha(b) for n,b in files.items()})
    host,user,password=_cfg(CONFIG);client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30);s=client.open_sftp();remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL);s.mkdir(remote);checks={}
        for n,b in files.items():
            with s.open(remote+'/'+n,'wx') as f:f.write(b)
            with s.open(remote+'/'+n,'rb') as f:checks[n]=sha(f.read())==sha(b)
        write(OUT/'UPLOAD_ACCEPTANCE.json',checks);s.close()
        if not all(checks.values()):raise RuntimeError('Exclusive audit upload identity mismatch')
    finally:client.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);write(OUT/'LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain launch; inspect actual tree before recovery')
    status=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG);write(OUT/'INITIAL_STATUS.json',status)
    print(json.dumps({'launch_exit':launch['exit_code'],'status':json.loads(status['stdout']) if status['exit_code']==0 else status,'optimizer_steps':0}));return 0
if __name__=='__main__':raise SystemExit(main())
