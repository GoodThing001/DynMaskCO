"""Registered three-arm OLD-fixture optimizer/save-reload engineering only."""
import hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
from deploy_delivery_multistep_full_gradient_root_v2 import SUPERVISOR
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';SOURCE=D/'multistep_full_gradient_root_v2'
OUT=D/'multistep_optimizer_root_v1';REL='research/review_3080_20261010/multistep_optimizer_root_v1'
WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    OUT.mkdir(exist_ok=False)
    code="import psutil,json,os,pathlib,subprocess; me={os.getpid(),os.getppid()}; g=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10); print(json.dumps({'available_ram_bytes':psutil.virtual_memory().available,'disk_free_bytes':psutil.disk_usage('C:/Users/zh/project/MASKCO-Main').free,'allowed_cpu':psutil.Process().cpu_affinity(),'gpu':g.stdout,'python_processes':[{'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'argv_names':[pathlib.Path(s).name for s in p.cmdline() if s.endswith('.py')],'affinity':p.cpu_affinity(),'owned':any('review_3080_20261010' in s and not s.startswith('exec(') and s.endswith('.py') for s in p.cmdline())} for p in psutil.process_iter() if 'python' in p.name().lower() and p.pid not in me]}))"
    pre=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),30,CONFIG);write(OUT/'RESOURCE_PREFLIGHT.json',pre)
    if pre['exit_code']!=0:raise RuntimeError('Read-only process/resource check failed')
    actual=json.loads(pre['stdout'])
    if actual['available_ram_bytes']<8*1024**3 or actual['disk_free_bytes']<4*1024**3 or 15 not in actual['allowed_cpu'] or any(p['owned'] for p in actual['python_processes']):raise RuntimeError('Actual resource or owned-job barrier; no dispatch')
    program=(ROOT/'tools/qualify_delivery_multistep_optimizer_program_root_v1.py').read_bytes();compile(program,'registered_optimizer.py','exec')
    accepted=D/'multistep_full_gradient_audit_root_v1/downloaded/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json';assert json.loads(accepted.read_text())['failed']==0
    inputs={}
    for n in ('full_gradient.py','features_objective.py','ENGINEERING_REGISTRATION.json','STATES.jsonl','RESULT.json','MATCHED_TENSORS.json','FULL_SOURCE_START.json','FULL_SOURCE_END.json'):
        p=SOURCE/'downloaded'/n
        if not p.exists():p=SOURCE/n
        inputs[n]=sha(p.read_bytes())
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Two fixed steps each/three arms; previously observed all8 H-trajectory states; permanently research-ineligible engineering models','program_sha256':sha(program),'inputs':inputs,'prior_independent_sha256':sha(accepted.read_bytes()),'cpu':[15],'minimum_ram_gib':8,'minimum_disk_gib':4,'max_s':1200,'random_initialization_seed':888,'random_construction':'original3Dfactory_no_pretrained_parameters_then_same7zero_projection_rows','schedule':[[0,1,2,3],[4,5,6,7]],'optimizer':{'name':'sgd_without_momentum','learning_rate':.00001},'arms':{'pretrained_CE':{'initialization':'original_pretrained','coefficient':0.},'pretrained_paired':{'initialization':'original_pretrained','coefficient':.2},'random_task_paired':{'initialization':'random888','coefficient':.2}},'fixed_steps_each':2,'checkpoint':WIN+r'\MASKCO_code\ckpts\cvrp100.ckpt','checkpoint_sha256':'03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db','new_science_family':False,'fresh_data':False,'research_model_eligible':False,'scientific_training_qualified':False,'shared_MD_uploaded':False}
    write(OUT/'ENGINEERING_REGISTRATION.json',reg)
    files={'train.py':program,'ENGINEERING_REGISTRATION.json':(OUT/'ENGINEERING_REGISTRATION.json').read_bytes(),'supervisor.py':SUPERVISOR.replace("'full_gradient.py'","'train.py'").replace("'optimizer_steps':0","'engineering_steps_requested':6").encode(),'launcher.py':LAUNCHER.replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'engineering_scope':'three arm fixed2 optimizer old fixture'").encode(),'status.py':STATUS.encode()}
    for n,b in files.items():
        if n!='ENGINEERING_REGISTRATION.json':(OUT/n).write_bytes(b)
    write(OUT/'UPLOAD_MANIFEST.json',{n:sha(b) for n,b in files.items()})
    host,user,password=_cfg(CONFIG);client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30);s=client.open_sftp();remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL);s.mkdir(remote);checks={}
        for n,b in files.items():
            with s.open(remote+'/'+n,'wx') as f:f.write(b)
            with s.open(remote+'/'+n,'rb') as f:checks[n]=sha(f.read())==sha(b)
        write(OUT/'UPLOAD_ACCEPTANCE.json',checks);s.close()
        if not all(checks.values()):raise RuntimeError('Exclusive optimizer upload identity mismatch')
    finally:client.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);write(OUT/'LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain launch; inspect actual tree before recovery')
    status=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG);write(OUT/'INITIAL_STATUS.json',status)
    print(json.dumps({'launch_exit':launch['exit_code'],'status':json.loads(status['stdout']) if status['exit_code']==0 else status,'new_science_family':False}));return 0
if __name__=='__main__':raise SystemExit(main())
