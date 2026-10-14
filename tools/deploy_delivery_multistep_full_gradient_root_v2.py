"""Exclusive registered CPU engineering forward/backward; no optimizer or new data."""
import hashlib,json,posixpath,sys,time
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
OUT=D/'multistep_full_gradient_root_v2';REL='research/review_3080_20261010/multistep_full_gradient_root_v2'
WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
SUPERVISOR=r'''import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
argv=[sys.executable,'-B',str(base/'full_gradient.py')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
started=time.time();code=1;failure=None
with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':started,'max_s':1200},f)
try:
 with (base/'run.stdout.log').open('xb') as out,(base/'run.stderr.log').open('xb') as err:
  child=subprocess.Popen(argv,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (base/'worker.pid').write_text(str(child.pid))
  try:code=child.wait(timeout=1200)
  except subprocess.TimeoutExpired:child.kill();child.wait();code=124;failure='bounded_engineering_timeout'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_s':time.time()-started,'optimizer_steps':0,'supervisor_pid':os.getpid()},f,indent=2)
'''
def main():
    OUT.mkdir(exist_ok=False)
    code="import psutil,json,os,pathlib; me={os.getpid(),os.getppid()}; print(json.dumps({'available_ram_bytes':psutil.virtual_memory().available,'allowed_cpu':psutil.Process().cpu_affinity(),'python_processes':[{'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'argv_names':[pathlib.Path(s).name for s in p.cmdline() if s.endswith('.py')],'owned':any('review_3080_20261010' in s and not s.startswith('exec(') and s.endswith('.py') for s in p.cmdline())} for p in psutil.process_iter() if 'python' in p.name().lower() and p.pid not in me]}))"
    pre=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),30,CONFIG);write(OUT/'RESOURCE_PREFLIGHT.json',pre)
    if pre['exit_code']!=0:raise RuntimeError('Read-only process/resource check failed')
    actual=json.loads(pre['stdout'])
    if actual['available_ram_bytes']<8*1024**3 or 15 not in actual['allowed_cpu'] or any(p['owned'] for p in actual['python_processes']):raise RuntimeError('Actual resource or owned-job barrier; no dispatch')
    program=(ROOT/'tools/qualify_delivery_multistep_full_gradient_program_root_v1.py').read_bytes()
    features=(D/'features_objective_qa_v1/features_objective.py').read_bytes()
    states=(D/'multistep_insertable_teacher_engineering_root_v2/STATES.jsonl').read_bytes()
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'REGISTERED_OLD_STATE_FULL_GRADIENT_ENGINEERING_ONLY','scope':'All eight previously accepted H-deployment teacher states, full original MaskCO forward/backward at same original initialization, coefficients0/.2, no optimization',
         'cpu':[15],'max_s':1200,'minimum_ram_gib':8,'program_sha256':sha(program),'feature_source_sha256':sha(features),'states_sha256':sha(states),
         'prior_independent_teacher_sha256':sha((D/'multistep_insertable_teacher_engineering_root_v2/INDEPENDENT_MULTISTEP_REPLAY.json').read_bytes()),
         'checkpoint':WIN+r'\MASKCO_code\ckpts\cvrp100.ckpt','checkpoint_sha256':'03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db',
         'contract_hash':'ba40cd0d08fc21ef09d418c7df2bff940294ee2cfecebd2a4ba9ddde35a8268a','old_report_cross_runtime_cost_tolerance':1e-8,
         'selected_rows':'all8 in original order without label screening','optimizer_steps':0,'research_model_eligible':False,'new_science_family':False,'fresh_data':False,'shared_MD_uploaded':False}
    write(OUT/'ENGINEERING_REGISTRATION.json',reg)
    files={'full_gradient.py':program,'features_objective.py':features,'STATES.jsonl':states,'ENGINEERING_REGISTRATION.json':(OUT/'ENGINEERING_REGISTRATION.json').read_bytes(),
           'supervisor.py':SUPERVISOR.encode(),'launcher.py':LAUNCHER.replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'engineering_scope':'full gradient only, no optimizer or new science'").encode(),'status.py':STATUS.encode()}
    for n,b in files.items():
        if n!='ENGINEERING_REGISTRATION.json':(OUT/n).write_bytes(b)
    write(OUT/'UPLOAD_MANIFEST.json',{n:sha(b) for n,b in files.items()})
    host,user,password=_cfg(CONFIG);client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30);s=client.open_sftp()
        remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL)
        s.mkdir(remote);checks={}
        for n,b in files.items():
            with s.open(remote+'/'+n,'wx') as f:f.write(b)
            with s.open(remote+'/'+n,'rb') as f:checks[n]=sha(f.read())==sha(b)
        write(OUT/'UPLOAD_ACCEPTANCE.json',checks);s.close()
        if not all(checks.values()):raise RuntimeError('Exclusive registered upload identity mismatch')
    finally:client.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);write(OUT/'LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain/failed launch; inspect actual tree before recovery')
    status=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG);write(OUT/'INITIAL_STATUS.json',status)
    print(json.dumps({'launch_exit':launch['exit_code'],'status':json.loads(status['stdout']) if status['exit_code']==0 else status,'optimizer_steps':0,'scientific_training_qualified':False}));return 0
if __name__=='__main__':raise SystemExit(main())
