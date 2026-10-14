"""Exclusive registered CPU engineering forward/backward; no optimizer or new data."""
import hashlib,json,posixpath,sys,time
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
OUT=D/'server_teacher_boundary_root_v3';REL='research/review_3080_20261010/server_teacher_boundary_root_v3'
WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
SUPERVISOR=r'''import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
argv=[sys.executable,'-B',str(base/'teacher.py')]
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
    program=(ROOT/'tools/qualify_delivery_server_teacher_boundary_program_root_v2.py').read_bytes()
    teacher=(D/'multistep_insertable_teacher_engineering_root_v2/retained_gls.py').read_bytes()
    requests=(ROOT/'tools/server_teacher_boundary_REQUESTS_root_v1.json').read_bytes()
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Actual server teacher and H deployment on one previously observed second-step state and three declared old18-order empty-slot derived boundaries; no scientific sample or resampling','cpu':[15],'max_s':1200,'minimum_ram_gib':8,'program_sha256':sha(program),'inputs':{'retained_gls.py':sha(teacher),'REQUESTS.json':sha(requests)},'prior_states_sha256':sha((D/'multistep_insertable_teacher_engineering_root_v2/STATES.jsonl').read_bytes()),'contract_hash':'ba40cd0d08fc21ef09d418c7df2bff940294ee2cfecebd2a4ba9ddde35a8268a','request_cap_s':10.,'teacher_search_cap_s':3.,'derived_rule':'second old H state; retain first2 original vehicle routes/orders, subset and reindex all fields, place original routes in slots1/3, empty0/2/4; masks odd-index/all/none','new_science_family':False,'fresh_data':False,'research_model_eligible':False,'scientific_training_qualified':False,'shared_MD_uploaded':False,'engineering_recovery_of':'server_teacher_boundary_root_v2','original_seal_failure':'server_teacher_boundary_root_v1','failed_result_sha256':'a36edf2dc4f67db86d42cf95ae9824a1f28f4eab353d968f608d2f5bc1d67d71','seal_diagnosis_sha256':'baefa64146796a08c913e0393e162927eac716e4c3a3a06a53bd39e8bf621281','closure_initialization_before_start':True}
    write(OUT/'ENGINEERING_REGISTRATION.json',reg)
    files={'teacher.py':program,'retained_gls.py':teacher,'REQUESTS.json':requests,'ENGINEERING_REGISTRATION.json':(OUT/'ENGINEERING_REGISTRATION.json').read_bytes(),'supervisor.py':SUPERVISOR.encode(),'launcher.py':LAUNCHER.encode(),'status.py':STATUS.encode()}
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
