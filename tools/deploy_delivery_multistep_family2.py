"""Launch only preregistered family2 teacher phase in a new exclusive directory."""
import hashlib,json,posixpath,sys
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';OUT=D/'multistep_insertable_family2_root_v1';REL='research/review_3080_20261010/multistep_insertable_family2_root_v1'
WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
SUPERVISOR=r'''import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
reg=json.loads((base/'RESEARCH_REGISTRATION.json').read_text(encoding='utf-8'))
argv=[sys.executable,'-B',str(base/'collector.py')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
started=time.time();code=1;failure=None
with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':started,'max_s':1800,'phase':'fresh_teacher_only'},f)
try:
 with (base/'run.stdout.log').open('xb') as out,(base/'run.stderr.log').open('xb') as err:
  child=subprocess.Popen(argv,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (base/'worker.pid').write_text(str(child.pid))
  try:code=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:child.kill();child.wait();code=124;failure='bounded_teacher_phase_timeout_no_retry'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_s':time.time()-started,'optimizer_steps':0,'supervisor_pid':os.getpid(),'training_launched':False},f,indent=2)
'''
def main():
    reg=json.loads((OUT/'RESEARCH_REGISTRATION.json').read_text(encoding='utf-8'));binding=json.loads((OUT/'EXTERNAL_REGISTRATION_BINDING.json').read_text())
    if sha((OUT/'RESEARCH_REGISTRATION.json').read_bytes())!=binding['registration_sha256'] or not reg['phase1_teacher_execution_enabled'] or reg['training']['execution_enabled']:raise ValueError('Teacher-only bound registration')
    code="import psutil,json,os,pathlib; me={os.getpid(),os.getppid()}; print(json.dumps({'available_ram_bytes':psutil.virtual_memory().available,'allowed_cpu':psutil.Process().cpu_affinity(),'python_processes':[{'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'argv_names':[pathlib.Path(s).name for s in p.cmdline() if s.endswith('.py')],'owned':any('review_3080_20261010' in s and not s.startswith('exec(') and s.endswith('.py') for s in p.cmdline())} for p in psutil.process_iter() if 'python' in p.name().lower() and p.pid not in me]}))"
    pre=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),30,CONFIG);write('RESOURCE_PREFLIGHT.json',pre)
    if pre['exit_code']!=0:raise RuntimeError('Read-only resource check failed; no launch')
    actual=json.loads(pre['stdout'])
    if actual['available_ram_bytes']<8*1024**3 or 15 not in actual['allowed_cpu'] or any(p['owned'] for p in actual['python_processes']):raise RuntimeError('Resource/owned-job barrier, no launch')
    names=['collector.py','retained_gls.py','RESEARCH_REGISTRATION.json','EXTERNAL_REGISTRATION_BINDING.json','QUALIFIED_SERVER_REFERENCE.json']
    files={n:(OUT/n).read_bytes() for n in names}
    if sha(files['collector.py'])!=reg['collector_sha256'] or sha(files['retained_gls.py'])!=reg['teacher_sha256'] or sha(files['QUALIFIED_SERVER_REFERENCE.json'])!=reg['qualified_actual_reference_sha256']:raise ValueError('Frozen byte identity')
    # Launcher metadata must describe this finite phase, rather than an old engineering run.
    launcher=LAUNCHER.replace("'max_seconds':1200,'previous_attempt':'v1 interrupted before supervisor body; retained'","'max_seconds':1800,'phase':'family2_teacher_only'").replace("'maximum_seconds':1200","'maximum_seconds':1800")
    files.update({'supervisor.py':SUPERVISOR.encode(),'launcher.py':launcher.encode(),'status.py':STATUS.encode()})
    for n in ('supervisor.py','launcher.py','status.py'):
        with (OUT/n).open('xb') as f:f.write(files[n])
        compile(files[n],n,'exec')
    write('UPLOAD_MANIFEST.json',{'registration_sha256':binding['registration_sha256'],'source_sha256':{n:sha(b) for n,b in files.items()},'shared_MD_upload':False,'automatic_training':False})
    host,user,password=_cfg(CONFIG);client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30);s=client.open_sftp();remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL)
        s.mkdir(remote);checks={}
        for n,b in files.items():
            with s.open(remote+'/'+n,'wx') as f:f.write(b)
            with s.open(remote+'/'+n,'rb') as f:checks[n]=sha(f.read())==sha(b)
        write('UPLOAD_ACCEPTANCE.json',checks);s.close()
        if not all(checks.values()):raise RuntimeError('Actual remote upload identity mismatch; no launch')
    finally:client.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);write('LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain launch; inspect actual own process before recovery')
    status=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG);write('INITIAL_STATUS.json',status)
    print(json.dumps({'launch_exit':launch['exit_code'],'status':json.loads(status['stdout']) if status['exit_code']==0 else status,'phase':'teacher_only','registration_sha256':binding['registration_sha256']}))
if __name__=='__main__':main()
