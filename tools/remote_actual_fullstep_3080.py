"""Exclusive isolated 3080 CPU engineering job; never rewrite scientific runs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import sys
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build, run_capture

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
SOURCE=ROOT/'research/review_3080_20261010/delivery/actual_checkpoint_engineering_v1'
LOCAL=ROOT/'research/review_3080_20261010/delivery/remote_actual_bundle_v1'
CONFIG=ROOT/'.vscode/sftp.3080.json'
WIN_ROOT=r'C:\Users\zh\project\MASKCO-Main'
PY=r'D:\anaconda\envs\MASKCO_env\python.exe'
REL='research/review_3080_20261010/remote_actual_fullstep_v1'

SUPERVISOR=r'''import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
reg=json.loads((base/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())
env=os.environ.copy()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONIOENCODING']='utf-8'
argv=[sys.executable,str(base/'actual_full_step_qa.py'),'--root',str(base/'extension'),'--upstream',str(base/'upstream'),'--checkpoint',reg['original_checkpoint'],'--registration',str(base/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json'),'--out',str(base/'output'),'--cpu','15']
start=time.time()
with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':start,'maximum_seconds':1200},f)
code=1;failure=None
try:
 with (base/'run.stdout.log').open('xb') as out,(base/'run.stderr.log').open('xb') as err:
  worker=subprocess.Popen(argv,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (base/'worker.pid').write_text(str(worker.pid))
  try:code=worker.wait(timeout=1200)
  except subprocess.TimeoutExpired:
   worker.kill();worker.wait();code=124;failure='registered_engineering_timeout'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_seconds':time.time()-start,'supervisor_pid':os.getpid()},f,indent=2)
'''

LAUNCHER=r'''import json,pathlib,subprocess,sys
base=pathlib.Path(__file__).resolve().parent
if (base/'LAUNCH.json').exists():raise RuntimeError('Already launched; no duplicate run')
with (base/'supervisor.stdout.log').open('xb') as out,(base/'supervisor.stderr.log').open('xb') as err:
 p=subprocess.Popen([sys.executable,str(base/'supervisor.py')],cwd=base,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW,close_fds=True)
with (base/'LAUNCH.json').open('x') as f:json.dump({'pid':p.pid,'mode':'bounded_hidden_windows_supervisor','max_seconds':1200},f)
print(json.dumps({'supervisor_pid':p.pid,'maximum_seconds':1200,'research_model_eligible':False}))
'''

STATUS=r'''import json,pathlib,psutil
base=pathlib.Path(__file__).resolve().parent
r={'base':str(base),'finished':(base/'RUN_EXIT.json').exists()}
if r['finished']:r['run_exit']=json.loads((base/'RUN_EXIT.json').read_text())
else:
 try:
  pid=json.loads((base/'LAUNCH.json').read_text())['pid'];p=psutil.Process(pid)
  r['owned_process_tree']=[{'pid':q.pid,'name':q.name(),'cmdline':q.cmdline(),'cpu_affinity':q.cpu_affinity(),'rss':q.memory_info().rss,'status':q.status()} for q in [p]+p.children(recursive=True)]
 except (OSError,psutil.Error) as e:r['process_error']=type(e).__name__
print(json.dumps(r))
'''


def sha(raw):return hashlib.sha256(raw).hexdigest()


def connect():
    host,user,password=_cfg(str(CONFIG))
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host,port=22,username=user,password=password,timeout=30)
    return client


def remote_target():
    base=json.loads(CONFIG.read_text())['remotePath'].rstrip('/')
    return base,posixpath.join(base,REL)


def deploy():
    LOCAL.mkdir(exist_ok=False)
    reg=json.loads((SOURCE/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json').read_text())
    files={}
    for key,expected in reg['sources'].items():
        group,relative=key.split('/',1)
        source=(ROOT/'MASKCO_code' if group=='upstream' else EXT)/relative
        raw=source.read_bytes()
        if sha(raw)!=expected:raise RuntimeError('Local frozen source changed: '+key)
        files[key]=raw
    for name,expected in reg['engineering_scripts'].items():
        raw=(SOURCE/name).read_bytes()
        if sha(raw)!=expected:raise RuntimeError('Engineering source changed: '+name)
        files[name]=raw
    for name,expected in reg['teacher_files'].items():
        raw=(EXT/reg['teacher_split']/name).read_bytes()
        if sha(raw)!=expected:raise RuntimeError('Archived TRAIN file changed: '+name)
        files['extension/'+reg['teacher_split']+'/'+name]=raw
    files['ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json']=(SOURCE/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json').read_bytes()
    files.update({'supervisor.py':SUPERVISOR.encode(),'launcher.py':LAUNCHER.encode(),'status.py':STATUS.encode()})
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'actual_checkpoint_engineering_only',
       'host':'3080_windows','jax_backend':'cpu','cpu_affinity':[15],'maximum_seconds':1200,
       'original_checkpoint':WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
       'checkpoint_sha256':reg['original_checkpoint_sha256'],'exclusive_remote_directory':REL,
       'research_model_eligible':False,'shared_MD_upload':False,'source_sha256':{k:sha(v) for k,v in files.items()}}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    (LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').write_bytes(files['REMOTE_ENGINEERING_REGISTRATION.json'])
    base,target=remote_target();client=connect();sftp=client.open_sftp()
    try:
        try:sftp.stat(target)
        except FileNotFoundError:pass
        else:raise FileExistsError('Exclusive remote directory exists; never overwrite/relaunch')
        directories=set()
        for name in sorted(files):
            path=posixpath.join(target,name);parent=posixpath.dirname(path)
            components=parent[len(base):].strip('/').split('/')
            current=base
            for component in components:
                current=posixpath.join(current,component)
                if current in directories:continue
                try:sftp.stat(current)
                except FileNotFoundError:sftp.mkdir(current)
                directories.add(current)
            with sftp.open(path,'wb') as f:f.write(files[name])
        checks={}
        for name,raw in files.items():
            with sftp.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==sha(raw)
        if not all(checks.values()):raise RuntimeError('Actual remote source mismatch before launch')
        (LOCAL/'REMOTE_INPUT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2))
    finally:sftp.close();client.close()
    result=run_capture(build(str(CONFIG),WIN_ROOT,PY,REL+'/launcher.py',[],'windows'),60,str(CONFIG))
    (LOCAL/'LAUNCH_RESPONSE.json').write_text(json.dumps(result,indent=2))
    print(result['stdout']);return result['exit_code']


def status():
    result=run_capture(build(str(CONFIG),WIN_ROOT,PY,REL+'/status.py',[],'windows'),60,str(CONFIG))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (LOCAL/('STATUS_'+stamp+'.json')).write_text(json.dumps(result,indent=2))
    print(result['stdout']);return result['exit_code']


def download():
    base,target=remote_target();client=connect();sftp=client.open_sftp()
    out=LOCAL/'downloaded';out.mkdir(exist_ok=False)
    try:
        for name in ('RUN_EXIT.json','LAUNCH.json','WORKER_IDENTITY.json','run.stdout.log','run.stderr.log'):
            with sftp.open(posixpath.join(target,name),'rb') as f:(out/name).write_bytes(f.read())
        checks={}
        manifest=json.loads((LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())
        for name,expected in manifest['source_sha256'].items():
            with sftp.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==expected
        try:output=sftp.listdir(posixpath.join(target,'output'))
        except FileNotFoundError:output=[]
        for name in output:
            if name.endswith('.json'):
                with sftp.open(posixpath.join(target,'output',name),'rb') as f:(out/name).write_bytes(f.read())
        (out/'ACTUAL_END_SOURCE_CHECK.json').write_text(json.dumps(checks,indent=2))
        if 'ENGINEERING_ONLY_FULL_FINAL.pkl' in output:
            h=hashlib.sha256()
            with sftp.open(posixpath.join(target,'output/ENGINEERING_ONLY_FULL_FINAL.pkl'),'rb') as f:
                while True:
                    raw=f.read(4*1024*1024)
                    if not raw:break
                    h.update(raw)
            (out/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json').write_text(json.dumps({'sha256':h.hexdigest(),'full_remote_bytes_independently_read':True,'research_model_eligible':False},indent=2))
    finally:sftp.close();client.close()
    print(json.dumps({'downloaded':str(out),'actual_sources_unchanged':all(checks.values())}))
    return 0


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['deploy','status','download'])
    args=parser.parse_args();raise SystemExit(globals()[args.mode]())
