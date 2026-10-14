"""New preregistered snapshot teacher stage, then separately accepted training."""
import ast,hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
ROOT=t.ROOT;E=t.EXT;LINE=ROOT/'research/review_3080_20261010/delivery'
SOURCE=LINE/'finite_training_research_root_v2';LOCAL=LINE/'remote_finite_research_root_v2'
REL='research/review_3080_20261010/remote_finite_research_root_v2'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def scripts(stage):
    maximum=240 if stage=='teacher' else 5400
    name='teacher_pipeline.py' if stage=='teacher' else 'train_full_masked.py'
    out='teacher_output' if stage=='teacher' else 'training_output'
    extra=[] if stage=='teacher' else ['--checkpoint',t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
        '--teacher-dir',None,'--accepted-teacher',None,'--cpu','15']
    code="""import hashlib,json,os,pathlib,psutil,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
reg=json.loads((base/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in reg['source_sha256'].items():
 if sha(base/name)!=expected:raise ValueError('Frozen source changed before stage:'+name)
if 'STAGE'=='training':
 accept=json.loads((base/'ACCEPTED_TEACHER_STATES.json').read_text())
 if accept.get('independent_acceptance') is not True or accept.get('failed')!=0 or accept.get('states')!=256:raise ValueError('No independent256 teacher acceptance')
 if accept['teacher_archive_sha256']!=sha(base/'teacher_output/TEACHER_STATES.jsonl'):raise ValueError('Teacher archive changed')
env=os.environ.copy()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8')
argv=[sys.executable,'-B',str(base/'finite_training_research_root_v2'/'SCRIPT'),'--root',str(base/'extension'),'--upstream',str(base/'upstream'),'--registration',str(base/'finite_training_research_root_v2/RESEARCH_REGISTRATION.json'),'--out',str(base/'OUTPUT'),'--mode','research']
for flag,gate in [('--teacher-gate','teacher_engineering'),('--objective-gate','actual_10D_objective'),('--budget-gate','deployment_10s_budget')]:
 argv.extend([flag,str(base/'finite_training_research_root_v2/gates'/(gate+'.json'))])
if 'STAGE'=='training':argv.extend(['--checkpoint',reg['original_checkpoint'],'--teacher-dir',str(base/'teacher_output'),'--accepted-teacher',str(base/'ACCEPTED_TEACHER_STATES.json'),'--cpu','15'])
start=time.time()
with (base/'STAGE_WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':start,'maximum_seconds':MAXIMUM,'registration_sha256':sha(base/'finite_training_research_root_v2/RESEARCH_REGISTRATION.json')},f)
code=1;failure=None
try:
 with (base/'STAGE.stdout.log').open('xb') as stdout,(base/'STAGE.stderr.log').open('xb') as stderr:
  worker=subprocess.Popen(argv,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
  psutil.Process(worker.pid).cpu_affinity([15])
  try:code=worker.wait(timeout=MAXIMUM)
  except subprocess.TimeoutExpired:
   children=psutil.Process(worker.pid).children(recursive=True)
   for child in reversed(children):
    try:child.kill()
    except psutil.NoSuchProcess:pass
   worker.kill();worker.wait();psutil.wait_procs(children,timeout=10);code=124;failure='registered_timeout_owned_tree_reaped'
except Exception:failure=traceback.format_exc()
end={name:sha(base/name)==expected for name,expected in reg['source_sha256'].items()}
with (base/'STAGE_RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_seconds':time.time()-start,'actual_end_source_check':end},f,indent=2)
"""
    code=code.replace('STAGE',stage).replace('SCRIPT',name).replace('OUTPUT',out).replace('MAXIMUM',str(maximum))
    ast.parse(code)
    launcher=t.LAUNCHER.replace("base/'LAUNCH.json'","base/'"+stage+"_LAUNCH.json'").replace("base/'supervisor.py'","base/'"+stage+"_supervisor.py'")
    launcher=launcher.replace('1200',str(maximum)).replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'stage':'fresh_"+stage+"_root_v2'")
    status="""import json,pathlib,psutil
base=pathlib.Path(__file__).resolve().parent
r={'stage':'STAGE','finished':(base/'STAGE_RUN_EXIT.json').exists()}
if r['finished']:r['run_exit']=json.loads((base/'STAGE_RUN_EXIT.json').read_text())
else:
 try:
  pid=json.loads((base/'STAGE_LAUNCH.json').read_text())['pid'];p=psutil.Process(pid)
  r['owned_process_tree']=[{'pid':q.pid,'status':q.status(),'cpu_affinity':q.cpu_affinity(),'rss':q.memory_info().rss,'cmdline':q.cmdline()} for q in [p]+p.children(recursive=True)]
 except (psutil.Error,OSError) as e:r['error']=str(e)
print(json.dumps(r))
""".replace('STAGE',stage)
    return {stage+'_supervisor.py':code.encode(),stage+'_launcher.py':launcher.encode(),stage+'_status.py':status.encode()}
def upload_new(files,target,client):
    s=client.open_sftp()
    try:
        dirs=set()
        for name,raw in sorted(files.items()):
            path=posixpath.join(target,name);parent=posixpath.dirname(path)
            # mkdir ancestors without deleting, moving or overwriting existing files.
            pending=[];current=parent
            while current and current not in dirs:
                try:s.stat(current);break
                except FileNotFoundError:pending.append(current);current=posixpath.dirname(current)
            for d in reversed(pending):s.mkdir(d);dirs.add(d)
            try:s.stat(path)
            except FileNotFoundError:pass
            else:raise FileExistsError('Never overwrite frozen file:'+name)
            with s.open(path,'wb') as f:f.write(raw)
        return {name:sha(s.open(posixpath.join(target,name),'rb').read())==sha(raw) for name,raw in files.items()}
    finally:s.close()
def deploy_teacher():
    LOCAL.mkdir(exist_ok=False);reg=json.loads((SOURCE/'RESEARCH_REGISTRATION.json').read_text())
    if reg['registration_id']!='D-FULLMASK-SNAPSHOT-TRAIN-root-v2':raise ValueError('Wrong registered scope')
    files={}
    for key,expected in reg['dependency_sources'].items():
        group,name=key.split('/',1);raw=((ROOT/'MASKCO_code' if group=='upstream' else E)/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Frozen dependency changed')
        files[key]=raw
    for name,expected in reg['programs'].items():
        raw=(SOURCE/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Registered programs changed')
        files['finite_training_research_root_v2/'+name]=raw
    for name,expected in reg['sibling_programs'].items():
        raw=(LINE/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Sibling changed')
        files[name]=raw
    files['finite_training_research_root_v2/RESEARCH_REGISTRATION.json']=(SOURCE/'RESEARCH_REGISTRATION.json').read_bytes()
    for name,expected in reg['accepted_gate_sha256'].items():
        raw=(SOURCE/'gates'/(name+'.json')).read_bytes()
        if sha(raw)!=expected:raise ValueError('Gate changed')
        files['finite_training_research_root_v2/gates/'+name+'.json']=raw
    files.update(scripts('teacher'));files.update(scripts('training'))
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{k:sha(v) for k,v in files.items()},
        'original_checkpoint':t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt','science_registration_sha256':sha(files['finite_training_research_root_v2/RESEARCH_REGISTRATION.json']),
        'teacher_only_launch':True,'training_requires_independent256_states':True,'confirmation_enabled':False,'shared_MD_uploaded':False}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    write(LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json',manifest)
    config=json.loads(t.CONFIG.read_text());target=posixpath.join(config['remotePath'].rstrip('/'),REL);client=t.connect()
    try:
        s=client.open_sftp()
        try:
            try:s.stat(target)
            except FileNotFoundError:pass
            else:raise FileExistsError('Do not relaunch or reuse directory')
        finally:s.close()
        checks=upload_new(files,target,client);write(LOCAL/'REMOTE_INPUT_ACCEPTANCE.json',checks)
        if not all(checks.values()):raise ValueError('Actual source mismatch')
    finally:client.close()
    return invoke('teacher_launcher.py','TEACHER_LAUNCH_RESPONSE.json')
def invoke(name,save):
    r=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/'+name,[],'windows'),60,str(t.CONFIG))
    write(LOCAL/save,r);print(r['stdout']);return r['exit_code']
def status(stage):
    r=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/'+stage+'_status.py',[],'windows'),60,str(t.CONFIG))
    write(LOCAL/(stage+'_STATUS_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),r);print(r['stdout']);return r['exit_code']
def download(stage):
    out=LOCAL/(stage+'_downloaded');out.mkdir(exist_ok=False)
    config=json.loads(t.CONFIG.read_text());target=posixpath.join(config['remotePath'].rstrip('/'),REL)
    c=t.connect();s=c.open_sftp()
    try:
        for name in [stage+'_RUN_EXIT.json',stage+'_LAUNCH.json',stage+'_WORKER_IDENTITY.json',stage+'.stdout.log',stage+'.stderr.log']:
            with s.open(posixpath.join(target,name),'rb') as f:(out/name).write_bytes(f.read())
        exit=json.loads((out/(stage+'_RUN_EXIT.json')).read_text())
        if exit['exit_code']!=0:print(json.dumps({'failed':True,'exit':exit['exit_code']}));return 1
        for name in s.listdir(posixpath.join(target,stage+'_output')):
            if name.endswith(('.json','.jsonl')):
                with s.open(posixpath.join(target,stage+'_output',name),'rb') as f:(out/name).write_bytes(f.read())
        manifest=json.loads((LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').read_text());checks={}
        for name,expected in manifest['source_sha256'].items():
            with s.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==expected
        write(out/'INDEPENDENT_ACTUAL_END_SOURCE_CHECK.json',checks)
        print(json.dumps({'downloaded':str(out),'actual_sources_unchanged':all(checks.values())}));return int(not all(checks.values()))
    finally:s.close();c.close()
def deploy_training():
    p=LOCAL/'teacher_downloaded/ACCEPTED_TEACHER_STATES.json';accept=json.loads(p.read_text())
    if accept.get('independent_acceptance') is not True or accept.get('failed')!=0 or accept.get('states')!=256:raise ValueError('Teacher stage not independently accepted')
    config=json.loads(t.CONFIG.read_text());target=posixpath.join(config['remotePath'].rstrip('/'),REL);c=t.connect()
    try:
        checks=upload_new({'ACCEPTED_TEACHER_STATES.json':p.read_bytes()},target,c);write(LOCAL/'TRAINING_GATE_UPLOAD_ACCEPTANCE.json',checks)
        if not all(checks.values()):raise ValueError('Independent teacher gate changed')
    finally:c.close()
    return invoke('training_launcher.py','TRAINING_LAUNCH_RESPONSE.json')
if __name__=='__main__':
    action=sys.argv[1]
    if action=='deploy_teacher':code=deploy_teacher()
    elif action=='deploy_training':code=deploy_training()
    elif action.startswith('status_'):code=status(action.split('_',1)[1])
    elif action.startswith('download_'):code=download(action.split('_',1)[1])
    else:raise ValueError('Unknown stage')
    sys.exit(code)
