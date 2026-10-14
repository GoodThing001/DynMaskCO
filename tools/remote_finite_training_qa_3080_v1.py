"""Run only the registered eight-state, two-step batch trainer engineering QA."""
from datetime import datetime,timezone
import json,hashlib,posixpath,shutil
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
LINE=ROOT/'research/review_3080_20261010/delivery'
SOURCE=LINE/'finite_training_preparation_v1'
AGENT=Path(r'C:\Users\mmc\.codex\worktrees\ba63\MASKCO-Main\.arbor\delivery\research\review_3080_20261010\delivery\finite_training_preparation_v1')
LOCAL=LINE/'remote_finite_training_bundle_v1'
REL='research/review_3080_20261010/remote_finite_training_qa_v1'
REGSHA='1387fbb41ee5fbce2496eca133dddb89c03ddf1bc67b3485ccc3e8f15ee77ade'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def integrate():
    if sha((AGENT/'FINITE_ENGINEERING_REGISTRATION.json').read_bytes())!=REGSHA:raise ValueError('Registration changed')
    SOURCE.mkdir(exist_ok=False);records=[]
    for p in AGENT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        q=SOURCE/p.relative_to(AGENT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        records.append({'path':q.relative_to(SOURCE).as_posix(),'sha256':sha(q.read_bytes())})
    (SOURCE/'ROOT_INTEGRATION_IDENTITY.json').write_text(json.dumps({'copied':records,'independent_execution':False},indent=2))
    print(json.dumps({'files':len(records),'copied_not_reexecuted':True}))

def deploy():
    LOCAL.mkdir(exist_ok=False)
    rawreg=(SOURCE/'FINITE_ENGINEERING_REGISTRATION.json').read_bytes()
    if sha(rawreg)!=REGSHA:raise ValueError('Registered finite input changed')
    reg=json.loads(rawreg);files={}
    for key,expected in reg['dependency_sources'].items():
        group,relative=key.split('/',1);raw=((ROOT/'MASKCO_code' if group=='upstream' else EXT)/relative).read_bytes()
        if sha(raw)!=expected:raise ValueError('Actual dependency changed: '+key)
        files[key]=raw
    for name,expected in reg['programs'].items():
        raw=(SOURCE/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Finite program changed: '+name)
        files['finite_training_preparation_v1/'+name]=raw
    for name,expected in reg['sibling_programs'].items():
        raw=(LINE/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Sibling program changed: '+name)
        files[name]=raw
    files['finite_training_preparation_v1/FINITE_ENGINEERING_REGISTRATION.json']=rawreg
    for name in ('TEACHER_STATES.jsonl','TEACHER_SUMMARY.json','SOURCE_END.json'):
        files['finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/'+name]=(SOURCE/'ENGINEERING_TEACHER_QA_v1'/name).read_bytes()
    summary=json.loads(files['finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_SUMMARY.json'])
    if summary['registration_sha256']!=REGSHA or summary['teacher_archive_sha256']!=sha(files['finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl']):raise ValueError('Teacher identity changed')
    supervisor=t.SUPERVISOR.replace('1200','300')
    supervisor=supervisor.replace("base/'actual_full_step_qa.py'","base/'finite_training_preparation_v1/train_full_masked.py'")
    supervisor=supervisor.replace("base/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json'","base/'finite_training_preparation_v1/FINITE_ENGINEERING_REGISTRATION.json'")
    supervisor=supervisor.replace("'--cpu','15']","'--cpu','15','--teacher-dir',str(base/'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1'),'--mode','engineering']")
    launcher=t.LAUNCHER.replace('1200','300').replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'stage':'new_batch4_two_step_engineering'")
    files.update({'supervisor.py':supervisor.encode(),'launcher.py':launcher.encode(),'status.py':t.STATUS.encode()})
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'finite_training_engineering_only',
       'original_checkpoint':t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
       'maximum_seconds':300,'cpu_affinity':[15],'exclusive_remote_directory':REL,
       'source_sha256':{name:sha(raw) for name,raw in files.items()},'research_model_eligible':False,'shared_MD_upload':False}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    (LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').write_bytes(files['REMOTE_ENGINEERING_REGISTRATION.json'])
    cfg=json.loads(t.CONFIG.read_text());base=cfg['remotePath'].rstrip('/');target=posixpath.join(base,REL)
    client=t.connect();sftp=client.open_sftp()
    try:
        try:sftp.stat(target)
        except FileNotFoundError:pass
        else:raise FileExistsError('Output exists; never overwrite or rerun')
        directories=set()
        for name,raw in sorted(files.items()):
            path=posixpath.join(target,name);current=base
            for component in posixpath.dirname(path)[len(base):].strip('/').split('/'):
                current=posixpath.join(current,component)
                if current not in directories:
                    try:sftp.stat(current)
                    except FileNotFoundError:sftp.mkdir(current)
                    directories.add(current)
            with sftp.open(path,'wb') as stream:stream.write(raw)
        checks={}
        for name,raw in files.items():
            with sftp.open(posixpath.join(target,name),'rb') as stream:checks[name]=sha(stream.read())==sha(raw)
        (LOCAL/'REMOTE_INPUT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2))
        if not all(checks.values()):raise ValueError('Remote inputs differ')
    finally:sftp.close();client.close()
    result=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/launcher.py',[],'windows'),60,str(t.CONFIG))
    (LOCAL/'LAUNCH_RESPONSE.json').write_text(json.dumps(result,indent=2));print(result['stdout']);return result['exit_code']

def status():
    t.LOCAL=LOCAL;t.REL=REL;return t.status()

def download():
    cfg=json.loads(t.CONFIG.read_text());target=posixpath.join(cfg['remotePath'].rstrip('/'),REL)
    out=LOCAL/'downloaded';out.mkdir(exist_ok=False)
    client=t.connect();sftp=client.open_sftp();saved={}
    try:
        for name in ('RUN_EXIT.json','LAUNCH.json','WORKER_IDENTITY.json','run.stdout.log','run.stderr.log'):
            with sftp.open(posixpath.join(target,name),'rb') as stream:(out/name).write_bytes(stream.read())
        checks={}
        for name,expected in json.loads((LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())['source_sha256'].items():
            with sftp.open(posixpath.join(target,name),'rb') as stream:checks[name]=sha(stream.read())==expected
        for name in sftp.listdir(posixpath.join(target,'output')):
            path=posixpath.join(target,'output',name)
            if name.endswith(('.json','.jsonl')):
                with sftp.open(path,'rb') as stream:(out/name).write_bytes(stream.read())
            elif name.endswith('.pkl'):
                h=hashlib.sha256();count=0
                with sftp.open(path,'rb') as stream:
                    while True:
                        raw=stream.read(4*1024*1024)
                        if not raw:break
                        count+=len(raw);h.update(raw)
                saved[name]={'sha256':h.hexdigest(),'bytes':count,'full_remote_bytes_read':True}
        (out/'ACTUAL_END_SOURCE_CHECK.json').write_text(json.dumps(checks,indent=2))
        (out/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json').write_text(json.dumps(saved,indent=2))
    finally:sftp.close();client.close()
    print(json.dumps({'downloaded':str(out),'actual_sources_unchanged':all(checks.values()),'saved_files':len(saved)}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('integrate','deploy','status','download'))
    a=p.parse_args();code=globals()[a.mode]();raise SystemExit(0 if code is None else code)
