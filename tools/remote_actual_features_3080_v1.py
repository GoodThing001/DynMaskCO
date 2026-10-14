"""Exclusive, registered 300-second actual cold-chain feature/objective QA."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import shutil
import sys
import remote_actual_fullstep_3080_v2 as transport

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
WT = Path(r'C:\Users\mmc\.codex\worktrees\ba63\MASKCO-Main')
SOURCE = ROOT/'research/review_3080_20261010/delivery/features_objective_qa_v1'
AGENT = WT/'.arbor/delivery/research/review_3080_20261010/delivery/features_objective_qa_v1'
OLD = ROOT/'research/review_3080_20261010/delivery/actual_checkpoint_engineering_v1'
LOCAL = ROOT/'research/review_3080_20261010/delivery/remote_features_bundle_v1'
REL = 'research/review_3080_20261010/remote_actual_features_v1'
REGISTER_SHA = 'c1110955d11e4f619d55fe11ab7f711d20d1f7bf30e354ef07122f62aa5e7a8a'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def integrate():
    if sha((AGENT/'FEATURES_OBJECTIVE_ENGINEERING_REGISTRATION.json').read_bytes()) != REGISTER_SHA:
        raise ValueError('Agent registration changed')
    SOURCE.mkdir(exist_ok=False)
    records=[]
    for p in sorted(AGENT.iterdir()):
        if p.is_file():
            shutil.copy2(p, SOURCE/p.name)
            records.append({'name':p.name,'sha256':sha(p.read_bytes())})
    (SOURCE/'ROOT_INTEGRATION_IDENTITY.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'copied':len(records),'independent_execution':False}))

def deploy():
    LOCAL.mkdir(exist_ok=False)
    rawreg=(SOURCE/'FEATURES_OBJECTIVE_ENGINEERING_REGISTRATION.json').read_bytes()
    if sha(rawreg)!=REGISTER_SHA:
        raise ValueError('Registered engineering inputs changed')
    reg=json.loads(rawreg)
    files={}
    for key,expected in reg['sources'].items():
        group,relative=key.split('/',1)
        p=(ROOT/'MASKCO_code' if group=='upstream' else EXT)/relative
        raw=p.read_bytes()
        if sha(raw)!=expected:raise ValueError('Frozen local dependency changed: '+key)
        files[key]=raw
    for name,expected in reg['programs'].items():
        raw=(SOURCE/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Registered program changed: '+name)
        files['features_objective_qa_v1/'+name]=raw
    for name,expected in reg['old_dependency_programs'].items():
        raw=(OLD/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Old dependency changed: '+name)
        files['actual_checkpoint_engineering_v1/'+name]=raw
    for name,expected in reg['teacher_files'].items():
        raw=(EXT/reg['teacher_split']/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Old teacher changed: '+name)
        files['extension/'+reg['teacher_split']+'/'+name]=raw
    files['features_objective_qa_v1/FEATURES_OBJECTIVE_ENGINEERING_REGISTRATION.json']=rawreg
    supervisor=transport.SUPERVISOR.replace('1200','300')
    supervisor=supervisor.replace("base/'actual_full_step_qa.py'", "base/'features_objective_qa_v1/run_actual_features_qa.py'")
    supervisor=supervisor.replace("base/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json'", "base/'features_objective_qa_v1/FEATURES_OBJECTIVE_ENGINEERING_REGISTRATION.json'")
    launcher=transport.LAUNCHER.replace('1200','300').replace("'previous_attempt':'v1 interrupted before supervisor body; retained'", "'stage':'new_feature_and_objective_engineering_only'")
    files.update({'supervisor.py':supervisor.encode(),'launcher.py':launcher.encode(),'status.py':transport.STATUS.encode()})
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),
      'stage':'actual_features_and_objective_engineering_only','host':'3080_windows',
      'jax_backend':'cpu','cpu_affinity':[15],'maximum_seconds':300,
      'original_checkpoint':transport.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
      'checkpoint_sha256':reg['original_checkpoint_sha256'],'exclusive_remote_directory':REL,
      'research_model_eligible':False,'shared_MD_upload':False,
      'source_sha256':{name:sha(raw) for name,raw in files.items()}}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    (LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').write_bytes(files['REMOTE_ENGINEERING_REGISTRATION.json'])
    cfg=json.loads(transport.CONFIG.read_text())
    base=cfg['remotePath'].rstrip('/');target=posixpath.join(base,REL)
    client=transport.connect();sftp=client.open_sftp()
    try:
        try:sftp.stat(target)
        except FileNotFoundError:pass
        else:raise FileExistsError('Output exists; no overwrite or repeated run')
        directories=set()
        for name,raw in sorted(files.items()):
            path=posixpath.join(target,name);parent=posixpath.dirname(path);current=base
            for component in parent[len(base):].strip('/').split('/'):
                current=posixpath.join(current,component)
                if current not in directories:
                    try:sftp.stat(current)
                    except FileNotFoundError:sftp.mkdir(current)
                    directories.add(current)
            with sftp.open(path,'wb') as f:f.write(raw)
        checks={}
        for name,raw in files.items():
            with sftp.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==sha(raw)
        (LOCAL/'REMOTE_INPUT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2))
        if not all(checks.values()):raise RuntimeError('Remote input identity changed')
    finally:sftp.close();client.close()
    result=transport.run_capture(transport.build(str(transport.CONFIG),transport.WIN_ROOT,transport.PY,REL+'/launcher.py',[],'windows'),60,str(transport.CONFIG))
    (LOCAL/'LAUNCH_RESPONSE.json').write_text(json.dumps(result,indent=2))
    print(result['stdout']);return result['exit_code']

def status():
    transport.LOCAL=LOCAL;transport.REL=REL
    return transport.status()

def download():
    cfg=json.loads(transport.CONFIG.read_text());target=posixpath.join(cfg['remotePath'].rstrip('/'),REL)
    out=LOCAL/'downloaded';out.mkdir(exist_ok=False)
    client=transport.connect();sftp=client.open_sftp()
    try:
        for name in ('RUN_EXIT.json','LAUNCH.json','WORKER_IDENTITY.json','run.stdout.log','run.stderr.log'):
            with sftp.open(posixpath.join(target,name),'rb') as f:(out/name).write_bytes(f.read())
        checks={}
        manifest=json.loads((LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())
        for name,expected in manifest['source_sha256'].items():
            with sftp.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==expected
        output=sftp.listdir(posixpath.join(target,'output'))
        saved={}
        for name in output:
            path=posixpath.join(target,'output',name)
            if name.endswith('.json'):
                with sftp.open(path,'rb') as f:(out/name).write_bytes(f.read())
            elif name.endswith('_ENGINEERING_ONLY.pkl'):
                h=hashlib.sha256();count=0
                with sftp.open(path,'rb') as f:
                    while True:
                        raw=f.read(4*1024*1024)
                        if not raw:break
                        h.update(raw);count+=len(raw)
                saved[name]={'sha256':h.hexdigest(),'bytes':count,'full_remote_bytes_read':True}
        (out/'ACTUAL_END_SOURCE_CHECK.json').write_text(json.dumps(checks,indent=2))
        (out/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json').write_text(json.dumps(saved,indent=2))
    finally:sftp.close();client.close()
    print(json.dumps({'downloaded':str(out),'actual_source_unchanged':all(checks.values()),'saved_files':len(saved)}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['integrate','deploy','status','download'])
    a=p.parse_args();code=globals()[a.mode]();raise SystemExit(0 if code is None else code)
