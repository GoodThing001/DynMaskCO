"""Exclusive actual10D model budget QA, no scientific data or MD upload."""
import ast,hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
ROOT=t.ROOT;SOURCE=ROOT/'research/review_3080_20261010/delivery/model_budget_integration_root_v1'
LOCAL=ROOT/'research/review_3080_20261010/delivery/remote_model_budget_bundle_root_v1'
REL='research/review_3080_20261010/remote_model_budget_root_v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def deploy():
    LOCAL.mkdir(exist_ok=False)
    finite=ROOT/'research/review_3080_20261010/delivery/finite_training_preparation_v1'
    reg0=json.loads((finite/'FINITE_ENGINEERING_REGISTRATION.json').read_text());files={}
    for key,expected in reg0['dependency_sources'].items():
        group,name=key.split('/',1);p=(ROOT/'MASKCO_code' if group=='upstream' else t.EXT)/name
        raw=p.read_bytes()
        if sha(raw)!=expected:raise ValueError('Dependency changed '+key)
        files[key]=raw
    for name in ('features_objective_qa_v1/features_objective.py','actual_checkpoint_engineering_v1/inspect_checkpoint.py'):
        files[name]=(SOURCE.parent/name).read_bytes()
    archive=finite/'ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl';files['TEACHER_STATES.jsonl']=archive.read_bytes()
    files['budget_qa.py']=(SOURCE/'budget_qa.py').read_bytes();ast.parse(files['budget_qa.py'])
    weights=json.loads((ROOT/'research/review_3080_20261010/delivery/remote_finite_training_bundle_v1/downloaded/INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json').read_text())
    science={'version':'D-ACTUAL-10D-TOTAL10S-ENGINEERING-root-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'program_sha256':sha(files['budget_qa.py']),'source_sha256':reg0['dependency_sources'],
        'teacher_archive_sha256':sha(files['TEACHER_STATES.jsonl']),
        'arms':['pretrained','random','discard','nonlearning','reconstruction_control','paired_objective','gls'],
        'engineering_weights':{arm:{'path':t.WIN_ROOT+'\\research\\review_3080_20261010\\remote_finite_training_qa_v1\\output\\'+arm+'_FIXED_FINAL.pkl',
              'sha256':weights[arm+'_FIXED_FINAL.pkl']['sha256']} for arm in ('reconstruction_control','paired_objective')},
        'fixture_ids':[900000,900001],'mask_seed':202610110072,'fixed_attempts_each':8,'mask_fraction':.5,
        'total_cap_s':10,'all_preparation_inference_search_certification_transmission_included':True,
        'excluded':'only original weight loading and fixed synthetic51 prewarming before ready',
        'snapshot_only':True,'new_research_data_generated':False,'research_model_eligible':False,'online_gain':False,'maximum_seconds':600}
    regraw=json.dumps(science,indent=2).encode();files['BUDGET_ENGINEERING_REGISTRATION.json']=regraw
    (SOURCE/'BUDGET_ENGINEERING_REGISTRATION.json').write_bytes(regraw)
    supervisor=t.SUPERVISOR.replace('1200','600').replace("base/'actual_full_step_qa.py'","base/'budget_qa.py'")
    supervisor=supervisor.replace("base/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json'","base/'BUDGET_ENGINEERING_REGISTRATION.json'")
    extra=",'--teacher-archive',str(base/'TEACHER_STATES.jsonl'),'--expected-registration-sha',"+repr(sha(regraw))
    supervisor=supervisor.replace("'--cpu','15']","'--cpu','15'"+extra+"]")
    supervisor=supervisor.replace("worker.kill();worker.wait();code=124;failure='registered_engineering_timeout'", """import psutil
   children=psutil.Process(worker.pid).children(recursive=True)
   for child in reversed(children):
    try:child.kill()
    except psutil.NoSuchProcess:pass
   worker.kill();worker.wait();psutil.wait_procs(children,timeout=10);code=124;failure='registered_engineering_timeout_owned_tree_reaped'""")
    launcher=t.LAUNCHER.replace('1200','600').replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'stage':'actual_10D_total10s_engineering'")
    ast.parse(supervisor);files.update({'supervisor.py':supervisor.encode(),'launcher.py':launcher.encode(),'status.py':t.STATUS.encode()})
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'stage':science['version'],
        'maximum_seconds':600,'cpu_affinity':[15],'exclusive_remote_directory':REL,
        'original_checkpoint':t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt','source_sha256':{k:sha(v) for k,v in files.items()},
        'research_model_eligible':False,'shared_MD_upload':False}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    (LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').write_bytes(files['REMOTE_ENGINEERING_REGISTRATION.json'])
    config=json.loads(t.CONFIG.read_text());base=config['remotePath'].rstrip('/');target=posixpath.join(base,REL)
    c=t.connect();s=c.open_sftp()
    try:
        try:s.stat(target)
        except FileNotFoundError:pass
        else:raise FileExistsError('Already exists; never rerun or overwrite')
        dirs=set()
        for name,raw in sorted(files.items()):
            path=posixpath.join(target,name);current=base
            for component in posixpath.dirname(path)[len(base):].strip('/').split('/'):
                current=posixpath.join(current,component)
                if current in dirs:continue
                try:s.stat(current)
                except FileNotFoundError:s.mkdir(current)
                dirs.add(current)
            with s.open(path,'wb') as f:f.write(raw)
        checks={}
        for name,raw in files.items():
            with s.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==sha(raw)
        (LOCAL/'REMOTE_INPUT_ACCEPTANCE.json').write_text(json.dumps(checks,indent=2))
        if not all(checks.values()):raise ValueError('Remote inputs changed')
    finally:s.close();c.close()
    result=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/launcher.py',[],'windows'),60,str(t.CONFIG))
    (LOCAL/'LAUNCH_RESPONSE.json').write_text(json.dumps(result,indent=2));print(result['stdout']);return result['exit_code']
if __name__=='__main__':
    action=sys.argv[1]
    if action=='deploy':code=deploy()
    elif action in ('status','download'):
        t.LOCAL=LOCAL;t.REL=REL;code=getattr(t,action)()
    else:raise ValueError('Unknown action')
    sys.exit(code)
