"""Qualify exact fixed128 weights in a new seven-arm deadline wrapper.

Two previously observed engineering fixtures; each arm also receives one
deliberate worker delay under a10ms deadline. No fresh scientific data.
"""
import ast,hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
from run_delivery_finite_research_root_v2 import upload_new
ROOT=t.ROOT;D=ROOT/'research/review_3080_20261010/delivery'
SOURCE=D/'final_model_budget_root_v2';LOCAL=D/'remote_final_model_budget_bundle_root_v2'
REL='research/review_3080_20261010/remote_final_model_budget_root_v2'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def deploy():
    acceptance=D/'remote_finite_research_root_v2/training_downloaded/ACCEPTED_FIXED_TRAINING.json'
    accepted=json.loads(acceptance.read_text())
    if accepted['failed']!=0 or accepted['actual_saved_tensor_reload_independently_qualified'] is not True:raise ValueError('No independent fixed128 actual tensor acceptance')
    SOURCE.mkdir(exist_ok=False);LOCAL.mkdir(exist_ok=False)
    parent=D/'model_budget_integration_root_v1/budget_qa.py';body=parent.read_text()
    body=body.replace("reg['engineering_weights']","reg['fixed_final_weights']")
    body=body.replace("actual['fixed_step']!=2 or actual['research_model_eligible'] is not False",
        "actual['fixed_step']!=128 or actual['research_model_eligible'] is not True or actual['registration_sha256']!=reg['training_registration_sha256']")
    body=body.replace('Expected engineering fixed2 weights','Expected independently accepted fixed128 research weights')
    body=body.replace("'research_model':False,","'research_model':mode in ('reconstruction_control','paired_objective'),\n                       'actual_saved_weight_sha256':reg['fixed_final_weights'][mode]['sha256'] if mode in ('reconstruction_control','paired_objective') else None,")
    marker="                if p.poll() is not None:break\n        finally:"
    assert body.count(marker)==1
    injection='''                if p.poll() is not None:break
            if p.poll() is not None:raise RuntimeError('Normal QA worker failed before timeout injection')
            # This deliberate fault is a separate engineering request, never a
            # scientific observation. Preserve the same certified incumbent.
            fixture=fixtures[0];started=time.perf_counter();deadline=started+.01
            delayed={'data':fixture['data'],'initial_routes':fixture['incumbent_routes'],
                'schedule':make_schedule(51,8,202610110072,fixture['instance_id'],fraction=.5),
                'deadline':deadline,'sleep_qa':True}
            p.stdin.write(json.dumps(delayed)+'\\n');p.stdin.flush()
            try:response=q.get(timeout=max(0.,deadline-time.perf_counter()))
            except queue.Empty:response={'kind':'timeout'}
            submitted=time.perf_counter()
            if response.get('kind')!='timeout':raise ValueError('Deliberate delayed worker incorrectly submitted')
            p.kill();p.wait(timeout=5)
            write(a.out/(arm+'_TIMEOUT_INJECTION.json'),{'arm':arm,'started':started,'deadline':deadline,
                'observed':submitted,'fault':'one second worker delay with10ms all-in request deadline',
                'timeout':True,'own_incumbent_fallback':True,'fallback_routes':fixture['incumbent_routes'],
                'fixture_id':fixture['instance_id'],'worker_pid':p.pid,'worker_reaped':p.poll() is not None,
                'model_or_scientific_budget_changed':False,'engineering_only':True})
        finally:'''
    body=body.replace(marker,injection)
    ast.parse(body);(SOURCE/'budget_qa.py').write_text(body,encoding='utf-8')
    finite=json.loads((D/'finite_training_research_root_v2/RESEARCH_REGISTRATION.json').read_text());files={}
    for key,expected in finite['dependency_sources'].items():
        group,name=key.split('/',1);raw=((ROOT/'MASKCO_code' if group=='upstream' else t.EXT)/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('Frozen source changed')
        files[key]=raw
    for name in ('features_objective_qa_v1/features_objective.py','actual_checkpoint_engineering_v1/inspect_checkpoint.py'):
        files[name]=(D/name).read_bytes()
    archive=D/'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl'
    files['TEACHER_STATES.jsonl']=archive.read_bytes();files['budget_qa.py']=body.encode()
    files['ACCEPTED_FIXED_TRAINING.json']=acceptance.read_bytes()
    registration={'version':'D-FIXED128-MODEL-TOTAL10S-AND-TIMEOUT-QA-root-v2',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(),'program_sha256':sha(files['budget_qa.py']),
        'source_sha256':finite['dependency_sources'],'teacher_archive_sha256':sha(files['TEACHER_STATES.jsonl']),
        'arms':['pretrained','random','discard','nonlearning','reconstruction_control','paired_objective','gls'],
        'fixed_final_weights':{arm:{'path':t.WIN_ROOT+r'\research\review_3080_20261010\remote_finite_research_root_v2\training_output'+'\\'+arm+'_FIXED_FINAL.pkl',
            'sha256':accepted['final_weights'][arm]} for arm in ('reconstruction_control','paired_objective')},
        'training_registration_sha256':accepted['registration_sha256'],
        'accepted_training_sha256':sha(files['ACCEPTED_FIXED_TRAINING.json']),
        'parent_engineering_program_sha256':sha(parent.read_bytes()),'fixture_ids':[900000,900001],
        'mask_seed':202610110072,'fixed_attempts_each':8,'mask_fraction':.5,'total_cap_s':10,
        'all_preparation_inference_search_certification_transmission_included':True,
        'excluded':'only weight load, static imports and fixed synthetic51 warmup before ready',
        'deliberate_timeout_each_arm':True,'fault_delay_s':1,'fault_request_deadline_s':.01,
        'no_scientific_data_generated':True,'snapshot_only':True,'online_gain':False,'maximum_seconds':600}
    regraw=json.dumps(registration,indent=2).encode();files['BUDGET_ENGINEERING_REGISTRATION.json']=regraw
    (SOURCE/'BUDGET_ENGINEERING_REGISTRATION.json').write_bytes(regraw)
    supervisor=t.SUPERVISOR.replace('1200','600').replace("base/'actual_full_step_qa.py'","base/'budget_qa.py'")
    supervisor=supervisor.replace("base/'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json'","base/'BUDGET_ENGINEERING_REGISTRATION.json'")
    extra=",'--teacher-archive',str(base/'TEACHER_STATES.jsonl'),'--expected-registration-sha',"+repr(sha(regraw))
    supervisor=supervisor.replace("'--cpu','15']","'--cpu','15'"+extra+"]")
    supervisor=supervisor.replace("worker.kill();worker.wait();code=124;failure='registered_engineering_timeout'",'''import psutil
   children=psutil.Process(worker.pid).children(recursive=True)
   for child in reversed(children):
    try:child.kill()
    except psutil.NoSuchProcess:pass
   worker.kill();worker.wait();psutil.wait_procs(children,timeout=10);code=124;failure='own_timeout_tree_reaped' ''')
    launcher=t.LAUNCHER.replace('1200','600').replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'stage':'fixed128_final_model_budget_QA'")
    files.update({'supervisor.py':supervisor.encode(),'launcher.py':launcher.encode(),'status.py':t.STATUS.encode()})
    for name in ('budget_qa.py','supervisor.py','launcher.py'):ast.parse(files[name])
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'stage':registration['version'],
        'exclusive_remote_directory':REL,'maximum_seconds':600,'cpu_affinity':[15],
        'original_checkpoint':t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
        'source_sha256':{name:sha(raw) for name,raw in files.items()},'shared_MD_uploaded':False,'fresh_research_run':False}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode()
    write(LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json',manifest)
    cfg=json.loads(t.CONFIG.read_text());target=posixpath.join(cfg['remotePath'].rstrip('/'),REL);c=t.connect()
    try:
        checks=upload_new(files,target,c);write(LOCAL/'SOURCE_UPLOAD_IDENTITY.json',checks)
        if not all(checks.values()):raise ValueError('Actual source changed before start')
    finally:c.close()
    r=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/launcher.py',[],'windows'),60,str(t.CONFIG))
    write(LOCAL/'LAUNCH_RESPONSE.json',r);print(r['stdout']);return r['exit_code']

if __name__=='__main__':
    action=sys.argv[1]
    if action=='deploy':code=deploy()
    elif action in ('status','download'):
        t.LOCAL=LOCAL;t.REL=REL;code=getattr(t,action)()
    else:raise ValueError('Unknown action')
    sys.exit(code)
