"""Preregister implementation of existing16 snapshot diagnostic; fixed final weights."""
import ast,hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
from run_delivery_finite_research_root_v2 import upload_new
ROOT=t.ROOT;D=ROOT/'research/review_3080_20261010/delivery'
LOCAL=D/'remote_fresh_development_root_v1';SOURCE=D/'fresh_development_root_v1'
REL='research/review_3080_20261010/remote_fresh_development_root_v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)

BODY=r'''
def main(a,reg):
    import numpy as np,psutil
    psutil.Process().cpu_affinity([15]);bootstrap(a.root)
    if sha(a.registration)!=a.expected_registration_sha or sha(Path(__file__))!=reg['program_sha256']:raise ValueError('Frozen program/registration changed')
    for key,expected in reg['source_sha256'].items():
        group,name=key.split('/',1)
        if sha((a.upstream if group=='upstream' else a.root)/name)!=expected:raise ValueError('Frozen dependency changed')
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_equal_schedule import generate,make_schedule
    c=DeliveryContract();a.out.mkdir(exist_ok=False)
    fixtures=[]
    for i in reg['instance_ids']:
        data,routes=generate(reg['data_seed'],i)
        initial=evaluate(data,routes,c)
        if not initial['hard'] or not initial['complete']:raise ValueError('True initial hard failure; no screening')
        fixture={'instance_id':i,'data':data,'incumbent_routes':routes,'initial_report':initial}
        fixtures.append(fixture);write(a.out/('fixture_'+str(i)+'.json'),fixture)
    workers={};records=[]
    def spawn(arm):
        q=queue.Queue();errors=[]
        argv=[sys.executable,'-B',str(Path(__file__)),'--worker','--arm',arm,'--root',str(a.root),
          '--upstream',str(a.upstream),'--checkpoint',str(a.checkpoint),'--registration',str(a.registration),'--cpu','15']
        p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        def reader():
            try:
                for line in p.stdout:q.put(json.loads(line))
            except Exception as e:q.put({'kind':'reader_error','error':str(e)})
            finally:q.put({'kind':'closed'})
        threading.Thread(target=reader,daemon=True).start()
        threading.Thread(target=lambda:errors.extend(p.stderr.readlines()),daemon=True).start()
        try:ready=q.get(timeout=120)
        except Exception:
            p.kill();p.wait();raise
        if ready.get('kind')!='ready':
            p.kill();p.wait();raise RuntimeError('Activation failed:'+arm+str(errors))
        if ready['actual_cpu_affinity']!=[15] or ready['contract_hash']!=c.identity():
            p.kill();p.wait();raise ValueError('Resource/contract mismatch')
        return {'p':p,'q':q,'ready':ready,'stderr':errors}
    def reap(entry):
        p=entry['p']
        if p.poll() is None:p.kill()
        p.wait(timeout=5)
        return {'pid':p.pid,'reaped':p.poll() is not None,'stderr':entry['stderr']}
    retired=[]
    try:
        for arm in reg['arms']:workers[arm]=spawn(arm)
        for fixture,order in zip(fixtures,reg['balanced_arm_order']):
            for position,arm in enumerate(order):
                if workers[arm]['p'].poll() is not None:workers[arm]=spawn(arm)
                entry=workers[arm];p=entry['p'];q=entry['q']
                t0=time.perf_counter();deadline=t0+10.
                request={'data':fixture['data'],'initial_routes':fixture['incumbent_routes'],
                  'schedule':make_schedule(51,8,reg['mask_seed'],fixture['instance_id'],fraction=.5),'deadline':deadline}
                p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
                try:response=q.get(timeout=max(0.,deadline-time.perf_counter()))
                except queue.Empty:response={'kind':'timeout'}
                submitted=time.perf_counter();late=submitted>deadline or response.get('kind')!='response'
                if response.get('kind') in ('closed','reader_error'):raise RuntimeError('Actual worker failed; preserve partial:'+arm)
                out=response.get('out');fallback=late or (out is not None and out.get('timeout',False))
                final_routes=fixture['incumbent_routes'] if fallback else out['routes']
                final_report=fixture['initial_report'] if fallback else out['report']
                record={'arm':arm,'instance_id':fixture['instance_id'],'order_position':position,
                  'started':t0,'deadline':deadline,'submitted':submitted,'wall_s':submitted-t0,
                  'timeout':fallback,'parent_deadline_missed':late,'own_incumbent_fallback':fallback,
                  'worker_pid':p.pid,'ready':entry['ready'],'actual_result':out,
                  'final_routes':final_routes,'final_report':final_report,
                  'actual_charged_parent_cpu_affinity':psutil.Process().cpu_affinity()}
                # Install/fallback happens at submitted; writing immutable evidence is outside decision budget.
                records.append(record);write(a.out/(arm+'_'+str(fixture['instance_id'])+'.json'),record)
                if late:retired.append(reap(entry))
            write(a.out/('progress_'+str(fixture['instance_id'])+'.json'),{'complete_instances':fixture['instance_id']+1,'records':len(records)})
    finally:
        retired +=[reap(entry) for entry in workers.values()]
        write(a.out/'WORKERS_REAPED.json',retired)
    end={key:sha((a.upstream if key.split('/',1)[0]=='upstream' else a.root)/key.split('/',1)[1]) for key in reg['source_sha256']}
    write(a.out/'RESULT.json',{'registration_sha256':sha(a.registration),'records':records,'source_end':end,
      'source_unchanged':end==reg['source_sha256'],'snapshot_only':True,'engineering_only':False,
      'development_diagnostic_only':True,'checkpoint_selection':False,'confirmation_enabled':False,'stable_online_gain':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--arm')
    for key in ('root','upstream','checkpoint','registration','out'):ap.add_argument('--'+key,type=Path)
    ap.add_argument('--cpu',type=int,default=15);ap.add_argument('--expected-registration-sha');a=ap.parse_args()
    reg=json.loads(a.registration.read_text())
    if a.worker:worker(a,reg)
    else:main(a,reg)
'''

def deploy():
    accepted=D/'remote_final_model_budget_bundle_root_v3/downloaded/ACCEPTED_FINAL_MODEL_BUDGET.json'
    if json.loads(accepted.read_text())['independent_acceptance'] is not True:raise ValueError('Budget acceptance missing')
    final=D/'remote_finite_research_root_v2/training_downloaded/ACCEPTED_FIXED_TRAINING.json'
    if json.loads(final.read_text())['failed']!=0:raise ValueError('Final128 qualification missing')
    SOURCE.mkdir(exist_ok=False);LOCAL.mkdir(exist_ok=False)
    prefix=(D/'remote_final_model_budget_bundle_root_v3/actual_executed_bytes_r2/budget_qa.py').read_text().split('def main(a,reg):')[0]
    prefix=prefix.replace("saved,original_sha=load_trusted(a.checkpoint,a.upstream)","saved,original_sha=load_trusted(a.checkpoint,a.upstream)\n        if original_sha!=reg['original_checkpoint_sha256']:raise ValueError('Original checkpoint identity mismatch')")
    body=prefix+BODY;ast.parse(body);raw=body.encode();(SOURCE/'development.py').write_bytes(raw)
    sciencep=D/'finite_training_research_root_v2/RESEARCH_REGISTRATION.json';science=json.loads(sciencep.read_text())
    qa=json.loads((D/'final_model_budget_root_v3/BUDGET_ENGINEERING_REGISTRATION.json').read_text())
    import numpy as np
    arms=['paired_objective','reconstruction_control','pretrained','random','nonlearning','gls']
    rng=np.random.default_rng(science['seeds']['arm_order']);base=list(rng.permutation(arms));order=[]
    for block in range(3):
        shifts=list(map(int,rng.permutation(6)))
        for shift in shifts:order.append(base[shift:]+base[:shift])
    order=order[:16]
    for arm in arms:
        counts=[sum(row[pos]==arm for row in order) for pos in range(6)]
        if max(counts)-min(counts)>1:raise ValueError('Arm positions not balanced')
    registration={**qa,'version':'D-FIXED128-FRESH-DEV16-root-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
      'program_sha256':sha(raw),'science_registration_sha256':sha(sciencep.read_bytes()),
      'actual_budget_acceptance_sha256':sha(accepted.read_bytes()),'actual_fixed_training_acceptance_sha256':sha(final.read_bytes()),
      'arms':arms,'arm_names':dict(zip(arms,['paired_objective','reconstruction_control','original_warmstart','random10D_untrained','nonlearning_fixed8','GLS'])),
      'original_checkpoint_sha256':science['original_checkpoint_sha256'],
      'data_seed':science['seeds']['development'],'mask_seed':science['seeds']['development'],
      'instance_ids':list(range(16)),'balanced_arm_order':order,'arm_order_seed':science['seeds']['arm_order'],
      'data_generation_is_static_input_setup':True,'no_scientific_data_generated':False,
      'dynamic_claim':False,'confirmation_enabled':False,'checkpoint_selection':False,'maximum_seconds':1800,
      'deliberate_timeout_each_arm':False,'mask_fraction':.5,'fixed_attempts_each':8,
      'warmup_exclusions':'only static import, weight load, synthetic51 activation before ready; no fresh instance inference excluded',
      'scope':'16 new pre-dispatch snapshots diagnostic only; no hyperparameter or checkpoint selection'}
    regraw=json.dumps(registration,indent=2).encode();(SOURCE/'DEVELOPMENT_REGISTRATION.json').write_bytes(regraw)
    files={'development.py':raw,'DEVELOPMENT_REGISTRATION.json':regraw,
      'RESEARCH_REGISTRATION.json':sciencep.read_bytes(),'ACCEPTED_FIXED_TRAINING.json':final.read_bytes(),'ACCEPTED_FINAL_MODEL_BUDGET.json':accepted.read_bytes()}
    for key,expected in science['dependency_sources'].items():
        group,name=key.split('/',1);payload=((ROOT/'MASKCO_code' if group=='upstream' else t.EXT)/name).read_bytes()
        if sha(payload)!=expected:raise ValueError('Frozen source changed')
        files[key]=payload
    for name in ('features_objective_qa_v1/features_objective.py','actual_checkpoint_engineering_v1/inspect_checkpoint.py'):files[name]=(D/name).read_bytes()
    supervisor='''import json,os,pathlib,psutil,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
start=time.time();code=1;failure=None
env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
argv=[sys.executable,'-B',str(base/'development.py'),'--root',str(base/'extension'),'--upstream',str(base/'upstream'),'--checkpoint',CHECKPOINT,'--registration',str(base/'DEVELOPMENT_REGISTRATION.json'),'--out',str(base/'output'),'--expected-registration-sha',REGSHA]
try:
 with (base/'run.stdout.log').open('xb') as stdout,(base/'run.stderr.log').open('xb') as stderr:
  p=subprocess.Popen(argv,cwd=base,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
  with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'pid':p.pid,'supervisor_pid':os.getpid(),'argv':argv,'maximum_seconds':1800},f)
  psutil.Process(p.pid).cpu_affinity([15])
  try:code=p.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   children=psutil.Process(p.pid).children(recursive=True)
   for child in reversed(children):
    try:child.kill()
    except psutil.NoSuchProcess:pass
   p.kill();p.wait();psutil.wait_procs(children,timeout=10);code=124;failure='own_registered_timeout'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_seconds':time.time()-start,'supervisor_pid':os.getpid()},f)
'''.replace('CHECKPOINT',repr(t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt')).replace('REGSHA',repr(sha(regraw)))
    launcher=t.LAUNCHER.replace('1200','1800').replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'stage':'fresh16_snapshot_development_root_v1'")
    files.update({'supervisor.py':supervisor.encode(),'launcher.py':launcher.encode(),'status.py':t.STATUS.encode()})
    for name in ('development.py','supervisor.py','launcher.py'):ast.parse(files[name])
    manifest={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'exclusive_remote_directory':REL,
      'source_sha256':{name:sha(payload) for name,payload in files.items()},'maximum_seconds':1800,
      'snapshot_only':True,'shared_MD_uploaded':False,'fresh_development':True,'confirmation_enabled':False}
    files['REMOTE_ENGINEERING_REGISTRATION.json']=json.dumps(manifest,indent=2).encode();write(LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json',manifest)
    cfg=json.loads(t.CONFIG.read_text());target=posixpath.join(cfg['remotePath'].rstrip('/'),REL);c=t.connect()
    try:
        checks=upload_new(files,target,c);write(LOCAL/'SOURCE_UPLOAD_IDENTITY.json',checks)
        if not all(checks.values()):raise ValueError('Actual source identity failed')
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
