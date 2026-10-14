"""Fresh-process fixed-final tensor verification; never trains or selects weights."""
import ast, hashlib, json, posixpath, sys
from datetime import datetime, timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
from run_delivery_finite_research_root_v2 import upload_new
ROOT=t.ROOT
B=ROOT/'research/review_3080_20261010/delivery/remote_finite_research_root_v2'
LOCAL=ROOT/'research/review_3080_20261010/delivery/independent_training_tensors_root_v2'
TRAIN_REL='research/review_3080_20261010/remote_finite_research_root_v2'
REL='research/review_3080_20261010/independent_training_tensors_root_v2'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)

HELPER=r'''import copy,gc,hashlib,json,os,pickle,sys,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import psutil
psutil.Process().cpu_affinity([15])
base=Path(__file__).resolve().parent
reg=json.loads((base/'REGISTRATION.json').read_text())
train=Path(reg['training_directory'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
if sha(train/'training_output/TRAINING_RESULT.json')!=reg['training_result_sha256']:raise ValueError('Training result changed')
for name,expected in reg['frozen_sources'].items():
 if sha(train/name)!=expected:raise ValueError('Actual frozen dependency changed:'+name)
sys.path.insert(0,str(train/'actual_checkpoint_engineering_v1'))
sys.path.insert(0,str(train/'features_objective_qa_v1'))
for name in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(train/'extension/scripts'/name))
from inspect_checkpoint import load_trusted
saved,original_sha=load_trusted(Path(reg['original_checkpoint']),train/'upstream')
import numpy as np,jax.numpy as jnp
from flax import nnx
from features_objective import extend_projection,make_features,UNITS
from modules.functional import coord_normalize
from delivery_contract import DeliveryContract
cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10
fresh=cfg.construct_model();graph=nnx.graphdef(fresh)
initial=copy.deepcopy(saved['params'])
initial['init_proj']['kernel'].value=jnp.asarray(extend_projection(initial['init_proj']['kernel'].value))
def inventory(params):
 return {str(p):dict(sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest(),
    shape=list(np.asarray(v.value).shape),elements=int(np.asarray(v.value).size),
    dtype=str(np.asarray(v.value).dtype),finite=bool(np.isfinite(np.asarray(v.value)).all()))
    for p,v in params.flat_state()}
initial_inventory=inventory(initial);del saved,initial,fresh;gc.collect()
rows=[json.loads(s) for s in (train/'teacher_output/TEACHER_STATES.jsonl').read_text().splitlines()]
teacher=rows[0];data=teacher['data'];c=DeliveryContract()
coords=np.asarray(coord_normalize(jnp.asarray(data['coords'],jnp.float32)[None]))[0]
features=jnp.asarray(make_features(data,c,coords,UNITS,c.identity()),jnp.float32)
context=jnp.asarray(np.asarray(teacher['partial_adjacency'],np.float32)[None])
timestep=jnp.asarray([float(np.asarray(teacher['partial_adjacency'])[1:].sum()/100.)],jnp.float32)
encode=nnx.jit(lambda m,x:m.encode(x));decode=nnx.jit(lambda m,h,t,a:m.decode(h,t,a,target='logit'))
result={'scope':'independent fresh process actual saved tensors and output reload; not deployment or gain',
 'original_checkpoint_sha256':original_sha,'initial_inventory':initial_inventory,'arms':{},'fixed_step':128,
 'research_registration_sha256':reg['research_registration_sha256'],'training_result_sha256':reg['training_result_sha256'],
 'teacher_state_sha256':teacher['state_sha256'],'dynamic_online_qualified':False,'stable_online_gain':False}
for arm,expected in reg['weights'].items():
 p=train/'training_output'/(arm+'_FIXED_FINAL.pkl')
 if sha(p)!=expected['sha256']:raise ValueError('Actual saved weight mismatch')
 with p.open('rb') as f:loaded=pickle.load(f)
 if loaded['fixed_step']!=128 or loaded['research_model_eligible'] is not True or loaded['registration_sha256']!=reg['research_registration_sha256']:raise ValueError('Not registered fixed final')
 if loaded['model_config'].dtype!='float32' or loaded['model_config'].encoder_input_dim!=10:raise ValueError('Wrong complete model config')
 inventory1=inventory(loaded['params']);model=nnx.merge(graph,loaded['params'])
 h=encode(model,features);a=np.asarray(decode(model,h,timestep,context))
 b=np.asarray(decode(model,h,timestep,jnp.zeros_like(context)))
 offdiag=~np.eye(51,dtype=bool)
 if not np.isfinite(a[:,offdiag]).all() or np.array_equal(a[:,offdiag],b[:,offdiag]):raise ValueError('Masked model inactive or nonfinite')
 del loaded,model,h;gc.collect()
 with p.open('rb') as f:loaded2=pickle.load(f)
 model2=nnx.merge(graph,loaded2['params']);h2=encode(model2,features);a2=np.asarray(decode(model2,h2,timestep,context))
 same=bool(np.array_equal(a,a2))
 if not same or inventory(loaded2['params'])!=inventory1:raise ValueError('Independent saved reload mismatch')
 np.savez_compressed(base/(arm+'_OUTPUTS.npz'),masked=a,unmasked=b,reloaded=a2)
 changes={group:sum(v['sha256']!=initial_inventory[p]['sha256'] for p,v in inventory1.items()
     if group=='full' or p.startswith("('"+group+"',")) for group in ('full','encoder','decoder')}
 if len(inventory1)!=234 or sum(v['elements'] for v in inventory1.values())!=76263424 or not all(v['finite'] for v in inventory1.values()):raise ValueError('Not full finite model')
 if any(changes[g]==0 for g in changes):raise ValueError('Complete model update missing')
 result['arms'][arm]={'actual_sha256':sha(p),'actual_size_bytes':p.stat().st_size,'final_inventory':inventory1,
   'actual_initial_digest':digest({p:v['sha256'] for p,v in initial_inventory.items()}),
   'actual_final_digest':digest({p:v['sha256'] for p,v in inventory1.items()}),
   'actual_changes':changes,'saved_reload_outputs_exact':same,'retained_context_output_active':True,
   'offdiag_context_max_abs_difference':float(np.max(np.abs(a[:,offdiag]-b[:,offdiag]))),
   'output_archive_sha256':sha(base/(arm+'_OUTPUTS.npz'))}
 del loaded2,model2,h2,a,b,a2;gc.collect()
result['actual_source_end']={p:sha(train/p)==expected for p,expected in reg['frozen_sources'].items()}
with (base/'ACTUAL_FIXED_TENSOR_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({'fixed128_actual_saved_tensors_verified':True,'dynamic_online_gain':False}),flush=True)
'''

def deploy():
    cfg=json.loads(t.CONFIG.read_text());remote_base=cfg['remotePath'].rstrip('/')
    c=t.connect();s=c.open_sftp()
    target=posixpath.join(remote_base,REL);train=posixpath.join(remote_base,TRAIN_REL)
    try:
        try:
            with s.open(posixpath.join(train,'training_RUN_EXIT.json'),'rb') as f:exit=json.loads(f.read())
        except FileNotFoundError:
            print(json.dumps({'pending':True,'reason':'training not complete; no verifier created or launched'}));return 0
        if exit['exit_code']!=0:raise ValueError('Training failed; preserve failure before any recovery')
        with s.open(posixpath.join(train,'training_output/TRAINING_RESULT.json'),'rb') as f:raw=f.read()
        result=json.loads(raw);weights={}
        for arm in ('reconstruction_control','paired_objective'):
            path=posixpath.join(train,'training_output',arm+'_FIXED_FINAL.pkl');h=hashlib.sha256();size=0
            with s.open(path,'rb') as f:
                # Pipeline read requests; still hash every actual returned byte.
                # Older Paramiko accepts only file_size; all versions keep the
                # same SHA-pinned checkpoint and never modify remote files.
                import inspect
                if 'max_concurrent_requests' in inspect.signature(f.prefetch).parameters:
                    f.prefetch(file_size=s.stat(path).st_size,max_concurrent_requests=32)
                else:
                    f.prefetch(file_size=s.stat(path).st_size)
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
            if h.hexdigest()!=result['arms'][arm]['checkpoint_sha256']:raise ValueError('Independent actual complete saved-byte identity failed')
            weights[arm]={'sha256':h.hexdigest(),'bytes':size,'whole_remote_bytes_read':True}
        sources=json.loads((B/'REMOTE_ENGINEERING_REGISTRATION.json').read_text())
        registration={'registered_utc':datetime.now(timezone.utc).isoformat(),'training_directory':t.WIN_ROOT+'\\'+TRAIN_REL.replace('/','\\'),
            'original_checkpoint':t.WIN_ROOT+r'\MASKCO_code\ckpts\cvrp100.ckpt',
            'research_registration_sha256':result['registration_sha256'],'training_result_sha256':sha(raw),
            'frozen_sources':sources['source_sha256'],'weights':weights,'maximum_seconds':300,
            'new_training_or_data':False,'shared_MD_uploaded':False,'model_selection':False}
        LOCAL.mkdir(exist_ok=False);write(LOCAL/'INDEPENDENT_WHOLE_SAVED_WEIGHT_BYTES.json',weights)
        write(LOCAL/'REGISTRATION.json',registration)
    finally:s.close();c.close()
    supervisor='''import hashlib,json,os,pathlib,psutil,subprocess,sys,time
base=pathlib.Path(__file__).resolve().parent
start=time.time();env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8')
with (base/'stdout.log').open('xb') as stdout,(base/'stderr.log').open('xb') as stderr:
 p=subprocess.Popen([sys.executable,'-B',str(base/'verify_saved.py')],cwd=base,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
 psutil.Process(p.pid).cpu_affinity([15])
 try:code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  children=psutil.Process(p.pid).children(recursive=True)
  for child in reversed(children):
   try:child.kill()
   except psutil.NoSuchProcess:pass
  p.kill();p.wait();psutil.wait_procs(children,timeout=10);code=124
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'wall_seconds':time.time()-start,'own_worker_reaped':p.poll() is not None},f)
'''
    files={'verify_saved.py':HELPER.encode(),'REGISTRATION.json':(LOCAL/'REGISTRATION.json').read_bytes(),
        'supervisor.py':supervisor.encode(),'launcher.py':t.LAUNCHER.replace('1200','300').encode(),'status.py':t.STATUS.encode()}
    for name in ('verify_saved.py','supervisor.py','launcher.py'):ast.parse(files[name])
    c=t.connect()
    try:
        checks=upload_new(files,target,c);write(LOCAL/'SOURCE_UPLOAD_IDENTITY.json',checks)
        if not all(checks.values()):raise ValueError('Verifier source changed')
    finally:c.close()
    r=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/launcher.py',[],'windows'),60,str(t.CONFIG))
    write(LOCAL/'LAUNCH_RESPONSE.json',r);print(r['stdout']);return r['exit_code']

def status():
    r=t.run_capture(t.build(str(t.CONFIG),t.WIN_ROOT,t.PY,REL+'/status.py',[],'windows'),60,str(t.CONFIG))
    write(LOCAL/('STATUS_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),r)
    print(r['stdout']);return r['exit_code']

def download():
    cfg=json.loads(t.CONFIG.read_text());base=posixpath.join(cfg['remotePath'].rstrip('/'),REL)
    c=t.connect();s=c.open_sftp()
    try:
        with s.open(posixpath.join(base,'RUN_EXIT.json'),'rb') as f:raw=f.read()
        if json.loads(raw)['exit_code']!=0:raise ValueError('Verifier failed; preserve remote failure')
        out=LOCAL/'downloaded';out.mkdir(exist_ok=False)
        for name in s.listdir(base):
            if name.endswith(('.json','.npz','.log','.py')):
                with s.open(posixpath.join(base,name),'rb') as f:(out/name).write_bytes(f.read())
        source_check={'actual_verifier_source_matches_prepared_helper':
            sha((out/'verify_saved.py').read_bytes())==sha(HELPER.encode()),
            'actual_registration_matches_local_frozen_registration':
            sha((out/'REGISTRATION.json').read_bytes())==sha((LOCAL/'REGISTRATION.json').read_bytes())}
        write(out/'INDEPENDENT_VERIFIER_ACTUAL_SOURCE_CHECK.json',source_check)
        if not all(source_check.values()):raise ValueError('Independent verifier source or registration changed')
        print(json.dumps({'downloaded':str(out),'requires_local_independent_inventory_and_output_audit':True}));return 0
    finally:s.close();c.close()

if __name__=='__main__':
    sys.exit({'deploy':deploy,'status':status,'download':download}[sys.argv[1]]())
