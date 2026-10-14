import copy,gc,hashlib,json,os,pickle,sys,time
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
