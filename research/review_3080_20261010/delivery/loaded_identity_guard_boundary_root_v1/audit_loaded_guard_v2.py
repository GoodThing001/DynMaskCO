"""Saved-only exact file/map/case reference plus independent full model initialization."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import copy,gc,json,hashlib,sys,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def read(n):return json.loads((B/n).read_text(encoding='utf-8'))
def main():
 import psutil,numpy as np,jax,jax.numpy as jnp
 from flax import nnx
 native=prime_native_identity();psutil.Process().cpu_affinity([15]);reg=read('EXECUTION_REGISTRATION.json');result=read('RESULT.json')
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Independent8GiB barrier')
 old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
 for p in (up,old/'actual_checkpoint_engineering_v1',ext/'scripts/delivery',ext/'scripts/coldchain'):sys.path.insert(0,str(p))
 from inspect_checkpoint import load_trusted
 saved,h=load_trusted(Path(reg['checkpoint']),up)
 if h!=reg['checkpoint_sha256']:raise ValueError('Original checkpoint identity')
 cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
 params=copy.deepcopy(saved['params']);k=np.asarray(params['init_proj']['kernel'].value);params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0));model=cfg.construct_model();nnx.update(model,params);del params,saved;gc.collect()
 rng=nnx.Rngs(20261012);a=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng);b=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
 def inventory(state):return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in state.flat_state()}
 head={str((layer,part)):dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype),sha256=hashlib.sha256(np.asarray(v).tobytes()).hexdigest()) for layer,part,v in [('input','kernel',a.kernel.value),('input','bias',a.bias.value),('output','kernel',b.kernel.value),('output','bias',b.bias.value)]}
 reference_params=dict(backbone=inventory(nnx.state(model,nnx.Param)),head=head)
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','RESULT.json','SOURCE_START.json','SOURCE_END.json','INITIAL_LIVE_PARAMETERS.json','FINAL_LIVE_PARAMETERS.json','BASELINE_LOADED_FILES.json')]+[B/r['file'] for r in result['records']]
 trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain'];closure(extras,trees);s=closure(extras,trees);save(B,'AUDIT_SOURCE_START.json',s)
 checks=[]
 def check(n,v):checks.append(dict(name=n,ok=bool(v)))
 seal=read('SOURCE_START.json');initial=read('INITIAL_LIVE_PARAMETERS.json');base=read('BASELINE_LOADED_FILES.json')
 for domain,values in reference_params.items():
  for k,v in values.items():check('independent_initial_live:'+domain+':'+k,initial[domain].get(k)==v);check('independent_final_live:'+domain+':'+k,read('FINAL_LIVE_PARAMETERS.json')[domain].get(k)==v)
 for record in result['records']:
  snapshot=read(record['file'])
  for p,h in snapshot['full_reference']['files'].items():
   witness=B/('attempt_loaded_change.py' if Path(p).name=='loaded_probe.py' and record['name']=='06_loaded_byte_changed' else 'attempt_unloaded_change.py' if Path(p).name=='unloaded_registered_probe.py' and record['name']=='07_unloaded_tree_changed' else p)
   check('all_full_reference_byte_witness:'+record['name']+':'+p,sha(witness)==h)
  case=read(record['file']);check('case_original_sha:'+record['name'],sha(B/record['file'])==record['sha256'])
  actual=case['actual_loaded'];reasons=[]
  for p,h in actual['files'].items():
   if p not in seal['files']:reasons.append(('unsealed_loaded_file',p))
   elif seal['files'][p]!=h:reasons.append(('loaded_byte_changed',p))
   check('actual_loaded_full_reference_sha:'+record['name']+':'+p,case['full_reference']['files'].get(p)==h)
   # Mutation-source bytes are archived separately, never fabricated later.
   witness=B/('attempt_loaded_change.py' if Path(p).name=='loaded_probe.py' and record['name']=='06_loaded_byte_changed' else 'attempt_unloaded_change.py' if Path(p).name=='unloaded_registered_probe.py' and record['name']=='07_unloaded_tree_changed' else p)
   check('actual_loaded_byte_witness:'+record['name']+':'+p,sha(witness)==h)
  for n,binding in base['module_bindings'].items():
   if n not in actual['module_bindings'] or actual['module_bindings'][n]!=binding:reasons.append(('existing_module_origin_rebound',n))
  if case['actual_params']!=initial:reasons.append(('live_parameter_identity_changed',''))
  check('independent_guard_decision:'+record['name'],(not reasons)==case['decision']['accepted'])
  check('registered_expected_loaded:'+record['name'],case['decision']['accepted']==case['expected_loaded_accept'])
  full_changed=[p for p,h in seal['files'].items() if case['full_reference']['files'].get(p)!=h];full_new=sorted(set(case['full_reference']['files'])-set(seal['files']))
  check('independent_full_delta:'+record['name'],full_changed==case['full_delta']['changed'] and full_new==case['full_delta']['new_files'])
  full=not full_changed and not full_new and case['actual_params']==initial and not any(k=='existing_module_origin_rebound' for k,p in reasons)
  check('independent_full_gate:'+record['name'],full==case['full_accepted']);check('registered_expected_full:'+record['name'],full==case['expected_full_accept'])
  if record['name']=='04_parameter_rebound':
   altered=copy.deepcopy(initial);v=np.asarray(a.bias.value)+np.ones_like(np.asarray(a.bias.value));altered['head']["('input', 'bias')"]['sha256']=hashlib.sha256(v.tobytes()).hexdigest();check('actual_rebound_parameter_reference',altered==case['actual_params'])
  else:check('all_live_reference:'+record['name'],case['actual_params']==reference_params)
  check('clock_order:'+record['name'],list(map(float.fromhex,case['clock_hex'].values()))==sorted(map(float.fromhex,case['clock_hex'].values())))
 negative=read('07_unloaded_tree_changed.json');check('unloaded_sufficiency_counterexample_preserved',negative['decision']['accepted'] and not negative['full_accepted'] and any(Path(p).name=='unloaded_registered_probe.py' for p in negative['full_delta']['changed']))
 check('native_boundary_actually_exercised',result['native_unsealed_case_exercised']);end=closure(extras,trees);d=delta(s,end);save(B,'AUDIT_SOURCE_END.json',dict(**end,**d));check('independent_loaded_byte_closure',not d['changed'] and not d['new_files'])
 output=dict(check_count=len(checks),failure_count=sum(not c['ok'] for c in checks),checks=checks,all_file_and_live_reference_scope='Actual loaded maps/files compared to every saved original full-tree SHA and independently read original byte witnesses, all234+4 live leaves independent original initialization; zero guard or fault rerun.',original_full_tree_sufficiency=False,candidate_installed=False,stable_online_gain=False)
 save(B,'INDEPENDENT_LOADED_GUARD_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
