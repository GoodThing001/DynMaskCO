"""Independent original model initialization and sequential all-file SHA."""
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
 native=prime_native_identity();psutil.Process().cpu_affinity([15]);reg=read('EXECUTION_REGISTRATION.json')
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Independent8GiB barrier')
 old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
 for p in (up,old/'actual_checkpoint_engineering_v1',ext/'scripts/delivery',ext/'scripts/coldchain'):sys.path.insert(0,str(p))
 from inspect_checkpoint import load_trusted
 saved,checkpoint_sha=load_trusted(Path(reg['checkpoint']),up)
 if checkpoint_sha!=reg['checkpoint_sha256']:raise ValueError('Original checkpoint SHA')
 cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
 params=copy.deepcopy(saved['params']);k=np.asarray(params['init_proj']['kernel'].value)
 params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0));model=cfg.construct_model();nnx.update(model,params);del params,saved;gc.collect()
 rng=nnx.Rngs(20261012);a=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng);b=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
 def inventory(state):return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in state.flat_state()}
 head={str((layer,part)):dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype),sha256=hashlib.sha256(np.asarray(value).tobytes()).hexdigest()) for layer,part,value in [('input','kernel',a.kernel.value),('input','bias',a.bias.value),('output','kernel',b.kernel.value),('output','bias',b.bias.value)]}
 reference_params=dict(backbone=inventory(nnx.state(model,nnx.Param)),head=head)
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','RESULT.json','INITIAL_LIVE_PARAMETERS.json','FINAL_LIVE_PARAMETERS.json','PARALLEL_SOURCE_COLLECTION.json','SEQUENTIAL_REFERENCE_COLLECTION.json','SOURCE_START.json','SOURCE_END.json')]
 trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain'];closure(extras,trees);start=closure(extras,trees);save(B,'AUDIT_SOURCE_START.json',start)
 checks=[]
 def check(n,ok):checks.append(dict(name=n,ok=bool(ok)))
 first=read('INITIAL_LIVE_PARAMETERS.json');last=read('FINAL_LIVE_PARAMETERS.json');candidate=read('PARALLEL_SOURCE_COLLECTION.json');sequential=read('SEQUENTIAL_REFERENCE_COLLECTION.json');s=read('SOURCE_START.json');e=read('SOURCE_END.json');result=read('RESULT.json')
 check('candidate_file_set_exact',set(candidate['files'])==set(sequential['files']));check('candidate_module_map_exact',candidate['module_files']==sequential['module_files']);check('map_sources_hashed',all(p in candidate['files'] for p in candidate['module_files'].values()))
 for p,h in candidate['files'].items():
  check('reference_sha:'+p,sequential['files'].get(p)==h);check('current_independent_sha:'+p,sha(p)==h)
 for p,h in s['files'].items():check('producer_start_end_sha:'+p,e['files'].get(p)==h)
 check('producer_no_unsealed',not e['changed'] and not e['new_files'])
 for domain,values in reference_params.items():
  check('exact_leaf_set:'+domain,set(first[domain])==set(values)==set(last[domain]))
  for k,v in values.items():check('independent_initial_parameter:'+domain+':'+k,first[domain].get(k)==v);check('independent_final_parameter:'+domain+':'+k,last[domain].get(k)==v)
 for stem in ('parallel','reference','parameter'):
  t0=float.fromhex(result['clocks'][stem+'_start_hex']);t1=float.fromhex(result['clocks'][stem+'_end_hex']);check('clock_order:'+stem,t1>=t0)
 check('single_cpu_affinity',result['cpu_affinity']==[15]);check('all_read_no_cache_scope',result['no_sha_cache'] and result['entire_file_bytes_read'] and result['original_file_set_or_scope_not_reduced'])
 end=closure(extras,trees);d=delta(start,end);save(B,'AUDIT_SOURCE_END.json',dict(**end,**d));check('independent_loaded_closure',not d['changed'] and not d['new_files']);check('independent_start_maps_hashed',all(p in start['files'] for p in start['module_files'].values()))
 output=dict(check_count=len(checks),failure_count=sum(not c['ok'] for c in checks),checks=checks,full_actual_initial_and_final_234_plus4_leaves=True,actual_parameter_count=sum(int(np.prod(v['shape'])) for values in reference_params.values() for v in values.values()),old_QA_repeated=False,model_forward_executions=0,full_scientific_or_online_budget_qualified=False,stable_online_gain=False)
 save(B,'INDEPENDENT_IDENTITY_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
