"""One exact-scope identity collection scheduler, no file cache or model forward."""
import ast,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R/'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1';T=R/'research/review_3080_20261010/delivery/primitive_schedule_semantics_root_v1';O=R/'research/review_3080_20261010/delivery/parallel_identity_collection_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (O/n).write_text(s,encoding='utf-8',newline='\n')
RUN='''import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import json,time,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from parallel_runtime_identity_v1 import closure_parallel
from startup_native_identity_v1 import prime_native_identity
def main():
 import psutil
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB resource barrier')
 from delivery_score_split_engine_v1 import InitialEngine
 engine=InitialEngine(B,reg,'pretrained_initial')
 initial=dict(backbone=engine.inventory(engine.model),head=engine.inventory(engine.head))
 save(B,'INITIAL_LIVE_PARAMETERS.json',initial)
 pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='identity_cpu15')
 try:
  # Native/stdlib/thread startup only, never model/sample or file hash warmup.
  barrier=threading.Barrier(4);futures=[pool.submit(barrier.wait,20) for _ in range(4)]
  for f in futures:f.result(timeout=30)
  extras=engine.extras;trees=engine.trees;closure(extras,trees)
  start=closure(extras,trees);save(B,'SOURCE_START.json',start)
  t0=time.perf_counter();candidate=closure_parallel(extras,trees,pool);t1=time.perf_counter()
  save(B,'PARALLEL_SOURCE_COLLECTION.json',candidate)
  t2=time.perf_counter();reference=closure(extras,trees);t3=time.perf_counter()
  save(B,'SEQUENTIAL_REFERENCE_COLLECTION.json',reference)
  p0=time.perf_counter();final=dict(backbone=engine.inventory(engine.model),head=engine.inventory(engine.head));p1=time.perf_counter()
  save(B,'FINAL_LIVE_PARAMETERS.json',final)
  end=closure(extras,trees);d=delta(start,end);save(B,'SOURCE_END.json',dict(**end,**d))
  result=dict(candidate_files=len(candidate['files']),reference_files=len(reference['files']),file_sha_map_exact=candidate['files']==reference['files'],module_map_exact=candidate['module_files']==reference['module_files'],candidate_source_delta=delta(start,candidate),source_qualified=not d['changed'] and not d['new_files'],all_live_parameters_unchanged=initial==final,backbone_leaves=len(initial['backbone']),head_leaves=len(initial['head']),parallel_seconds=t1-t0,sequential_reference_seconds=t3-t2,live_parameter_inventory_seconds=p1-p0,clocks=dict(parallel_start_hex=t0.hex(),parallel_end_hex=t1.hex(),reference_start_hex=t2.hex(),reference_end_hex=t3.hex(),parameter_start_hex=p0.hex(),parameter_end_hex=p1.hex()),io_threads=4,cpu_affinity=psutil.Process().cpu_affinity(),no_sha_cache=True,entire_file_bytes_read=True,original_file_set_or_scope_not_reduced=True,model_forwards=0,C0_solver_calls=0,optimizer_updates=0,scientific_family3_started=False,full_request_budget_qualified=False,stable_online_gain=False,os_cache_fairness_controlled=False)
  save(B,'RESULT.json',result)
  return 0 if result['file_sha_map_exact'] and result['module_map_exact'] and result['all_live_parameters_unchanged'] and result['source_qualified'] else 1
 finally:pool.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''
AUDIT='''"""Independent original model initialization and sequential all-file SHA."""
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
'''
def main():
 O.mkdir(parents=True,exist_ok=False)
 for n in ('action_head_runtime_v1.py','startup_native_identity_v1.py','timed_evidence_io_v1.py','delivery_score_split_engine_v1.py','features_objective.py','delivery_actual_action_head_v1.py','delivery_actual_score_chain_v1.py','delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py'):write(n,(OLD/n).read_text(encoding='utf-8'))
 raw=(OLD/'action_head_runtime_v1.py').read_text(encoding='utf-8');raw=raw.replace('def closure(extra,trees):','def closure_parallel(extra,trees,executor):').replace("'files':{str(p):sha(p) for p in sorted(files)}","'files':dict(executor.map(_hash_pair,sorted(files)))")
 raw+='\ndef _hash_pair(p):\n    return str(p),sha(p)\n';write('parallel_runtime_identity_v1.py',raw)
 write('run_parallel_identity_v1.py',RUN);write('audit_parallel_identity_v1.py',AUDIT)
 mapping={'run_primitive_schedule_v1.py':'run_parallel_identity_v1.py','audit_primitive_schedule_v1.py':'audit_parallel_identity_v1.py','supervise_primitive_schedule_v1.py':'supervise_parallel_identity_v1.py','primitive_remote_launcher_v1.py':'identity_remote_launcher_v1.py','primitive_remote_status_v1.py':'identity_remote_status_v1.py','INDEPENDENT_SEMANTICS_AUDIT.json':'INDEPENDENT_IDENTITY_AUDIT.json'}
 for old,new in [(k,v) for k,v in mapping.items() if k.endswith('.py') and k not in ('run_primitive_schedule_v1.py','audit_primitive_schedule_v1.py')]:
  s=(T/old).read_text(encoding='utf-8')
  for k,v in mapping.items():s=s.replace(k,v)
  write(new,s)
 transport=R/'tools/transport_delivery_parallel_identity_v1.py';s=(R/'tools/transport_delivery_primitive_schedule_v1.py').read_text(encoding='utf-8').replace('primitive_schedule_semantics_root_v1','parallel_identity_collection_root_v1').replace('primitive_remote_launcher_v1.py','identity_remote_launcher_v1.py').replace('primitive_remote_status_v1.py','identity_remote_status_v1.py');transport.write_text(s,encoding='utf-8',newline='\n');ast.parse(s)
 prior=json.loads((OLD/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'));reg={k:prior[k] for k in ('old_remote_root','checkpoint','checkpoint_sha256','contract_hash','local_transport_helpers')}
 reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_FIXED4_IO_THREADS_FULL_IDENTITY_COLLECTION_COMPONENT',execution_authorized=True,independent_timed_request_checker_sealed=True,
  package_files={p.name:sha(p) for p in O.iterdir() if p.suffix in ('.py','.json')},transport_tool_sha256=sha(transport),stage_timeout_seconds=120,reap_timeout_seconds=10,download_max_file_bytes=64*2**20,download_max_total_bytes=128*2**20,download_chunk_bytes=262144,
  method='One fixed4-thread IO scheduler for the original complete closure file set, with every SHA reading complete bytes and no cache. Same actual module/native/package/stdlib/source trees/checkpoint as original engine. Loaded actual pretrained float32 76M model+original head, initial/final live inventories234+4 and independent original initialization reference. No request/forward/C0/solver/training.',
  parameter_policy='Before and after identity scheduler collect both actual full model and head leaves. No parameter gate reduced or moved in any existing request; this new component is separately timed, not online8+2 qualification.',
  timing_policy='Parallel operation first after unavoidable complete source start collection; sequential reference follows. OS file caches shared and not controlled, no causal speed improvement claim. All enumeration/hash/actual mapped file reads inside each component clock. End inventories separately clocked.',
  cpu_affinity=[15],minimum_available_ram_gib=8,io_threads=4,original_34_or_three_case_QA_repeated=False,optimizer_updates=0,scientific_family3_started=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,stable_online_gain=False)
 write('EXECUTION_REGISTRATION.json',json.dumps(reg,indent=2));write('LOCAL_EXECUTION_BINDING.json',json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json')),indent=2))
 print(json.dumps(dict(root=str(O),registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),files=len(reg['package_files']))))
if __name__=='__main__':main()
