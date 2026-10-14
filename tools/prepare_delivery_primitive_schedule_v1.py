"""Prospective one-candidate, new synthetic inputs; never utility selection."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R/'research/review_3080_20261010/delivery/head_intermediate_arithmetic_root_v1'
ID=R/'research/review_3080_20261010/delivery/compiled_head_identity_root_v1';O=R/'research/review_3080_20261010/delivery/primitive_schedule_semantics_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (O/n).write_text(s,encoding='utf-8',newline='\n')

CANDIDATE='''"""Per-expression jit; original NNX Linear objects and original mean order."""
import time
import jax
import jax.numpy as jnp
def make_schedule(head):
 gather=jax.jit(lambda h,index:h[index]);mean=jax.jit(lambda weights,h:weights@h)
 concat=jax.jit(lambda a,b,c,d,e,f,g:jnp.concatenate([a,b,c,d,e,f,g],axis=-1))
 activate=jax.jit(lambda x:jnp.tanh(x))
 def score(embeddings,binding):
  rows=[];began=time.perf_counter()
  h=embeddings[0].astype(jnp.float32);actions=jnp.asarray(binding['actions'],jnp.int32)
  weights=[jnp.asarray(binding[n],jnp.float32) for n in ('route_weights','prefix_weights','suffix_weights')]
  scalars=jnp.asarray(binding['slot_position_length'],jnp.float32)
  prepared=time.perf_counter()
  def call(name,fn,*args):
   t0=time.perf_counter();x=fn(*args);t1=time.perf_counter();jax.block_until_ready(x);t2=time.perf_counter()
   rows.append(dict(name=name,dispatch_and_possible_compile_seconds=t1-t0,sync_seconds=t2-t1,start_hex=t0.hex(),dispatch_end_hex=t1.hex(),sync_end_hex=t2.hex()))
   return x
  parts=[call('gather_'+str(col),gather,h,actions[:,col]) for col in (0,3,4)]
  parts.extend(call('weighted_mean_'+str(k),mean,w,h) for k,w in enumerate(weights))
  x=call('concat',concat,*parts,scalars)
  hidden=call('original_nnx_input',head.input,x)
  activated=call('tanh',activate,hidden)
  output=call('original_nnx_output',head.output,activated)[:,0]
  ended=time.perf_counter()
  return output,dict(preparation_seconds=prepared-began,operation_rows=rows,total_before_host_seconds=ended-began,start_hex=began.hex(),prepared_hex=prepared.hex(),end_hex=ended.hex())
 return score
'''

RUN='''import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import io,json,time,traceback,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def main():
 import numpy as np,psutil,jax,jax.numpy as jnp
 from flax import nnx
 from delivery_actual_action_head_v1 import construct_head,anchor_weights
 from primitive_expression_schedule_v1 import make_schedule
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB barrier')
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'));head=construct_head();candidate=make_schedule(head)
 vals={str(k):np.asarray(v.value) for k,v in nnx.state(head,nnx.Param).flat_state()}
 identity={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in vals.items()}
 if identity!=reg['head_identity']:raise ValueError('original four leaf identity')
 np.savez(B/'HEAD_PARAMETERS.npz',**{n:vals[k] for n,k in zip(('k1','b1','k2','b2'),("('input', 'kernel')","('input', 'bias')","('output', 'kernel')","('output', 'bias')"))})
 tmp=io.BytesIO();np.savez(tmp,x=np.zeros(1,np.float32));tmp.seek(0);np.load(tmp).close();jax.devices();list(psutil.process_iter(['pid']))
 extras=[B/n for n in reg['package_files']]+[B/'EXECUTION_REGISTRATION.json']
 closure(extras,[]);start=closure(extras,[]);save(B,'SOURCE_START.json',start)
 records=[]
 for case in reg['synthetic_cases']:
  prep0=time.perf_counter();nodes=case['nodes'];remaining=list(range(3,nodes,3));before=[[i for i in range(1,nodes) if i not in remaining and i%3==v] for v in range(3)]
  actions=np.asarray([(i,v,p,route[p-1] if p else 0,route[p] if p<len(route) else 0) for i in remaining for v,route in enumerate(before) for p in range(len(route)+1)],np.int32)
  binding=anchor_weights(actions,before,remaining,nodes);latent=np.random.default_rng(case['latent_seed']).standard_normal((1,nodes,512),dtype=np.float32)
  embeddings=jnp.asarray(latent);prep1=time.perf_counter()
  t0=time.perf_counter();original=head(embeddings,binding);dispatch_end=time.perf_counter();jax.block_until_ready(original);sync_end=time.perf_counter();original_host=np.asarray(original);host_end=time.perf_counter()
  actual,timing=candidate(embeddings,binding);candidate_host0=time.perf_counter();actual_host=np.asarray(actual);candidate_host1=time.perf_counter()
  mismatch=int(np.count_nonzero(actual_host.view(np.uint32)!=original_host.view(np.uint32)))
  out={k:v for k,v in binding.items()};out.update(latent=latent,original_scores=original_host,candidate_scores=actual_host)
  name='%02d_CASE.npz'%case['index'];np.savez(B/name,**out)
  record=dict(case=case,before=before,remaining=remaining,actions_count=len(actions),artifact=name,artifact_sha256=sha(B/name),bitwise_mismatch_count=mismatch,exact_equal=mismatch==0,max_abs_error=float(np.max(np.abs(actual_host.astype(np.float64)-original_host.astype(np.float64)))),input_preparation_seconds=prep1-prep0,original=dict(dispatch_and_possible_compile_seconds=dispatch_end-t0,sync_seconds=sync_end-dispatch_end,host_seconds=host_end-sync_end,total_seconds=host_end-t0),candidate=dict(**timing,host_seconds=candidate_host1-candidate_host0),no_sample_warmup=True)
  save(B,'%02d_RECORD.json'%case['index'],record);records.append(record)
 end=closure(extras,[]);d=delta(start,end);save(B,'SOURCE_END.json',dict(**end,**d))
 result=dict(records=records,count=len(records),exact_equal_count=sum(r['exact_equal'] for r in records),synthetic_semantics_qualified=all(r['exact_equal'] for r in records),source_qualified=not d['changed'] and not d['new_files'],source_changed=d['changed'],source_new_files=d['new_files'],head_identity=identity,candidate_installed=False,original_34_QA_repeated=False,old_compiled_candidates_retried=False,backbone_executed=False,optimizer_steps=0,scientific_family3_started=False,full_request_budget_qualified=False,stable_online_gain=False)
 save(B,'RESULT.json',result);return 0 if result['synthetic_semantics_qualified'] and result['source_qualified'] else 1
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''

AUDIT='''"""Independent original-head replay on new synthetic cases; no new schedule import."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import io,json,hashlib,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def main():
 import numpy as np,psutil,jax,jax.numpy as jnp
 from flax import nnx
 native=prime_native_identity();psutil.Process().cpu_affinity([15]);reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'));result=json.loads((B/'RESULT.json').read_text(encoding='utf-8'))
 # Independently reconstruct original four-leaf NNX parameter initialization.
 rngs=nnx.Rngs(20261012);layer1=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rngs);layer2=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rngs)
 tmp=io.BytesIO();np.savez(tmp,x=np.zeros(1,np.float32));tmp.seek(0);np.load(tmp).close();jax.devices();list(psutil.process_iter(['pid']))
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','RESULT.json','HEAD_PARAMETERS.npz','SOURCE_START.json','SOURCE_END.json')]+[B/r['artifact'] for r in result['records']]
 closure(extras,[]);begin=closure(extras,[]);save(B,'AUDIT_SOURCE_START.json',begin)
 checks=[]
 def check(n,v):checks.append(dict(name=n,ok=bool(v)))
 start=json.loads((B/'SOURCE_START.json').read_text(encoding='utf-8'));end=json.loads((B/'SOURCE_END.json').read_text(encoding='utf-8'))
 for p,h in start['files'].items():check('actual_production_sha:'+p,end['files'].get(p)==h)
 check('no_unsealed_production_source',not end['changed'] and not end['new_files']);check('production_start_map_hashed',all(p in start['files'] for p in start['module_files'].values()))
 with np.load(B/'HEAD_PARAMETERS.npz',allow_pickle=False) as z:
  for n,v in [('k1',layer1.kernel.value),('b1',layer1.bias.value),('k2',layer2.kernel.value),('b2',layer2.bias.value)]:check('independent_parameter:'+n,np.array_equal(np.asarray(v),z[n]))
 for expected,record in zip(reg['synthetic_cases'],result['records']):
  check('case_registered:'+str(expected['index']),record['case']==expected);check('artifact_sha:'+str(expected['index']),sha(B/record['artifact'])==record['artifact_sha256'])
  with np.load(B/record['artifact'],allow_pickle=False) as z:q={n:z[n].copy() for n in z.files}
  nodes=expected['nodes'];remaining=list(range(3,nodes,3));before=[[i for i in range(1,nodes) if i not in remaining and i%3==v] for v in range(3)]
  actions=np.asarray([(i,v,p,route[p-1] if p else 0,route[p] if p<len(route) else 0) for i in remaining for v,route in enumerate(before) for p in range(len(route)+1)],np.int32)
  check('independent_actions:'+str(expected['index']),np.array_equal(actions,q['actions']));check('independent_latent:'+str(expected['index']),np.array_equal(np.random.default_rng(expected['latent_seed']).standard_normal((1,nodes,512),dtype=np.float32),q['latent']))
  w=[np.zeros((len(actions),nodes),np.float32) for _ in range(3)];scalars=np.empty((len(actions),3),np.float32)
  for k,(i,v,p,a,b) in enumerate(actions.tolist()):
   for arr,seq in zip(w,(before[v],before[v][:p],before[v][p:])):
    if seq:arr[k,seq]=np.float32(1./len(seq))
   scalars[k]=[v/2,p/(nodes-1),len(before[v])/(nodes-1)]
  for n,values in zip(('route_weights','prefix_weights','suffix_weights','slot_position_length'),w+[scalars]):check('independent_binding:'+n+':'+str(expected['index']),np.array_equal(values,q[n]))
  h=jnp.asarray(q['latent'][0],jnp.float32);a=jnp.asarray(actions,jnp.int32);parts=[h[a[:,0]],h[a[:,3]],h[a[:,4]]];parts.extend(jnp.asarray(arr,jnp.float32)@h for arr in w);parts.append(jnp.asarray(scalars,jnp.float32))
  out=layer2(jnp.tanh(layer1(jnp.concatenate(parts,axis=-1))))[:,0];jax.block_until_ready(out);actual=np.asarray(out)
  check('original_replay_exact:'+str(expected['index']),np.array_equal(actual,q['original_scores']));check('candidate_exact_original:'+str(expected['index']),np.array_equal(actual,q['candidate_scores']))
  mismatch=int(np.count_nonzero(q['candidate_scores'].view(np.uint32)!=q['original_scores'].view(np.uint32)));check('exact_mismatch_report:'+str(expected['index']),mismatch==record['bitwise_mismatch_count'])
  for span in record['candidate']['operation_rows']:check('span_order:'+str(expected['index'])+':'+span['name'],float.fromhex(span['start_hex'])<=float.fromhex(span['dispatch_end_hex'])<=float.fromhex(span['sync_end_hex']))
 finish=closure(extras,[]);d=delta(begin,finish);save(B,'AUDIT_SOURCE_END.json',dict(**finish,**d));check('independent_source',not d['changed'] and not d['new_files']);check('independent_begin_map_hashed',all(p in begin['files'] for p in begin['module_files'].values()))
 output=dict(check_count=len(checks),failure_count=sum(not x['ok'] for x in checks),checks=checks,synthetic_case_count=3,old_QA_repeated=False,shared_flax_JAX_reference=True,full_science_or_request_qualification=False,stable_online_gain=False)
 save(B,'INDEPENDENT_SEMANTICS_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''
def main():
 O.mkdir(parents=True,exist_ok=False)
 for n in ('action_head_runtime_v1.py','startup_native_identity_v1.py','timed_evidence_io_v1.py','delivery_actual_action_head_v1.py'):write(n,(OLD/n).read_text(encoding='utf-8'))
 write('primitive_expression_schedule_v1.py',CANDIDATE);write('run_primitive_schedule_v1.py',RUN);write('audit_primitive_schedule_v1.py',AUDIT)
 mapping={'run_intermediate_arithmetic_v1.py':'run_primitive_schedule_v1.py','audit_intermediate_arithmetic_v1.py':'audit_primitive_schedule_v1.py','supervise_intermediate_arithmetic_v1.py':'supervise_primitive_schedule_v1.py','intermediate_remote_launcher_v1.py':'primitive_remote_launcher_v1.py','intermediate_remote_status_v1.py':'primitive_remote_status_v1.py','INDEPENDENT_ARITHMETIC_AUDIT.json':'INDEPENDENT_SEMANTICS_AUDIT.json'}
 for old,new in [(k,v) for k,v in mapping.items() if k.endswith('.py') and k not in ('run_intermediate_arithmetic_v1.py','audit_intermediate_arithmetic_v1.py')]:
  src=(OLD/old).read_text(encoding='utf-8')
  for k,v in mapping.items():src=src.replace(k,v)
  write(new,src)
 transport=R/'tools/transport_delivery_primitive_schedule_v1.py';src=(R/'tools/transport_delivery_head_intermediates_v1.py').read_text(encoding='utf-8').replace('head_intermediate_arithmetic_root_v1','primitive_schedule_semantics_root_v1').replace('intermediate_remote_launcher_v1.py','primitive_remote_launcher_v1.py').replace('intermediate_remote_status_v1.py','primitive_remote_status_v1.py');transport.write_text(src,encoding='utf-8',newline='\n');ast.parse(src)
 prior=json.loads((OLD/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_ONE_PER_EXPRESSION_SCHEDULE_THREE_SYNTHETIC_CASES',execution_authorized=True,independent_timed_request_checker_sealed=True,head_identity=prior['head_identity'],prior_registration_sha256=sha(OLD/'EXECUTION_REGISTRATION.json'),
  synthetic_cases=[dict(index=i,nodes=n,latent_seed=2026101301+i) for i,n in enumerate((17,33,51))],
  method='One fixed per-expression jit for gather, weighted matrix mean, concat and tanh. Original NNX input/output Linear objects and source arithmetic retained. Each expression returns and explicitly synchronizes. No full-head jit. Strict bitwise original score gate on all three new fixed synthetic engineering inputs, no retries or variants.',
  warmup='Only interpreter/native/imports/parameter initialization, jax devices and tiny archive interface initialized before start. No sample or head-output warmup. One original and one candidate cold operation per case. No performance improvement inference from asymmetric caches/extra sync.',
  cpu_affinity=[15],minimum_available_ram_gib=8,stage_timeout_seconds=120,reap_timeout_seconds=10,stage_execution=['run_primitive_schedule_v1.py','audit_primitive_schedule_v1.py'],package_files={p.name:sha(p) for p in O.iterdir() if p.suffix in ('.py','.json')},transport_tool_sha256=sha(transport),local_transport_helpers=prior['local_transport_helpers'],download_max_file_bytes=64*2**20,download_max_total_bytes=128*2**20,download_chunk_bytes=262144,
  independent_scope='Independently reconstruct original seeded NNX Linear layers and synthetic latent/actions/current route weights; exact all scores/parameters/source/clock identities. No schedule/head/anchor helper imported by audit; shared original JAX/Flax implementation, not alternative model arithmetic.',
  family2_closed=True,scientific_family3_started=False,optimizer_steps=0,original_34_QA_repeated=False,candidate_installed=False,formal_controls_qualified=False,full_scientific_loaded_closure_qualified=False,full_request_budget_qualified=False,stable_online_gain=False)
 write('EXECUTION_REGISTRATION.json',json.dumps(reg,indent=2));write('LOCAL_EXECUTION_BINDING.json',json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json')),indent=2))
 print(json.dumps(dict(root=str(O),registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),files=len(reg['package_files']))))
if __name__=='__main__':main()
