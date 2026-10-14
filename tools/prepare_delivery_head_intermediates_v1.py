"""Seal one fixed new arithmetic observable, never a model/utility search."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OLD=R/'research/review_3080_20261010/delivery/compiled_head_identity_root_v1'
O=R/'research/review_3080_20261010/delivery/head_intermediate_arithmetic_root_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
    ast.parse(s) if n.endswith('.py') else None
    (O/n).write_text(s,encoding='utf-8',newline='\n')

RUN='''"""Only saved head inputs; new intermediates, zero backbone/solver/training."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import io,json,time,traceback,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity

def main():
 import numpy as np,psutil,jax,jax.numpy as jnp
 from flax import nnx
 from delivery_actual_action_head_v1 import construct_head
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB resource barrier')
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 entry=json.loads((B/'FIXED_INPUT.json').read_text(encoding='utf-8'))
 p=Path(entry['path']);old_compiled=Path(reg['old_compiled_output']['path'])
 if sha(p)!=entry['sha256'] or sha(old_compiled)!=reg['old_compiled_output']['sha256']:raise ValueError('sealed input SHA')
 with np.load(p,allow_pickle=False) as z:q={n:z[n].copy() for n in ('latent','actions','route_weights','prefix_weights','suffix_weights','slot_position_length','scores')}
 with np.load(old_compiled,allow_pickle=False) as z:old_c=z['scores'].copy()
 head=construct_head();state=nnx.state(head,nnx.Param)
 vals={str(k):np.asarray(v.value) for k,v in state.flat_state()}
 identity={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in vals.items()}
 if identity!=reg['head_identity']:raise ValueError('four parameter identities')
 k1,b1,k2,b2=[jnp.asarray(vals[k]) for k in ("('input', 'kernel')","('input', 'bias')","('output', 'kernel')","('output', 'bias')")]
 np.savez(B/'HEAD_PARAMETERS.npz',k1=k1,b1=b1,k2=k2,b2=b2)
 h=jnp.asarray(q['latent'][0],jnp.float32);a=jnp.asarray(q['actions'],jnp.int32)
 weights=[jnp.asarray(q[n],jnp.float32) for n in ('route_weights','prefix_weights','suffix_weights')];scalars=jnp.asarray(q['slot_position_length'],jnp.float32)
 names=['customer','predecessor','successor','route_mean','prefix_mean','suffix_mean','concat','linear1_dot','linear1_bias','tanh','linear2_dot','output']
 barrier=jax.lax.optimization_barrier
 def instrumented(h,a,r,p,s,scalars,k1,b1,k2,b2):
  parts=[barrier(h[a[:,0]]),barrier(h[a[:,3]]),barrier(h[a[:,4]]),barrier(r@h),barrier(p@h),barrier(s@h)]
  x=barrier(jnp.concatenate(parts+[scalars],axis=-1));d1=barrier(x@k1);v1=barrier(d1+b1);t=barrier(jnp.tanh(v1));d2=barrier(t@k2);out=barrier(d2+b2)[:,0]
  return tuple(parts+[x,d1,v1,t,d2,out])
 compiled=jax.jit(instrumented)
 jax.devices();list(psutil.process_iter(['pid']));tmp=io.BytesIO();np.savez(tmp,x=np.zeros(1,np.float32));tmp.seek(0);np.load(tmp).close()
 extras=[B/n for n in reg['package_files']]+[B/'EXECUTION_REGISTRATION.json',p,old_compiled]
 # Import/enumeration preinitialization happens BEFORE the actual start hash.
 closure(extras,[])
 start=closure(extras,[]);save(B,'SOURCE_START.json',start)
 clock=time.perf_counter()
 parts=[h[a[:,0]],h[a[:,3]],h[a[:,4]]]+[w@h for w in weights]
 x=jnp.concatenate(parts+[scalars],axis=-1);d1=x@k1;v1=head.input(x);t=jnp.tanh(v1);d2=t@k2;out=head.output(t)[:,0]
 eager=tuple(parts+[x,d1,v1,t,d2,out]);jax.block_until_ready(eager);eager_arrays=[np.asarray(v) for v in eager]
 eager_end=time.perf_counter()
 observed=compiled(h,a,*weights,scalars,k1,b1,k2,b2);jax.block_until_ready(observed);compiled_arrays=[np.asarray(v) for v in observed]
 compiled_end=time.perf_counter()
 np.savez(B/'INTERMEDIATES.npz',**dict([('eager_'+n,v) for n,v in zip(names,eager_arrays)]+[('compiled_'+n,v) for n,v in zip(names,compiled_arrays)]+[('saved_original',q['scores']),('saved_old_compiled',old_c)]))
 rows=[]
 for n,e,c in zip(names,eager_arrays,compiled_arrays):
  neq=int(np.count_nonzero(e.view(np.uint32)!=c.view(np.uint32)))
  rows.append(dict(name=n,shape=list(e.shape),dtype=str(e.dtype),elements=e.size,bitwise_mismatch_count=neq,exact_equal=neq==0,max_abs_error=float(np.max(np.abs(e.astype(np.float64)-c.astype(np.float64))))))
 end=closure(extras,[]);d=delta(start,end);save(B,'SOURCE_END.json',dict(**end,**d))
 result=dict(intermediate_rows=rows,first_divergent_intermediate=next((v['name'] for v in rows if not v['exact_equal']),None),eager_score_exact_saved_original=np.array_equal(eager_arrays[-1],q['scores']),instrumented_score_exact_saved_old_compiled=np.array_equal(compiled_arrays[-1],old_c),source_qualified=not d['changed'] and not d['new_files'],source_new_files=d['new_files'],module_maps_equal=start['module_files']==end['module_files'],eager_seconds=eager_end-clock,compiled_instrumented_seconds=compiled_end-eager_end,operation_count=1,fixed_arm=entry['arm'],fixed_step=entry['step'],full_backbone_executed=False,old_34_QA_repeated=False,optimizer_steps=0,scientific_family3_started=False,stable_online_gain=False,qualification='Arithmetic observation only; returned intermediates may change compiler schedule. First divergence does not uniquely identify an XLA pass. Old strict0/34FAIL retained; no candidate installation or threshold relaxation.')
 save(B,'RESULT.json',result)
 return 0 if result['source_qualified'] and result['eager_score_exact_saved_original'] else 1
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''

AUDIT='''"""Independent saved-byte arithmetic diagnosis; no JAX/head evaluation."""
import json,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def main():
 import numpy as np,psutil
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','INTERMEDIATES.npz','HEAD_PARAMETERS.npz','RESULT.json','SOURCE_START.json','SOURCE_END.json')]
 # Preinitialize every runtime dependency before collecting beginning identity.
 closure(extras,[]);begin=closure(extras,[]);save(B,'AUDIT_SOURCE_START.json',begin)
 checks=[]
 def check(n,ok,detail=None):checks.append(dict(name=n,ok=bool(ok),detail=detail))
 result=json.loads((B/'RESULT.json').read_text(encoding='utf-8'));source=json.loads((B/'SOURCE_START.json').read_text(encoding='utf-8'));end=json.loads((B/'SOURCE_END.json').read_text(encoding='utf-8'))
 for p,h in source['files'].items():check('production_source:'+p,end['files'].get(p)==h)
 check('production_no_unsealed_loaded_files',not end['new_files']);check('production_start_map_all_hashed',all(p in source['files'] for p in source['module_files'].values()))
 entry=json.loads((B/'FIXED_INPUT.json').read_text(encoding='utf-8'));check('fixed_input_sha',sha(entry['path'])==entry['sha256'])
 with np.load(entry['path'],allow_pickle=False) as z:q={n:z[n].copy() for n in ('latent','actions','route_weights','prefix_weights','suffix_weights','slot_position_length','scores')}
 with np.load(B/'INTERMEDIATES.npz',allow_pickle=False) as z:parts={n:z[n].copy() for n in z.files}
 with np.load(B/'HEAD_PARAMETERS.npz',allow_pickle=False) as z:params={n:z[n].copy() for n in z.files}
 h=q['latent'][0];a=q['actions']
 for n,col in [('customer',0),('predecessor',3),('successor',4)]:check('gather_identity:'+n,np.array_equal(parts['eager_'+n],h[a[:,col]]) and np.array_equal(parts['compiled_'+n],h[a[:,col]]))
 for n in ('eager','compiled'):
  expected=np.concatenate([parts[n+'_'+k] for k in ('customer','predecessor','successor','route_mean','prefix_mean','suffix_mean')]+[q['slot_position_length']],axis=-1)
  check('concat_identity:'+n,np.array_equal(expected,parts[n+'_concat']))
 for row in result['intermediate_rows']:
  e=parts['eager_'+row['name']];c=parts['compiled_'+row['name']];neq=int(np.count_nonzero(e.view(np.uint32)!=c.view(np.uint32)))
  check('reported_exact:'+row['name'],neq==row['bitwise_mismatch_count'] and (neq==0)==row['exact_equal'])
  check('reported_error:'+row['name'],float(np.max(np.abs(e.astype(np.float64)-c.astype(np.float64))))==row['max_abs_error'])
 check('original_final_binding',np.array_equal(parts['eager_output'],q['scores'])==result['eager_score_exact_saved_original'])
 check('compiled_instrumentation_binding',np.array_equal(parts['compiled_output'],parts['saved_old_compiled'])==result['instrumented_score_exact_saved_old_compiled'])
 check('first_divergence_order',next((r['name'] for r in result['intermediate_rows'] if not r['exact_equal']),None)==result['first_divergent_intermediate'])
 # BLAS differences are descriptive diagnostics only, never exact eligibility.
 diagnostics={}
 for side in ('eager','compiled'):
  for output,expected in [('linear1_dot',parts[side+'_concat']@params['k1']),('linear1_bias',parts[side+'_linear1_dot']+params['b1']),('tanh',np.tanh(parts[side+'_linear1_bias'])),('linear2_dot',parts[side+'_tanh']@params['k2']),('output',(parts[side+'_linear2_dot']+params['b2'])[:,0])]:
   diagnostics[side+'_'+output]=float(np.max(np.abs(expected.astype(np.float64)-parts[side+'_'+output].astype(np.float64))))
 finish=closure(extras,[]);d=delta(begin,finish);save(B,'AUDIT_SOURCE_END.json',dict(**finish,**d))
 check('independent_loaded_closure',not d['changed'] and not d['new_files']);check('independent_start_map_hashed',all(p in begin['files'] for p in begin['module_files'].values()))
 output=dict(check_count=len(checks),failure_count=sum(not x['ok'] for x in checks),checks=checks,numpy_operator_diagnostics=diagnostics,first_divergent_intermediate=result['first_divergent_intermediate'],old_strict34_result_unchanged=True,new_gate_relaxed=False,old_model_QA_repeated=False,full_head_science_qualified=False,stable_online_gain=False)
 save(B,'INDEPENDENT_ARITHMETIC_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''

def main():
 O.mkdir(parents=True,exist_ok=False)
 for n in ('action_head_runtime_v1.py','startup_native_identity_v1.py','timed_evidence_io_v1.py','delivery_actual_action_head_v1.py'):write(n,(OLD/n).read_text(encoding='utf-8'))
 write('run_intermediate_arithmetic_v1.py',RUN);write('audit_intermediate_arithmetic_v1.py',AUDIT)
 mapping={'run_compiled_head_v1.py':'run_intermediate_arithmetic_v1.py','audit_compiled_head_v1.py':'audit_intermediate_arithmetic_v1.py','supervise_compiled_head_v1.py':'supervise_intermediate_arithmetic_v1.py','compiled_head_remote_launcher_v1.py':'intermediate_remote_launcher_v1.py','compiled_head_remote_status_v1.py':'intermediate_remote_status_v1.py','INDEPENDENT_TIMED_REQUEST_AUDIT.json':'INDEPENDENT_ARITHMETIC_AUDIT.json','TIMED_AUDIT_FAILURE.json':'AUDIT_FAILURE.json'}
 for old,new in [(k,v) for k,v in mapping.items() if k.endswith('.py') and k not in ('run_compiled_head_v1.py','audit_compiled_head_v1.py')]:
  source=(OLD/old).read_text(encoding='utf-8')
  for k,v in mapping.items():source=source.replace(k,v)
  write(new,source)
 entry=json.loads((OLD/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))[0]
 write('FIXED_INPUT.json',json.dumps(entry,indent=2))
 oldreg=json.loads((OLD/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 manifest=json.loads((OLD/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'));oldout=next(x for x in manifest if x['name']=='00_OUTPUT.npz')
 remote_old=r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\compiled_head_identity_root_v1\00_OUTPUT.npz'
 transport=R/'tools/transport_delivery_head_intermediates_v1.py'
 s=(R/'tools/transport_delivery_compiled_head_v1.py').read_text(encoding='utf-8').replace('compiled_head_identity_root_v1','head_intermediate_arithmetic_root_v1').replace('compiled_head_remote_launcher_v1.py','intermediate_remote_launcher_v1.py').replace('compiled_head_remote_status_v1.py','intermediate_remote_status_v1.py')
 transport.write_text(s,encoding='utf-8',newline='\n');ast.parse(s)
 reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_FIXED_ONE_INPUT_NEW_INTERMEDIATE_DIAGNOSIS',execution_authorized=True,independent_timed_request_checker_sealed=True,
  fixed_input=entry,old_compiled_output=dict(path=remote_old,sha256=oldout['sha256']),head_identity=oldreg['head_identity'],original_registration_sha256=sha(OLD/'EXECUTION_REGISTRATION.json'),
  cpu_affinity=[15],minimum_available_ram_gib=8,stage_timeout_seconds=120,reap_timeout_seconds=10,stage_execution=['run_intermediate_arithmetic_v1.py','audit_intermediate_arithmetic_v1.py'],
  package_files={p.name:sha(p) for p in O.iterdir() if p.suffix in ('.py','.json')},transport_tool_sha256=sha(transport),local_transport_helpers=oldreg['local_transport_helpers'],download_max_file_bytes=64*2**20,download_max_total_bytes=128*2**20,download_chunk_bytes=262144,
  observation='One eager arithmetic-intermediate capture and one barrier-jit tuple capture on fixed first saved initial input. No backbone or old34 candidate/output qualification rerun. Compare to old saved final values to expose instrumentation changes, no compiler-pass causal proof.',
  source_preinitialization='Initialize all closure imports and native identity BEFORE actual start; preserve originals, no repair of missing old starting SHA.',
  strict_previous_gate_unchanged=True,old_34_QA_not_repeated=True,optimizer_steps=0,scientific_family3_started=False,learned_output_qualified=False,full_request_budget_qualified=False,stable_online_gain=False)
 write('EXECUTION_REGISTRATION.json',json.dumps(reg,indent=2));write('LOCAL_EXECUTION_BINDING.json',json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json')),indent=2))
 print(json.dumps(dict(root=str(O),registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),source_files=len(reg['package_files']))))
if __name__=='__main__':main()
