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
