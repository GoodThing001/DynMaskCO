"""Independent original-head replay on new synthetic cases; no new schedule import."""
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
