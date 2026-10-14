"""Only saved head inputs; new intermediates, zero backbone/solver/training."""
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
