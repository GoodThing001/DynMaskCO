"""Per-expression jit; original NNX Linear objects and original mean order."""
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
