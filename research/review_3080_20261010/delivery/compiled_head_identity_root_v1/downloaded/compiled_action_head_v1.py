"""Single compiled expression schedule; no warmup or model/control claims."""
import jax
import jax.numpy as jnp

def make_scorer():
    # Barriers explicitly retain intermediate arithmetic boundaries; exact
    # agreement is an observed acceptance gate, not assumed from this source.
    barrier=jax.lax.optimization_barrier
    def score(h,actions,route,prefix,suffix,scalars,k1,b1,k2,b2):
        parts=[barrier(h[actions[:,0]]),barrier(h[actions[:,3]]),barrier(h[actions[:,4]])]
        parts.extend([barrier(route@h),barrier(prefix@h),barrier(suffix@h)])
        parts.append(scalars)
        x=barrier(jnp.concatenate(parts,axis=-1))
        hidden=barrier(barrier(x@k1)+b1)
        activated=barrier(jnp.tanh(hidden))
        return barrier(barrier(activated@k2)+b2)[:,0]
    return jax.jit(score)
