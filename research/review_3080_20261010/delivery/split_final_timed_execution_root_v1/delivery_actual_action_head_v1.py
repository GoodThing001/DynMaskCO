"""Prospective action head over actual MaskCO decoder embeddings.

Unexecuted engineering source. No teacher input, optimizer, solver or C0 gate.
Upstream MaskCO remains unchanged; model/control qualification is external.
"""
import numpy as np

def anchor_weights(actions, before, remaining, nodes):
    """Host-side identity validation and means of current route/prefix/suffix."""
    if not isinstance(actions,np.ndarray) or actions.dtype!=np.int32 or actions.ndim!=2 or actions.shape[1]!=5:
        raise ValueError('exact int32 action identity')
    if nodes<2 or not before or not len(actions):raise ValueError('empty domain')
    flat=[j for route in before for j in route]
    if len(set(flat))!=len(flat) or len(set(remaining))!=len(remaining) or set(flat)&set(remaining) or set(flat)|set(remaining)!=set(range(1,nodes)):
        raise ValueError('current skeleton/remaining exact coverage')
    if len({tuple(r) for r in actions.tolist()})!=len(actions):raise ValueError('duplicate action')
    route_w=np.zeros((len(actions),nodes),np.float32);prefix_w=route_w.copy();suffix_w=route_w.copy();scalars=np.zeros((len(actions),3),np.float32)
    for k,(i,v,p,a,b) in enumerate(actions.tolist()):
        if i not in remaining or not 0<=v<len(before) or not 0<=p<=len(before[v]):raise ValueError('action domain')
        route=before[v]
        if a!=(route[p-1] if p else 0) or b!=(route[p] if p<len(route) else 0):raise ValueError('stale temporary anchor')
        for w,part in ((route_w,route),(prefix_w,route[:p]),(suffix_w,route[p:])):
            if part:w[k,part]=np.float32(1./len(part))
        # Explicit current slot and insertion position, never teacher slot/rank.
        scalars[k]=[v/max(1,len(before)-1),p/(nodes-1),len(route)/(nodes-1)]
    context=np.zeros((1,nodes,nodes),np.float32)
    for route in before:
        path=[0]+route+[0]
        for a,b in zip(path,path[1:]):
            if a!=b:context[0,a,b]=context[0,b,a]=1.
    timestep=np.asarray([context[0,1:].sum()/(2*(nodes-1))],np.float32)
    return dict(actions=actions.copy(),route_weights=route_w,prefix_weights=prefix_w,suffix_weights=suffix_w,slot_position_length=scalars,context=context,timestep=timestep)

def construct_head(embedding_dim=512,hidden_dim=64,seed=20261012):
    """Create learnable NNX head; caller must preregister execution and resources."""
    from flax import nnx
    import jax.numpy as jnp
    class ActionHead(nnx.Module):
        def __init__(self):
            rngs=nnx.Rngs(seed)
            self.input=nnx.Linear(6*embedding_dim+3,hidden_dim,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rngs)
            self.output=nnx.Linear(hidden_dim,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rngs)
        def __call__(self,embeddings,binding):
            if embeddings.shape!=(1,binding['context'].shape[-1],embedding_dim):raise ValueError('actual decoder embedding shape')
            h=embeddings[0].astype(jnp.float32);actions=jnp.asarray(binding['actions'],jnp.int32)
            parts=[h[actions[:,0]],h[actions[:,3]],h[actions[:,4]]]
            parts.extend(jnp.asarray(binding[k],jnp.float32)@h for k in ('route_weights','prefix_weights','suffix_weights'))
            parts.append(jnp.asarray(binding['slot_position_length'],jnp.float32))
            x=jnp.concatenate(parts,axis=-1)
            return self.output(jnp.tanh(self.input(x)))[:,0]
    return ActionHead()

def forward(model,head,features,binding):
    """Actual encode/decode(target=embed), final norm, then ordered action head."""
    import jax.numpy as jnp
    if features.shape!=(1,binding['context'].shape[-1],10):raise ValueError('full explicit 10D snapshot features required')
    encoded=model.encode(jnp.asarray(features,jnp.float32),attn_options={})
    embedded=model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),attn_options={},target='embed')
    if model.apply_final_norm:embedded=model.final_norm(embedded)
    return head(embedded,binding)
