"""Real upstream MaskCO CVRP backbone with two explicit ablatable additions."""
from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from profit_core import CONFIG, FREEZE, SCHEMA, VARIANTS, digest
from models.CVRPModel import CVRPModel

PROFILES = {
    'maskco_256': dict(num_layers=(16,6),dimension=256,heads=8),
    'smoke_cpu': dict(num_layers=(1,1),dimension=32,heads=4),
}


def portable_attention(q=None,k=None,v=None,qkv=None,bias=None,causal=False,sm_scale=None,softcap=None,**unused):
    """Instance-local JAX attention preserving softcap then adjacency bias."""
    if qkv is not None:
        q,k,v = (qkv[:,:,i] for i in range(3))
    dtype = q.dtype
    q,k,v = (a.astype(jnp.float32) for a in (q,k,v))
    logits = jnp.einsum('bqhd,bkhd->bhqk',q,k)*(sm_scale if sm_scale is not None else q.shape[-1]**-0.5)
    if softcap is not None:
        logits = softcap*jnp.tanh(logits/softcap)
    if bias is not None:
        logits += bias[:,None] if bias.ndim == 3 else bias
    if causal:
        logits = jnp.where(jnp.tril(jnp.ones(logits.shape[-2:],bool)),logits,-jnp.inf)
    return jnp.einsum('bhqk,bkhd->bqhd',jax.nn.softmax(logits,axis=-1),v).astype(dtype)


class ProfitMaskCO(nnx.Module):
    def __init__(self,variant='quality_directed',profile='maskco_256',seed=1000):
        if variant not in VARIANTS or profile not in PROFILES:
            raise ValueError('unknown method variant or backbone profile')
        dims = PROFILES[profile]
        d,h = dims['dimension'],dims['heads']
        self.config = dict(variant=variant,profile=profile,seed=int(seed))
        self.directed = VARIANTS[variant][1]
        rngs = nnx.Rngs(seed)
        self.backbone = CVRPModel(num_layers=dims['num_layers'],embed_dim=(d,d),num_heads=(h,h),
            encoder_input_dim=16,dtype=jnp.float32,softcap=30.0,final_norm=True,
            final_norm_use_scale=True,final_softcap=None,final_logit_scale=16/d,rngs=rngs)
        for layer in self.backbone.encoder.layers+self.backbone.decoder.layers:
            layer.attn.attention_fn = portable_attention
        if self.directed:
            self.query = nnx.Linear(d,d,use_bias=False,rngs=rngs)
            self.key = nnx.Linear(d,d,use_bias=False,rngs=rngs)

    def __call__(self,features,adjacency,timestep):
        h = self.backbone.encode(features,attn_options={})
        h = self.backbone.decode(h,timestep,adjacency,attn_options={},target='embed')
        if self.directed:
            h = self.backbone.final_norm(h)
            logits = jnp.matmul(self.query(h),self.key(h).transpose(0,2,1))*self.backbone.final_logit_scale
            return jnp.where(jnp.eye(logits.shape[-1],dtype=bool)[None],-1e9,logits)
        return self.backbone.feature2logit(h)


def parameter_digest(model):
    import hashlib
    value = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(nnx.state(model,nnx.Param)):
        a = np.asarray(leaf)
        value.update(str((a.shape,a.dtype)).encode())
        value.update(a.tobytes())
    return value.hexdigest()


def parameter_count(model):
    return sum(np.size(a) for a in jax.tree_util.tree_leaves(nnx.state(model,nnx.Param)))


class Trainer:
    def __init__(self,model,learning_rate=1e-4,weight_decay=1e-4):
        self.model = model
        self.learning_rate,self.weight_decay = learning_rate,weight_decay
        self.tx = optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(learning_rate,weight_decay=weight_decay))
        self.graph,self.state = nnx.split(model)
        self.opt_state = self.tx.init(self.state)
        self.steps = 0
        graph,tx = self.graph,self.tx
        @jax.jit
        def update(state,opt_state,batch):
            def loss(params):
                current = nnx.merge(graph,params)
                logits = current(batch['features'],batch['adjacency'],batch['timestep'])
                ce = -(batch['targets']*jax.nn.log_softmax(logits,axis=-1)).sum(axis=-1)
                return (ce*batch['row_weights']).sum()/batch['row_weights'].sum()
            loss_value,grads = jax.value_and_grad(loss)(state)
            updates,opt_state = tx.update(grads,opt_state,state)
            return optax.apply_updates(state,updates),opt_state,loss_value,optax.global_norm(grads)
        self._step = update

    def step(self,batch):
        old = [np.asarray(a).copy() for a in jax.tree_util.tree_leaves(self.state)]
        self.state,self.opt_state,loss,gradient = self._step(self.state,self.opt_state,
            {k:jnp.asarray(v) for k,v in batch.items()})
        loss,gradient = float(loss),float(gradient)
        if not math.isfinite(loss) or not math.isfinite(gradient):
            raise ValueError('non-finite loss/gradient')
        new = [np.asarray(a) for a in jax.tree_util.tree_leaves(self.state)]
        if not all(np.all(np.isfinite(a)) for a in new):
            raise ValueError('non-finite trained parameter')
        changed = sum(not np.array_equal(a,b) for a,b in zip(old,new))
        nnx.update(self.model,self.state)
        self.steps += 1
        return dict(step=self.steps,loss=loss,gradient_norm=gradient,updated_parameter_leaves=changed)


def predict(model,batch):
    result = np.asarray(model(*(jnp.asarray(batch[k]) for k in ('features','adjacency','timestep'))))[0]
    if not np.all(np.isfinite(result)):
        raise ValueError('non-finite heatmap')
    return result


def save_tree(path,tree):
    np.savez(path,**{'leaf%05d'%i:np.asarray(a) for i,a in enumerate(jax.tree_util.tree_leaves(tree))})


def load_tree(path,template):
    leaves,structure = jax.tree_util.tree_flatten(template)
    with np.load(path,allow_pickle=False) as data:
        if len(data.files) != len(leaves):
            raise ValueError('checkpoint leaf count mismatch')
        arrays = []
        for i,reference in enumerate(leaves):
            value = data['leaf%05d'%i]
            if value.shape != np.shape(reference) or value.dtype != np.asarray(reference).dtype or not np.all(np.isfinite(value)):
                raise ValueError('checkpoint shape/dtype/finite check failed')
            arrays.append(jnp.asarray(value))
    return structure.unflatten(arrays)


def save_checkpoint(path,trainer,metadata):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    save_tree(path/'weights.npz',trainer.state)
    save_tree(path/'optimizer.npz',trainer.opt_state)
    record = dict(model=trainer.model.config,feature_schema=SCHEMA,steps=trainer.steps,
        model_sha256=digest(CONFIG),research_lock_sha256=digest(FREEZE),
        learning_rate=trainer.learning_rate,weight_decay=trainer.weight_decay,metadata=metadata,
        files={name:digest(path/name) for name in ('weights.npz','optimizer.npz')})
    (path/'checkpoint.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


def load_checkpoint(path):
    path = Path(path)
    record = json.loads((path/'checkpoint.json').read_text(encoding='utf-8'))
    if record['feature_schema'] != SCHEMA or record['model_sha256'] != digest(CONFIG) or record['research_lock_sha256'] != digest(FREEZE):
        raise ValueError('checkpoint scientific identity mismatch')
    if any(digest(path/name) != value for name,value in record['files'].items()):
        raise ValueError('checkpoint bytes changed')
    if record['metadata']['task']['role'] != 'train':
        raise ValueError('checkpoint is not from an allowed training role')
    trainer = Trainer(ProfitMaskCO(**record['model']),record['learning_rate'],record['weight_decay'])
    trainer.state = load_tree(path/'weights.npz',trainer.state)
    trainer.opt_state = load_tree(path/'optimizer.npz',trainer.opt_state)
    trainer.steps = record['steps']
    nnx.update(trainer.model,trainer.state)
    return trainer,record
