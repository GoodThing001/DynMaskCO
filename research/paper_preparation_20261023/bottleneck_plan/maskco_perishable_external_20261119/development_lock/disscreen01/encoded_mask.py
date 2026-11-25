"""Evaluate the frozen MASKCO decoder on a new partial route with an existing mask encoding."""
import time
import numpy as np

def covering_masks(seed):
    rng=np.random.default_rng(seed)
    buckets=rng.permutation(np.arange(1,101,dtype=np.int32)).reshape(10,10)
    masks=[np.concatenate([buckets[(i+j)%10] for j in (0,1,3,6)]) for i in range(10)]
    return [masks[i] for i in rng.permutation(10)]

def with_split_inference(model):
    import jax
    from flax import nnx
    from profit_neural import CompiledPredictor
    if model.config["variant"]!="maskco_base":raise ValueError("Encoder reuse requires the fixed base input schema")
    graph,_=nnx.split(model)
    predictor=CompiledPredictor(model)
    @jax.jit
    def encode(state,features):
        return nnx.merge(graph,state).backbone.encode(features,attn_options={})
    @jax.jit
    def decode(state,encoded,timestep,adjacency):
        backbone=nnx.merge(graph,state).backbone
        current=backbone.decode(encoded,timestep,adjacency,attn_options={},target="embed")
        return backbone.feature2logit(current)
    predictor.mask_encode=encode
    predictor.route_decode=decode
    return predictor

def split_batch(policy,batch,key,cache):
    import jax.numpy as jnp
    from experiment_scope import require_operation
    require_operation("infer")
    if key not in cache:
        cache[key]=policy.mask_encode(policy.state,jnp.asarray(batch["features"]))
    result=policy.route_decode(policy.state,cache[key],jnp.asarray(batch["timestep"]),jnp.asarray(batch["adjacency"]))
    logits=np.asarray(result.block_until_ready())[0]
    if not np.all(np.isfinite(logits)):raise ValueError("Non-finite conditional prediction")
    return logits

def log_heat(logits):
    shifted=logits-logits.max(axis=1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))

class CachedRepairProvider:
    def __init__(self,policy):
        self.policy=policy
        self.inputs=None
        self.cache={}
        self.forward_calls=0
        self.forward_seconds=0.

    def heatmap(self,task,routes,removed,variant,step):
        from base_inputs import BaseInferenceInputs
        start=time.perf_counter()
        if self.inputs is None:
            self.inputs=BaseInferenceInputs(task)
        key=tuple(map(int,removed))
        batch=self.inputs.build(routes,removed)
        result=log_heat(split_batch(self.policy,batch,key,self.cache))
        self.forward_calls+=1
        self.forward_seconds+=time.perf_counter()-start
        return result

def load_split(study,access,core,source,locked,case):
    from unittest.mock import patch
    import profit_neural
    from base_inputs import BaseInferenceInputs
    original=profit_neural.CompiledPredictor
    def factory(model):
        with patch.object(profit_neural,"CompiledPredictor",original):
            return with_split_inference(model)
    with patch.object(profit_neural,"CompiledPredictor",factory):
        policy,scope=study.load("SLOT",access,core,source,locked,case)
    warm=access.Scope("encoder-decoder-warmup",frozenset([(0,"train","r101")]),frozenset(["initialize","infer","warmup"]))
    with access.restricted_scope(warm):
        task=core.ProfitTask("r101",fold=0,role="train")
        batch=BaseInferenceInputs(task).build(task.initial(),covering_masks(0)[0])
        split_batch(policy,batch,0,{})
    return policy,scope

def check_mask_family():
    for seed in (1000,1001,1002):
        groups=covering_masks(seed)
        counts=np.zeros((101,101),dtype=int)
        for nodes in groups:
            if len(set(nodes))!=40:raise ValueError("Invalid hidden-customer group")
            counts[np.ix_(nodes,nodes)]+=1
        if not np.all(np.diag(counts)[1:]==4) or counts[1:,1:].min()<1:raise ValueError("Incomplete customer or pair coverage")
if __name__=="__main__":check_mask_family()
