"""Extract the checked MaskCO representation before its symmetric output product."""
from pathlib import Path
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def attach_embeddings(study,access,core,source,locked):
    import jax
    from flax import nnx
    from unittest.mock import patch
    import profit_neural
    original=profit_neural.CompiledPredictor
    def bind(model):
        graph,_=nnx.split(model)
        policy=original(model)
        @jax.jit
        def forward(state,features,adjacency,timestep):
            backbone=nnx.merge(graph,state).backbone
            h=backbone.encode(features,attn_options={})
            h=backbone.decode(h,timestep,adjacency,attn_options={},target="embed")
            return backbone.final_norm(h)
        policy.masked_embedding=forward
        policy.edge_scale=float(model.backbone.final_logit_scale)
        return policy
    with patch.object(profit_neural,"CompiledPredictor",bind):
        return study.load("SLOT",access,core,source,locked,"c205")[0]


def embedding(policy,batch):
    import jax.numpy as jnp
    from experiment_scope import require_operation
    require_operation("infer")
    result=policy.masked_embedding(policy.state,*(jnp.asarray(batch[k]) for k in ("features","adjacency","timestep")))
    return np.asarray(result.block_until_ready())[0]


def physical(nodes,batch):
    nodes=np.asarray(nodes,dtype=float)
    delta=nodes[:,1:3]-nodes[0,1:3]
    radius=np.linalg.norm(delta,axis=1)
    scale=max(1.,np.quantile(radius,.9))
    adjacency=np.asarray(batch["adjacency"][0])
    degree=adjacency.sum(axis=1);mass=nodes[:,3]/20000.
    return np.column_stack((delta/scale,radius/scale,mass,nodes[:,4:7]/1440.,
                            batch["features"][0,:,10],degree/2.,adjacency@mass/np.maximum(1.,degree),
                            np.arange(len(nodes))==0)).astype(np.float32)


def normalized_logits(h):
    h=np.array(h,copy=True)
    np.fill_diagonal(h,-1.e9);h-=h.max(axis=1,keepdims=True)
    return h-np.log(np.exp(h).sum(axis=1,keepdims=True))


def main(index):
    import external_common as harness
    import pool_search as pool
    cfg=json.loads((HERE/"phase.json").read_text());case=cfg["blocks"][index]["case"]
    archives=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=harness.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from timed_search import single_cpu_guard
    from base_inputs import BaseInferenceInputs
    import profit_neural
    single_cpu_guard();policy=attach_embeddings(pool.study,access,core,source,locked)
    scope=access.Scope("directional-masked-decode",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    rng=np.random.default_rng(4117601);ms=[];rep=[];ph=[];err=0.;start=time.process_time()
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");builder=BaseInferenceInputs(task)
        for name in ("initial","terminal"):
            routes=tuple(map(tuple,archives[name]))
            if name=="initial" and routes!=tuple(map(tuple,task.initial())):raise ValueError("Different common start")
            task.evaluate(routes,audit=True)
            if "regions" in cfg["blocks"][index]:
                masks=[np.array(row["mask"],int) for row in cfg["blocks"][index]["regions"] if row["state"]==name]
            else:
                masks=[rng.choice(np.arange(1,101),40,replace=False) for _ in range(cfg["masks_per_state"])]
            for j,mask in enumerate(masks):
                if len(set(mask.tolist()))!=40 or mask.min()<1 or mask.max()>100:raise ValueError("Invalid customer mask")
                batch=builder.build(routes,mask);h=embedding(policy,batch)
                if not np.isfinite(h).all() or h.shape!=(101,256):raise ValueError("Invalid pretrained representation")
                if j==0:
                    expected=normalized_logits(profit_neural.predict(policy,batch))
                    actual=normalized_logits((h@h.T)*policy.edge_scale)
                    off=~np.eye(len(h),dtype=bool)
                    err=max(err,float(np.abs(actual[off]-expected[off]).max()))
                ms.append(mask);rep.append(h);ph.append(physical(task.ins["nodes"],batch))
    if err>.002:raise ValueError(f"Checkpoint logits disagree by {err}")
    np.savez_compressed(HERE/f"out/{index}.npz",mask=np.asarray(ms,np.uint8),embed=np.asarray(rep,np.float32),phys=np.asarray(ph,np.float32))
    report=dict(case=case,contexts=len(rep),scale=policy.edge_scale,max_logit_error=err,cpu_s=time.process_time()-start)
    (HERE/f"out/{index}.json").write_text(json.dumps(report,allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
