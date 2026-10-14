"""Same public input/weights, original checkpoint dtype vs explicit CPUfloat32."""
import argparse,hashlib,json,os,pickle,sys,time
from datetime import datetime,timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['JAX_PLATFORM_NAME']='cpu'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'actual_masked_fleet_engineering_v1'))
import masked_fleet as mf
import numpy as np

def run(scripts,upstream,checkpoint,packet_path,out):
    mf.bootstrap(scripts);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    reg=dict(id='P-ACTUAL-ORIGINAL-DTYPE-VS-CPU32-ENGINEERING-v1',registered=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=sha(checkpoint),packet_pickle_sha256=sha(packet_path),source_sha256=sha(__file__),
        same_known_packet=True,new_days=0,no_equivalence_threshold_predeclared=True,formal_gain=False)
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    from delivery_maskco import MaskCORouteProvider,partial_adjacency
    from flax import nnx
    import jax.numpy as jnp
    provider=MaskCORouteProvider(checkpoint,upstream)
    from training import load_ckpt
    params,_,_,cfg,_,_=load_ckpt(checkpoint)
    original=nnx.merge(nnx.graphdef(cfg.construct_model()),params)
    packet=pickle.loads(Path(packet_path).read_bytes())
    vids,pool,ids,loc,inst,routes,comm=mf.topology(packet)
    coords=np.asarray(inst['coords'],np.float32);centered=coords-coords.mean(axis=0,keepdims=True)
    scale=max(1e-8,float(np.linalg.norm(centered,axis=1).mean()))
    raw=np.concatenate((centered/scale,np.asarray(inst['demands'],np.float32)[:,None]/packet['capacity']),axis=1)[None]
    removed=set(map(int,np.random.default_rng(202610101903).choice(pool,size=1,replace=False)))
    adj=partial_adjacency(routes,{loc[o] for o in removed},len(ids)+1)
    timestep=min(1.,float(adj[1:].sum())/(2*max(1,len(ids))))
    h32=provider.features(inst,packet['capacity']);l32=provider.logits(h32,adj)
    encode=nnx.jit(lambda m,x:m.encode(x));decode=nnx.jit(lambda m,h,t,a:m.decode(h,t,a,target='logit'))
    ho=encode(original,jnp.asarray(raw));lo=np.asarray(decode(original,ho,jnp.asarray([timestep],jnp.float32),jnp.asarray(adj[None],jnp.float32)))[0]
    def diff(a,b):
        a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
        return dict(shape=list(a.shape),both_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
            max_abs=float(np.max(np.abs(a-b))),relative_l2=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)),bitwise_equal=bool(np.array_equal(a,b)))
    result=dict(registration=reg,original_config_dtype=str(cfg.dtype),adapter_dtype='float32',softcap=cfg.softcap,final_softcap=cfg.final_softcap,
        qk_norm=cfg.qk_norm,raw_input_sha256=mf.array_hash(raw),adjacency_sha256=mf.array_hash(adj),
        encoder_difference=diff(h32,ho),logits_difference=diff(l32,lo),
        original_logits_sha256=mf.array_hash(np.asarray(lo,np.float32)),adapter_logits_sha256=mf.array_hash(np.asarray(l32,np.float32)),
        checkpoint_end_sha256=sha(checkpoint),numerical_equivalence_claim=False,formal_online_gain=False)
    np.savez(out/'same_input_outputs.npz',raw=raw,adjacency=adj,encoder_original=np.asarray(ho,np.float32),encoder_adapter=np.asarray(h32,np.float32),
        logits_original=np.asarray(lo,np.float32),logits_adapter=np.asarray(l32,np.float32))
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--packet',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.scripts,a.upstream,a.checkpoint,a.packet,a.out)))
