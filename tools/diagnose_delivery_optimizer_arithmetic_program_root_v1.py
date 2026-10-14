"""OLD failed first-step arithmetic diagnosis; no optimizer or parameter mutation."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import copy,gc,hashlib,json,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent/'multistep_optimizer_root_v1';TENSOR=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';UP=OLD/'upstream'
for p in (UP,OLD/'actual_checkpoint_engineering_v1',TENSOR):sys.path.insert(0,str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    import numpy as np,jax,jax.numpy as jnp,psutil
    from flax import nnx
    from inspect_checkpoint import load_trusted
    from full_gradient import closure
    psutil.Process().cpu_affinity([15]);reg=json.loads((BASE/'AUDIT_REGISTRATION.json').read_text());source_reg=json.loads((SOURCE/'ENGINEERING_REGISTRATION.json').read_text())
    assert sha(__file__)==reg['program_sha256']
    for n,h in reg['inputs'].items():assert sha(SOURCE/n)==h,n
    saved,h=load_trusted(Path(source_reg['checkpoint']),UP);cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;m=cfg.construct_model();graph=nnx.graphdef(m);ps=copy.deepcopy(saved['params']);a=np.asarray(ps['init_proj']['kernel'].value);ps['init_proj']['kernel'].value=jnp.asarray(np.concatenate((a,np.zeros((7,512),a.dtype)),axis=0));del m,saved
    ts=json.loads((TENSOR/'MATCHED_TENSORS.json').read_text())['rows'][:4]
    batch=[jnp.asarray(np.asarray([t[k] for t in ts]),jnp.float32) for k in ('feature','context','timestep','ce_target','candidate','incumbent','label')]
    def loss(ps,x,c,t,ce,ca,ia,sign,coef):
        m=nnx.merge(graph,ps);lp=jax.nn.log_softmax(m.decode(m.encode(x),t,c),axis=-1);ce_loss=-(lp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2);delta=(lp[:,1:]*(ca[:,1:]-ia[:,1:])).sum(axis=(1,2))/100.;pref=jnp.where(sign==0,0.,jax.nn.softplus(-sign*delta));return (ce_loss+coef*pref).mean()
    vg=jax.jit(jax.value_and_grad(loss));checks=[];arms={};start=closure()
    tiny=np.finfo(np.float32).tiny
    def ftz(x):return np.where(np.abs(x)<tiny,np.copysign(np.float32(0),x),x).astype(np.float32)
    # Separately verify the backend's subnormal multiply behavior.
    operands=np.asarray([tiny/2,-tiny/2,tiny,0.],np.float32)
    actual_probe=np.asarray(jnp.asarray(operands)*jnp.asarray(.5,jnp.float32))
    oldflat=dict(ps.flat_state())
    for arm,coef in (('pretrained_CE',0.),('pretrained_paired',.2)):
        value,grads=vg(ps,*batch,jnp.asarray(coef,jnp.float32));float(value);updated=jax.tree.map(lambda p,g:p-.00001*g,ps,grads);gflat=dict(grads.flat_state());rows=[];stored=json.loads((SOURCE/(arm+'_STEPS.jsonl')).read_text().splitlines()[0]);exact=True
        for path,v in updated.flat_state():
            p=np.asarray(oldflat[path].value);g=np.asarray(gflat[path].value);observed=np.asarray(v.value);plain=p-np.float32(.00001)*g;modeled=ftz(ftz(p)-ftz(np.float32(.00001)*ftz(g)))
            mismatch=plain!=observed
            if np.any(mismatch):
                rows.append({'path':str(path),'different_elements':int(mismatch.sum()),'max_absolute_difference':float(np.max(np.abs(plain[mismatch]-observed[mismatch]))),'all_plain_mismatches_subnormal':bool(np.all(np.abs(plain[mismatch])<tiny)),'all_observed_mismatches_zero':bool(np.all(observed[mismatch]==0)),'FTZ_model_exact':bool(np.array_equal(modeled,observed))})
            exact=exact and np.array_equal(modeled,observed)
            assert hashlib.sha256(observed.tobytes()).hexdigest()==stored['after'][str(path)]['sha256']
            assert hashlib.sha256(g.tobytes()).hexdigest()==stored['gradient'][str(path)]['sha256']
        arms[arm]={'mismatches':rows,'FTZ_model_all234_exact':bool(exact),'gradient_and_actual_update_match_frozen_failure':True,'loss':float(value)};del grads,updated,gflat;gc.collect()
    end=closure()
    for n,h in reg['inputs'].items():assert sha(SOURCE/n)==h,n
    save('NUMERICAL_DIAGNOSIS.json',{'arms':arms,'float32_min_normal':float(tiny),'multiply_probe_input':operands.tolist(),'multiply_probe_observed':actual_probe.tolist(),'all_FTZ_predictions_exact':all(v['FTZ_model_all234_exact'] for v in arms.values()),'source_changed':[p for p,h in start.items() if end.get(p)!=h],'new_unsealed_files':sorted(set(end)-set(start)),'optimizer_steps':0,'frozen_failure_unchanged':True,'scientific_gain':False,'scope':'Actual same CPU/runtime/first4 OLD tensors and exact two frozen gradient/update inventories; controlled backend subnormal multiply probe. No tolerance relaxation or changed optimizer.'})
    print(json.dumps({'arms':{k:{'mismatch_leaves':len(v['mismatches']),'FTZ_exact':v['FTZ_model_all234_exact']} for k,v in arms.items()}}));return int(not all(v['FTZ_model_all234_exact'] for v in arms.values()))
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('AUDIT_FAILURE.json',{'traceback':traceback.format_exc()});raise
