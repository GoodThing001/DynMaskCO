"""Preregistered, untrained full-model CPU forwards on fixed old first states."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,importlib.metadata,io,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,digest,closure,delta

def main():
    import psutil
    process=psutil.Process();process.cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB unchanged resource barrier')
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('registered package identity:'+n)
    old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
    for p in (up,old/'actual_checkpoint_engineering_v1',BASE,ext/'scripts/delivery',ext/'scripts/coldchain'):sys.path.insert(0,str(p))
    import numpy as np,jax,jax.numpy as jnp,flax,optax,chex,scipy,ml_dtypes,etils,ortools,google.protobuf
    from flax import nnx
    from inspect_checkpoint import load_trusted,inspect
    from features_objective import make_features,extend_projection,UNITS
    from delivery_contract import DeliveryContract
    from modules.functional import coord_normalize
    from delivery_actual_action_head_v1 import anchor_weights,construct_head,forward
    from ortools.constraint_solver import pywrapcp
    pywrapcp.DefaultRoutingSearchParameters() # imports only, no solver execution
    jax.devices();psutil.process_iter.cache_clear();list(psutil.process_iter(['pid']))
    # Serialization preinitialization is outside all timed forwards.
    buf=io.BytesIO();np.savez(buf,x=np.zeros(1,np.float32));buf.seek(0);np.load(buf).close()
    contract=DeliveryContract();rows=json.loads((BASE/'INPUTS.json').read_text(encoding='utf-8'))
    if len(rows)!=30 or digest([r['key'] for r in rows])!=reg['ordered_keys_digest']:raise ValueError('fixed all30 key order')
    if contract.identity()!=reg['contract_hash']:raise ValueError('actual contract identity')
    checkpoint=Path(reg['checkpoint'])
    if sha(checkpoint)!=reg['checkpoint_sha256']:raise ValueError('actual checkpoint identity')
    # Construct parameters and head before start closure; no forward/gradient/update.
    saved,h=load_trusted(checkpoint,up);info=inspect(saved,h)
    if info['parameter_elements']!=76259840 or info['parameter_leaves']!=234:raise ValueError('full original model')
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10
    original=copy.deepcopy(saved['params']);original['init_proj']['kernel'].value=jnp.asarray(extend_projection(original['init_proj']['kernel'].value))
    del saved;gc.collect()
    head=construct_head();head_state=nnx.state(head,nnx.Param)
    def inventory(params):
        return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),sha256=__import__('hashlib').sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in params.flat_state()}
    head_identity=inventory(head_state)
    if sum(np.asarray(v.value).size for _,v in head_state.flat_state())!=196929 or len(head_identity)!=4:raise ValueError('fixed head architecture')
    extras=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json',checkpoint]
    trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain']
    start=closure(extras,trees)
    save(BASE,'SOURCE_START.json',dict(**start,python=sys.version,versions={n:importlib.metadata.version(n) for n in ['numpy','jax','jaxlib','flax','psutil']},affinity=process.cpu_affinity(),head_identity=head_identity,registration_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
    records=[];checks=[]
    def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
    class CaptureHead:
        def __call__(self,latent,binding):
            self.latent=latent
            return head(latent,binding)
    capture=CaptureHead()
    for arm in ('pretrained_initial','random888_initial'):
        cfg.rngs=888
        if arm=='random888_initial':
            cfg3=copy.deepcopy(cfg);cfg3.encoder_input_dim=3
            random_model=cfg3.construct_model();random_params=copy.deepcopy(nnx.state(random_model,nnx.Param))
            random_params['init_proj']['kernel'].value=jnp.asarray(extend_projection(random_params['init_proj']['kernel'].value))
            del random_model;gc.collect()
        model=cfg.construct_model()
        nnx.update(model,original if arm=='pretrained_initial' else random_params)
        if arm=='random888_initial':del random_params
        params=nnx.state(model,nnx.Param);before=inventory(params)
        ck(arm+':complete_float32_backbone',len(before)==234 and sum(np.asarray(v.value).size for _,v in params.flat_state())==76263424 and all(v['dtype']=='float32' for v in before.values()))
        save(BASE,arm+'_PARAMETER_IDENTITY.json',before)
        for index,row in enumerate(rows):
            data=row['data'];actions=np.asarray(row['actions'],np.int32)
            binding=anchor_weights(actions,row['before'],row['remaining'],len(data['demands']))
            co=np.asarray(coord_normalize(jnp.asarray(data['coords'],jnp.float32)[None]))[0]
            features=make_features(data,contract,co,UNITS,contract.identity())
            began=time.perf_counter()
            scores=np.asarray(forward(model,capture,features,binding))
            elapsed=time.perf_counter()-began
            # Independent head audit uses actual latent outputs, not teacher labels.
            latent=np.asarray(capture.latent)
            name=arm+'_%02d.npz'%index
            hp={('head_'+str(k[0])+'_'+str(k[1])):np.asarray(v.value) for k,v in head_state.flat_state()}
            with (BASE/name).open('xb') as f:np.savez(f,scores=scores,latent=latent,features=features,**binding,**hp)
            ck(name+':finite_float32',scores.shape==(len(actions),) and scores.dtype==np.float32 and np.isfinite(scores).all() and latent.shape==(1,51,512) and np.isfinite(latent).all())
            rec=dict(arm=arm,index=index,key=row['key'],input_digest=digest(row),context_action_digest=digest([row['before'],row['remaining'],row['actions']]),output_file=name,output_sha256=sha(BASE/name),forward_seconds=elapsed,source='actual_initial_model_untrained',learned_output_qualified=False)
            save(BASE,arm+'_%02d.json'%index,rec);records.append(rec)
        ck(arm+':unchanged_backbone',inventory(nnx.state(model,nnx.Param))==before)
        ck(arm+':same_unchanged_head',inventory(nnx.state(head,nnx.Param))==head_identity)
        del model,params;gc.collect()
    end=closure(extras,trees);change=delta(start,end)
    save(BASE,'SOURCE_END.json',dict(**end,**change))
    ck('no_source_change_or_unsealed_file',not change['changed'] and not change['new_files'])
    ck('checkpoint_unchanged',sha(checkpoint)==h)
    failed=sum(not c['passed'] for c in checks)
    save(BASE,'RESULT.json',dict(checks=checks,checks_count=len(checks),failed=failed,records=records,model_executed=True,optimizer_steps=0,source_qualified=not change['changed'] and not change['new_files'],scientific_training_qualified=False,formal_controls_qualified=False,full_C0_qualified=False,full_request_budget_qualified=False,stable_online_gain=False,scope='initial scoring component only; exactly one actual forward per arm/state; auditor uses saved latent, no independent backbone arithmetic/decoder/solver/C0 or scientific claim'))
    return int(failed>0)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
