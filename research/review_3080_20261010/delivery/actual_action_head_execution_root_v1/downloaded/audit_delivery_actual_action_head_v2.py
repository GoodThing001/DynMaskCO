"""Independent checkpoint reload/full forward and NumPy head replay; no physics or update."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,hashlib,io,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,digest,closure,delta

def main():
    import numpy as np,psutil,jax,jax.numpy as jnp
    from flax import nnx
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('unchanged independent 8GiB barrier')
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('registered audit package identity:'+n)
    result=json.loads((BASE/'RESULT.json').read_text(encoding='utf-8'))
    rows=json.loads((BASE/'INPUTS.json').read_text(encoding='utf-8'))
    # NPZ import path initialization before actual audit closure.
    b=io.BytesIO();np.savez(b,x=np.zeros(1));b.seek(0);np.load(b).close()
    old=Path(reg['old_remote_root']);ext=old/'extension';up=old/'upstream'
    sys.path.insert(0,str(up));sys.path.insert(0,str(old/'actual_checkpoint_engineering_v1'))
    from inspect_checkpoint import load_trusted,inspect
    sys.path.insert(0,str(ext/'scripts/delivery'));sys.path.insert(0,str(ext/'scripts/coldchain'))
    from delivery_contract import DeliveryContract
    contract=DeliveryContract();physics=contract.physics()
    if contract.identity()!=reg['contract_hash']:raise ValueError('contract identity')
    if sha(reg['checkpoint'])!=reg['checkpoint_sha256']:raise ValueError('checkpoint identity')
    for name,h in reg['normalization_sources'].items():
        if sha(up/name)!=h:raise ValueError('observed normalization source identity')
    saved,checkpoint_sha=load_trusted(Path(reg['checkpoint']),up)
    info=inspect(saved,checkpoint_sha)
    if info['parameter_elements']!=76259840 or info['parameter_leaves']!=234:raise ValueError('original complete checkpoint')
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
    pretrained=copy.deepcopy(saved['params']);kernel=np.asarray(pretrained['init_proj']['kernel'].value)
    pretrained['init_proj']['kernel'].value=jnp.asarray(np.concatenate([kernel,np.zeros((7,512),kernel.dtype)],axis=0))
    del saved;gc.collect()
    def inventory(params):
        return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in params.flat_state()}
    # Independently reproduce the fixed NNX initialization order, without importing
    # the new head constructor or the production runner/forward helper.
    head_rng=nnx.Rngs(20261012)
    input_linear=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=head_rng)
    output_linear=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=head_rng)
    expected_head=[np.asarray(input_linear.kernel.value),np.asarray(input_linear.bias.value),np.asarray(output_linear.kernel.value),np.asarray(output_linear.bias.value)]
    # Only current visible snapshot coordinates are present: all50 reveal/ready0.
    # Reproduce the observed upstream mean/L2/mean/divide grouping independently.
    def normalized_coordinates(data):
        if any(float(v)!=0. for field in ('reveal_time','ready_time') for v in data[field]):raise ValueError('snapshot visibility domain')
        x=jnp.asarray(data['coords'],jnp.float32)[None]
        centered=x-jnp.mean(x,axis=-2,keepdims=True)
        norms=jnp.linalg.norm(centered,axis=-1,keepdims=True,ord=2)
        return np.asarray(centered/jnp.mean(norms,axis=-2,keepdims=True))
    # Initialize actual model/import/backend and coordinate paths before closure.
    warm_model=cfg.construct_model();del warm_model
    jax.devices();normalized_coordinates(rows[0]['data']);gc.collect()
    list(psutil.process_iter(['pid']))
    extra=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json',BASE/'RESULT.json',BASE/'SOURCE_START.json',BASE/'SOURCE_END.json']+[BASE/r['output_file'] for r in result['records']]+[Path(reg['checkpoint']),BASE/'pretrained_initial_PARAMETER_IDENTITY.json',BASE/'random888_initial_PARAMETER_IDENTITY.json']
    trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain']
    start=closure(extra,trees);save(BASE,'AUDIT_SOURCE_START.json',start)
    checks=[];largest=0.;head_bytes=None;replays=[];model=None;loaded_arm=None;parameter_identity=None
    def ck(name,v):checks.append(dict(name=name,passed=bool(v)))
    ck('fixed60_untrained_records',len(result['records'])==60 and result['optimizer_steps']==0 and not result['formal_controls_qualified'])
    ck('fixed_arm_state_order',[(r['arm'],r['index']) for r in result['records']]==[(arm,i) for arm in ('pretrained_initial','random888_initial') for i in range(30)])
    ck('production_component_pass',result['failed']==0 and result['source_qualified'] and not result['stable_online_gain'])
    ck('fixed_all30_order',digest([r['key'] for r in rows])==reg['ordered_keys_digest'])
    ck('registered_cpu_backend',psutil.Process().cpu_affinity()==[15] and jax.default_backend()=='cpu')
    for record in result['records']:
        if record['arm']!=loaded_arm:
            if model is not None:
                ck(loaded_arm+':unchanged_reloaded_backbone',inventory(nnx.state(model,nnx.Param))==parameter_identity)
                del model;gc.collect()
            if record['arm']=='pretrained_initial':params=pretrained
            elif record['arm']=='random888_initial':
                cfg3=copy.deepcopy(cfg);cfg3.encoder_input_dim=3
                seeded=cfg3.construct_model();params=copy.deepcopy(nnx.state(seeded,nnx.Param))
                k=np.asarray(params['init_proj']['kernel'].value)
                params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0))
                del seeded;gc.collect()
            else:raise ValueError('unregistered arm')
            model=cfg.construct_model();nnx.update(model,params)
            parameter_identity=inventory(nnx.state(model,nnx.Param));loaded_arm=record['arm']
            ck(loaded_arm+':independent_original234_extended_parameters',len(parameter_identity)==234 and sum(np.asarray(v.value).size for _,v in nnx.state(model,nnx.Param).flat_state())==76263424 and all(v['dtype']=='float32' for v in parameter_identity.values()))
            ck(loaded_arm+':actual_parameter_identity',parameter_identity==json.loads((BASE/(loaded_arm+'_PARAMETER_IDENTITY.json')).read_text(encoding='utf-8')))
            del params
        tag=record['arm']+':'+str(record['index']);row=rows[record['index']]
        ck(tag+':original_input_binding',record['key']==row['key'] and record['input_digest']==digest(row) and record['context_action_digest']==digest([row['before'],row['remaining'],row['actions']]))
        ck(tag+':source_and_output_identity',sha(BASE/record['output_file'])==record['output_sha256'] and record['source']=='actual_initial_model_untrained' and record['learned_output_qualified'] is False)
        with np.load(BASE/record['output_file'],allow_pickle=False) as z:
            actions=np.asarray(row['actions'],np.int32);data=row['data'];n=len(data['demands']);before=row['before'];h=z['latent'][0]
            ck(tag+':exact_actions',np.array_equal(z['actions'],actions))
            context=np.zeros((1,n,n),np.float32)
            for route in before:
                path=[0]+route+[0]
                for a,b in zip(path,path[1:]):
                    if a!=b:context[0,a,b]=context[0,b,a]=1
            ck(tag+':current_context',np.array_equal(z['context'],context))
            ck(tag+':current_timestep',np.array_equal(z['timestep'],np.asarray([context[0,1:].sum()/(2*(n-1))],np.float32)))
            weights=[np.zeros((len(actions),n),np.float32) for _ in range(3)];scalars=np.zeros((len(actions),3),np.float32)
            for k,(i,v,p,a,b) in enumerate(actions.tolist()):
                route=before[v]
                ck(tag+':anchor'+str(k),i in row['remaining'] and a==(route[p-1] if p else 0) and b==(route[p] if p<len(route) else 0))
                for w,part in zip(weights,[route,route[:p],route[p:]]):
                    for customer in part:w[k,customer]=np.float32(1./len(part))
                scalars[k]=[v/max(1,len(before)-1),p/(n-1),len(route)/(n-1)]
            for name,w in zip(['route_weights','prefix_weights','suffix_weights'],weights):ck(tag+':'+name,np.array_equal(z[name],w))
            ck(tag+':scalars',np.array_equal(z['slot_position_length'],scalars))
            expected=np.zeros((1,n,10),np.float32)
            expected[:,:,:2]=normalized_coordinates(data)
            ck(tag+':independent_exact_coordinate_normalization',np.array_equal(expected[:,:,:2],z['features'][:,:,:2]) and np.isfinite(expected[:,:,:2]).all())
            expected[0,:,2]=np.asarray(data['demands'],np.float32)/contract.capacity
            for col,field in enumerate(['tw_start','tw_end','service_time','reveal_time','ready_time'],3):expected[0,:,col]=np.asarray(data[field],np.float32)/contract.horizon_h
            classes=np.asarray(data['temp_class'],int)
            expected[0,:,8]=np.asarray(physics.thermal.target_temperature_c,np.float32)[classes]/20.
            expected[0,:,9]=np.asarray(physics.quality.initial_value,np.float32)[classes]/9.
            expected[0,0,3:]=0
            ck(tag+':all10_independent_fields',np.array_equal(expected,z['features']))
            began=time.perf_counter()
            encoded=model.encode(jnp.asarray(expected,jnp.float32),attn_options={})
            embedded=model.decode(encoded,jnp.asarray([context[0,1:].sum()/(2*(n-1))],jnp.float32),jnp.asarray(context),attn_options={},target='embed')
            if model.apply_final_norm:embedded=model.final_norm(embedded)
            reloaded=np.asarray(embedded)
            ck(tag+':fresh_complete_forward_exact',np.array_equal(reloaded,z['latent']) and np.isfinite(reloaded).all())
            replays.append(dict(arm=record['arm'],index=record['index'],seconds=time.perf_counter()-began,latent_sha256=hashlib.sha256(reloaded.tobytes()).hexdigest(),exact=bool(np.array_equal(reloaded,z['latent']))))
            save(BASE,'REPLAY_'+record['arm']+'_%02d.json'%record['index'],replays[-1])
            hp=[z[k] for k in ['head_input_kernel','head_input_bias','head_output_kernel','head_output_bias']]
            ck(tag+':independent_head_seed_initialization',all(np.array_equal(a,b) for a,b in zip(hp,expected_head)))
            identity=[__import__('hashlib').sha256(a.tobytes()).hexdigest() for a in hp]
            if head_bytes is None:head_bytes=identity
            ck(tag+':same_head_all60',identity==head_bytes)
            x=np.concatenate([h[actions[:,0]],h[actions[:,3]],h[actions[:,4]]]+[w@h for w in weights]+[scalars],axis=1)
            reference=(np.tanh(x@hp[0]+hp[1])@hp[2]+hp[3])[:,0]
            error=float(np.max(np.abs(reference.astype(np.float64)-z['scores'].astype(np.float64))));largest=max(largest,error)
            ck(tag+':independent_numpy_score',error<=reg['numpy_head_max_abs_error'] and np.isfinite(reference).all())
    ck(loaded_arm+':unchanged_reloaded_backbone',inventory(nnx.state(model,nnx.Param))==parameter_identity)
    del model;gc.collect()
    ck('checkpoint_end_unchanged',sha(reg['checkpoint'])==checkpoint_sha)
    for n,h in reg['package_files'].items():ck('package:'+n,sha(BASE/n)==h)
    production_start=json.loads((BASE/'SOURCE_START.json').read_text(encoding='utf-8'));production_end=json.loads((BASE/'SOURCE_END.json').read_text(encoding='utf-8'))
    for p,h in production_start['files'].items():ck('production_source:'+p,production_end['files'].get(p)==h and sha(p)==h)
    ck('production_no_new_files',not production_end['new_files'])
    end=closure(extra,trees);change=delta(start,end);save(BASE,'AUDIT_SOURCE_END.json',dict(**end,**change))
    ck('audit_closure',not change['changed'] and not change['new_files'])
    failed=sum(not c['passed'] for c in checks)
    save(BASE,'INDEPENDENT_ACTION_HEAD_AUDIT.json',dict(checks=checks,checks_count=len(checks),failed=failed,max_abs_head_error=largest,scorer_replay_qualified=failed==0,independent_backbone_forward=failed==0,independent_coordinate_normalization=failed==0,replays=replays,independence_scope="Fresh process independent checkpoint reload/seeded initialization/features/context/direct full backbone forward, exact latent identity and separate NumPy head arithmetic; shared original JAX model/runtime, not independent transformer or physical implementation",full_C0_qualified=False,full_budget_qualified=False,scientific_model_qualified=False,stable_online_gain=False))
    return int(failed>0)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
