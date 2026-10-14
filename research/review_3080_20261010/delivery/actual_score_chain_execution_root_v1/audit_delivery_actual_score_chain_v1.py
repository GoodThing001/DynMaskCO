"""Unexecuted independent full-chain checker, sharing original JAX and C0.

Does not import the new head, consumer, options, guard, feature mapper or runner.
Old first scores are hash-bound only; their closed forward QA is not repeated.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,hashlib,io,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,digest,closure,delta


def main():
    import numpy as np,psutil,jax,jax.numpy as jnp
    from flax import nnx
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if not reg.get('execution_authorized') or not reg.get('independent_full_chain_checker_sealed'):
        raise RuntimeError('Exact execution registration required')
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Independent8GiB resource barrier')
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('Audit package identity:'+n)
    row=json.loads((BASE/'CHAIN_INPUT.json').read_text(encoding='utf-8'))
    result=json.loads((BASE/'RESULT.json').read_text(encoding='utf-8'))
    old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
    for p in (up,old/'actual_checkpoint_engineering_v1',ext/'scripts/delivery',ext/'scripts/coldchain'):
        sys.path.insert(0,str(p))
    from inspect_checkpoint import load_trusted
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    contract=DeliveryContract();physics=contract.physics()
    if contract.identity()!=reg['contract_hash'] or digest(row)!=reg['input_digest']:
        raise ValueError('Independent original input/contract identity')
    for n,h in reg['normalization_sources'].items():
        if sha(up/n)!=h:raise ValueError('Original normalization identity')
    prior=Path(reg['initial_component_remote_root'])
    for n,h in reg['initial_artifacts'].items():
        if sha(prior/n)!=h:raise ValueError('Immutable old initial artifact:'+n)
    saved,checkpoint_sha=load_trusted(Path(reg['checkpoint']),up)
    if checkpoint_sha!=reg['checkpoint_sha256']:raise ValueError('Checkpoint identity')
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
    pretrained=copy.deepcopy(saved['params']);k=np.asarray(pretrained['init_proj']['kernel'].value)
    pretrained['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0))
    del saved;gc.collect()
    rng=nnx.Rngs(20261012)
    linear_in=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
    linear_out=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
    hp=[np.asarray(linear_in.kernel.value),np.asarray(linear_in.bias.value),np.asarray(linear_out.kernel.value),np.asarray(linear_out.bias.value)]
    def inventory(params):
        return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),
            sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in params.flat_state()}
    # Source identity uses the production module's parameter names, not imported
    # production code. Expected arrays come from independent fixed NNX RNG order.
    head_identity={str((a,b)):dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
        for a,b,v in [('input','kernel',hp[0]),('input','bias',hp[1]),('output','kernel',hp[2]),('output','bias',hp[3])]}
    data=row['data'];n=len(data['demands'])
    if any(float(x)!=0. for name in ('ready_time','reveal_time') for x in data[name]):
        raise ValueError('Visible snapshot feature domain')
    co=jnp.asarray(data['coords'],jnp.float32)[None];centered=co-jnp.mean(co,axis=-2,keepdims=True)
    normalized=np.asarray(centered/jnp.mean(jnp.linalg.norm(centered,axis=-1,keepdims=True,ord=2),axis=-2,keepdims=True))
    features=np.zeros((1,n,10),np.float32);features[:,:,:2]=normalized
    features[0,:,2]=np.asarray(data['demands'],np.float32)/contract.capacity
    for c,name in enumerate(['tw_start','tw_end','service_time','reveal_time','ready_time'],3):
        features[0,:,c]=np.asarray(data[name],np.float32)/contract.horizon_h
    classes=np.asarray(data['temp_class'],int)
    features[0,:,8]=np.asarray(physics.thermal.target_temperature_c,np.float32)[classes]/20.
    features[0,:,9]=np.asarray(physics.quality.initial_value,np.float32)[classes]/9.
    features[0,0,3:]=0
    x=np.asarray(data['coords'],float);dist=np.sqrt(np.sum((x[:,None]-x[None,:])**2,axis=-1))
    ds=max(1e-9,float(dist[~np.eye(n,dtype=bool)].mean()))
    required={(a,b) for a in range(n) for b in range(a+1,n) if row['partial'][a][b]>0}
    def route_filter(route):
        if sum(float(data['demands'][i]) for i in route)>contract.capacity+1e-9:return False
        t=contract.loading_service_h;prev=0
        for i in route:
            if max(data['ready_time'][i],data['reveal_time'][i])>1e-9:return False
            t=max(t+dist[prev,i]/contract.speed_kmph,data['tw_start'][i])
            if t>data['tw_end'][i]+1e-9:return False
            t+=data['service_time'][i];prev=i
        return bool(t+dist[prev,0]/contract.speed_kmph<=contract.horizon_h+1e-9)
    def retained(routes):
        union=set()
        for route in routes:
            for a,b in zip([0]+route,route+[0]):
                if a!=b:union.add((min(a,b),max(a,b)))
        return required<=union
    buf=io.BytesIO();np.savez(buf,x=np.zeros(1));buf.seek(0);np.load(buf).close()
    warm=cfg.construct_model();del warm;jax.devices();list(psutil.process_iter(['pid']));gc.collect()
    extra=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json',Path(reg['checkpoint']),BASE/'RESULT.json',BASE/'SOURCE_START.json',BASE/'SOURCE_END.json']
    extra.extend(prior/n for n in reg['initial_artifacts'])
    for arm_result in result['results']:
        arm=arm_result['arm'];extra.extend([BASE/(arm+'_ENCODED.npz'),BASE/(arm+'_CHAIN_RESULT.json')])
        for s in arm_result['steps']:
            extra.append(BASE/s['file']);tr=json.loads((BASE/s['file']).read_text(encoding='utf-8'))
            if tr['score_origin']['new_decoder_forward']:extra.append(BASE/tr['score_origin']['file'])
    trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain']
    start=closure(extra,trees);save(BASE,'AUDIT_SOURCE_START.json',start)
    checks=[];replays=[];largest=0.;candidate_count=0
    def ck(name,v):checks.append(dict(name=name,passed=bool(v)))
    ck('fixed_two_initial_arms',[r['arm'] for r in result['results']]==['pretrained_initial','random888_initial'])
    ck('production_qualification_and_scope',result['failed']==0 and result['source_qualified'] and not result['formal_controls_qualified'] and not result['full_request_budget_qualified'] and not result['scientific_family3_started'])
    for out in result['results']:
        arm=out['arm']
        if arm=='pretrained_initial':params=pretrained
        else:
            cfg3=copy.deepcopy(cfg);cfg3.encoder_input_dim=3
            seeded=cfg3.construct_model();params=copy.deepcopy(nnx.state(seeded,nnx.Param))
            k=np.asarray(params['init_proj']['kernel'].value)
            params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0))
            del seeded;gc.collect()
        model=cfg.construct_model();nnx.update(model,params);identity=inventory(nnx.state(model,nnx.Param))
        ck(arm+':full_original_initial234_parameters',identity==json.loads((prior/(arm+'_PARAMETER_IDENTITY.json')).read_text(encoding='utf-8')) and len(identity)==234 and sum(np.asarray(v.value).size for _,v in nnx.state(model,nnx.Param).flat_state())==76263424)
        source_identity=digest(dict(arm=arm,checkpoint_sha256=checkpoint_sha,backbone_identity=identity,
            head_identity=head_identity,optimizer_steps=0,protocol_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
        ck(arm+':bound_untrained_source',out['source_identity']==source_identity and out['optimizer_steps']==0 and not out['learned_output_qualified'])
        encoded=np.asarray(model.encode(jnp.asarray(features,jnp.float32),attn_options={}))
        with np.load(BASE/(arm+'_ENCODED.npz'),allow_pickle=False) as z:
            ck(arm+':new_independent_encoder_exact',np.array_equal(encoded,z['encoded']) and np.array_equal(features,z['features']))
        routes=copy.deepcopy(row['before']);remaining=row['remaining'][:]
        incumbent=evaluate(data,row['initial'],contract,wave_h=0.)
        ck(arm+':full_incumbent_C0_exact',incumbent==out['incumbent'] and incumbent['hard'])
        for step,s in enumerate(out['steps']):
            tr=json.loads((BASE/s['file']).read_text(encoding='utf-8'));tag=arm+':'+str(step)
            ck(tag+':transition_file_identity',sha(BASE/s['file'])==s['sha256'])
            ck(tag+':current_input',tr['before']==routes and tr['remaining']==remaining and tr['step']==step and tr['source_identity']==source_identity and not tr['learned_output_qualified'])
            actions=[];meta=[]
            for customer in remaining:
                for v,route in enumerate(routes):
                    for p in range(len(route)+1):
                        a=route[p-1] if p else 0;b=route[p] if p<len(route) else 0
                        trial=copy.deepcopy(routes);trial[v].insert(p,customer)
                        actions.append([customer,v,p,a,b])
                        meta.append((retained(trial),route_filter(trial[v]),float((dist[a,customer]+dist[customer,b]-dist[a,b])/ds)))
            actions=np.asarray(actions,np.int32);candidate_count+=len(actions)
            ck(tag+':exact_binding_identity',tr['binding_digest']==digest(dict(data=data,key=row['key'],step=step,before=routes,remaining=remaining,actions=actions.tolist())))
            context=np.zeros((1,n,n),np.float32)
            for route in routes:
                for a,b in zip([0]+route,route+[0]):
                    if a!=b:context[0,a,b]=context[0,b,a]=1.
            timestep=np.asarray([context[0,1:].sum()/(2*(n-1))],np.float32)
            weights=[np.zeros((len(actions),n),np.float32) for _ in range(3)]
            scalars=np.zeros((len(actions),3),np.float32)
            for k,(i,v,p,a,b) in enumerate(actions.tolist()):
                route=routes[v]
                for w,part in zip(weights,[route,route[:p],route[p:]]):
                    for j in part:w[k,j]=np.float32(1./len(part))
                scalars[k]=[v/max(1,len(routes)-1),p/(n-1),len(route)/(n-1)]
            origin=tr['score_origin'];source=prior/origin['file'] if step==0 else BASE/origin['file']
            ck(tag+':score_archive_identity',sha(source)==origin['sha256'] and origin['new_decoder_forward']==(step!=0))
            with np.load(source,allow_pickle=False) as z:
                scores=z['scores'];h=z['latent'][0]
                ck(tag+':exact_anchors_features_context',np.array_equal(z['actions'],actions) and np.array_equal(z['context'],context) and np.array_equal(z['timestep'],timestep) and np.array_equal(z['features'],features))
                ck(tag+':exact_current_subsequence_means',all(np.array_equal(z[name],w) for name,w in zip(['route_weights','prefix_weights','suffix_weights'],weights)) and np.array_equal(z['slot_position_length'],scalars))
                ck(tag+':independent_seeded_head_identity',all(np.array_equal(z[name],a) for name,a in zip(['head_input_kernel','head_input_bias','head_output_kernel','head_output_bias'],hp)))
                if step>0:
                    began=time.perf_counter()
                    # Fresh encode for every NEW context is deliberate: verifies
                    # the new production cache against the original full path.
                    fresh=model.encode(jnp.asarray(features,jnp.float32),attn_options={})
                    latent=model.decode(fresh,jnp.asarray(timestep),jnp.asarray(context),attn_options={},target='embed')
                    if model.apply_final_norm:latent=model.final_norm(latent)
                    reloaded=np.asarray(latent)
                    ck(tag+':fresh_full_forward_vs_cache_exact',np.array_equal(np.asarray(fresh),encoded) and np.array_equal(reloaded,z['latent']))
                    hj=latent[0].astype(jnp.float32);aa=jnp.asarray(actions)
                    parts=[hj[aa[:,0]],hj[aa[:,3]],hj[aa[:,4]]]+[jnp.asarray(w)@hj for w in weights]+[jnp.asarray(scalars)]
                    expected_score=np.asarray(linear_out(jnp.tanh(linear_in(jnp.concatenate(parts,axis=-1))))[:,0])
                    ck(tag+':independent_JAX_score_exact',np.array_equal(expected_score,scores))
                    xx=np.concatenate([h[actions[:,0]],h[actions[:,3]],h[actions[:,4]]]+[w@h for w in weights]+[scalars],axis=1)
                    reference=(np.tanh(xx@hp[0]+hp[1])@hp[2]+hp[3])[:,0]
                    error=float(np.max(np.abs(reference.astype(np.float64)-scores.astype(np.float64))));largest=max(largest,error)
                    ck(tag+':independent_numpy_head_bound',error<=reg['numpy_head_max_abs_error'])
                    replay=dict(arm=arm,step=step,seconds=time.perf_counter()-began,latent_exact=bool(np.array_equal(reloaded,z['latent'])),scores_exact=bool(np.array_equal(expected_score,scores)),numpy_head_max_abs_error=error)
                    save(BASE,'REPLAY_'+arm+'_%02d.json'%step,replay);replays.append(replay)
                ck(tag+':finite_ordered_float32_scores',scores.dtype==np.float32 and scores.shape==(len(actions),) and np.isfinite(scores).all())
            customer_records=[];offset=0
            for customer in remaining:
                packed=[]
                while offset<len(actions) and actions[offset,0]==customer:
                    packed.append(actions[offset].tolist()+[bool(meta[offset][0]),bool(meta[offset][1]),float(scores[offset]),meta[offset][2]])
                    offset+=1
                eligible=sorted((x for x in packed if x[5] and x[6]),key=lambda x:(-x[7],x[8],x[1],x[2]))
                best=eligible[0] if eligible else None
                regret=None if len(eligible)==1 else eligible[0][7]-eligible[1][7] if eligible else 0.
                customer_records.append(dict(customer=customer,rows=packed,best=best,regret=regret))
            for customer,actual in zip(customer_records,tr['customers']):
                ck(tag+':customer'+str(customer['customer']),customer==actual)
            ck(tag+':full_customer_groups',len(tr['customers'])==len(customer_records))
            feasible=[x for x in customer_records if x['best'] is not None]
            chosen=max(feasible,key=lambda x:(x['regret'] is None,0. if x['regret'] is None else x['regret'],-x['customer']))['best'] if feasible else None
            ck(tag+':competing_customer_choice',chosen==tr['chosen'])
            if chosen is not None:
                routes[chosen[1]].insert(chosen[2],chosen[0]);remaining.remove(chosen[0])
            ck(tag+':exact_transition',routes==tr['after'] and remaining==tr['next_remaining'] and tr['status']==('inserted' if chosen is not None else 'no_feasible_action'))
            ck(tag+':unmasked_vehicle_and_strict_order',all([i for i in route if i not in row['mask']]==row['before'][v] for v,route in enumerate(routes)) and retained(routes))
            if chosen is None:
                ck(tag+':stop_without_retry',step==len(out['steps'])-1);break
        candidate=evaluate(data,routes,contract,wave_h=0.) if not remaining else None
        choose=bool(candidate and candidate['hard'] and candidate['cost']<incumbent['cost'])
        ck(arm+':full_candidate_C0_and_fallback_exact',candidate==out['candidate'] and choose==out['candidate_selected'] and out['selected_routes']==(routes if choose else row['initial']))
        ck(arm+':final_routes_coverage_status',routes==out['routes'] and remaining==out['remaining'] and out['status']==('complete' if not remaining else 'no_feasible_action'))
        ck(arm+':unchanged_full_parameters',inventory(nnx.state(model,nnx.Param))==identity and out['parameters_unchanged'])
        del model,params;gc.collect()
    production_start=json.loads((BASE/'SOURCE_START.json').read_text(encoding='utf-8'))
    production_end=json.loads((BASE/'SOURCE_END.json').read_text(encoding='utf-8'))
    for p,h in production_start['files'].items():ck('production_source:'+p,production_end['files'].get(p)==h and sha(p)==h)
    ck('no_unsealed_production_file',not production_end['new_files'] and not production_end['changed'])
    ck('unchanged_original_checkpoint',sha(reg['checkpoint'])==checkpoint_sha)
    end=closure(extra,trees);changes=delta(start,end);save(BASE,'AUDIT_SOURCE_END.json',dict(**end,**changes))
    ck('independent_audit_closure',not changes['changed'] and not changes['new_files'])
    failed=sum(not c['passed'] for c in checks)
    save(BASE,'INDEPENDENT_SCORE_CHAIN_AUDIT.json',dict(checks=checks,checks_count=len(checks),failed=failed,
        new_context_full_replays=len(replays),candidate_rows=candidate_count,replays=replays,max_abs_numpy_head_error=largest,
        initial_00_forward_QA_repeated=False,optimizer_steps=0,formal_controls_qualified=False,full_request_budget_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,
        independence_scope='Fresh original parameter reload, independent10D normalization/anchors/seededhead and original full forward for each NEW context versus production encoder cache, exact JAX scores; separate NumPy head arithmetic. Independent geometry/local feasibility/retainededges/regret/competing choices; original shared C0 physics, shared JAX/Transformer implementation.'))
    return int(failed>0)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
