"""Source-only independent timed/partial-state replay checker.

Shares original JAX/Transformer/C0; never imports the new engine, head, action
consumer, local filters, feature mapper, shared worker or caller as reference.
Replays only evidence produced by a separately registered NEW timed integration.
Closed old forward/chain QA is not invoked. Missing reached-stage evidence fails;
an unreached requested fault is reported unexercised, never silently retried.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,hashlib,io,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,digest,closure,delta

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    import numpy as np,psutil,jax,jax.numpy as jnp
    from flax import nnx
    reg=read(BASE/'EXECUTION_REGISTRATION.json')
    if reg.get('execution_authorized') is not True or reg.get('independent_timed_request_checker_sealed') is not True:
        raise RuntimeError('Separate accurate execution registration required')
    if (reg['total_request_seconds'],reg['child_seconds'],reg['reap_reserve_seconds'])!=(10,8,2):
        raise ValueError('Unchanged10/8/2 protocol')
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Unchanged independent8GiB resource barrier')
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('Registered source identity: '+n)
    row=read(BASE/'CHAIN_INPUT.json')
    if row['key']!=[12,3] or len(row['remaining'])!=25 or digest(row)!=reg['input_digest']:
        raise ValueError('Fixed pure current input')
    old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
    for p in (up,old/'actual_checkpoint_engineering_v1',ext/'scripts/delivery',ext/'scripts/coldchain'):
        sys.path.insert(0,str(p))
    from inspect_checkpoint import load_trusted
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    contract=DeliveryContract();physics=contract.physics()
    if contract.identity()!=reg['contract_hash']:raise ValueError('Original contract identity')
    for n,h in reg['normalization_sources'].items():
        if sha(up/n)!=h:raise ValueError('Original normalization source identity')
    saved,checkpoint_sha=load_trusted(Path(reg['checkpoint']),up)
    if checkpoint_sha!=reg['checkpoint_sha256']:raise ValueError('Original checkpoint identity')
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
    pretrained=copy.deepcopy(saved['params']);k=np.asarray(pretrained['init_proj']['kernel'].value)
    pretrained['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0))
    del saved;gc.collect()
    rng=nnx.Rngs(20261012)
    hin=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
    hout=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
    hp=[np.asarray(hin.kernel.value),np.asarray(hin.bias.value),np.asarray(hout.kernel.value),np.asarray(hout.bias.value)]
    def inventory(params):
        return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),
            sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in params.flat_state()}
    head_id={str((a,b)):dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
        for a,b,v in [('input','kernel',hp[0]),('input','bias',hp[1]),('output','kernel',hp[2]),('output','bias',hp[3])]}
    data=row['data'];n=len(data['demands'])
    if any(float(x)!=0. for name in ('ready_time','reveal_time') for x in data[name]):
        raise ValueError('This registered old visible snapshot domain only')
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
    def retained(routes):
        union=set()
        for route in routes:
            for a,b in zip([0]+route,route+[0]):
                if a!=b:union.add((min(a,b),max(a,b)))
        return required<=union
    def route_ok(route):
        if sum(float(data['demands'][i]) for i in route)>contract.capacity+1e-9:return False
        t=contract.loading_service_h;prev=0
        for i in route:
            if max(data['ready_time'][i],data['reveal_time'][i])>1e-9:return False
            t=max(t+dist[prev,i]/contract.speed_kmph,data['tw_start'][i])
            if t>data['tw_end'][i]+1e-9:return False
            t+=data['service_time'][i];prev=i
        return bool(t+dist[prev,0]/contract.speed_kmph<=contract.horizon_h+1e-9)
    def action_state(routes,remaining):
        actions=[];meta=[]
        for customer in remaining:
            for v,route in enumerate(routes):
                for p in range(len(route)+1):
                    a=route[p-1] if p else 0;b=route[p] if p<len(route) else 0
                    trial=copy.deepcopy(routes);trial[v].insert(p,customer)
                    actions.append([customer,v,p,a,b])
                    meta.append((retained(trial),route_ok(trial[v]),float((dist[a,customer]+dist[customer,b]-dist[a,b])/ds)))
        return np.asarray(actions,np.int32),meta
    def anchors(actions,routes):
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
        return context,timestep,weights,scalars
    b=io.BytesIO();np.savez(b,x=np.zeros(1));b.seek(0);np.load(b).close()
    warm=cfg.construct_model();del warm;jax.devices();list(psutil.process_iter(['pid']));gc.collect()
    extra=[p for p in BASE.rglob('*') if p.is_file()]+[Path(reg['checkpoint'])]
    trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain']
    start=closure(extra,trees);save(BASE,'TIMED_AUDIT_SOURCE_START.json',start)
    checks=[];unexercised=[];partials=[];missing_source_end=[];largest=0.;forward_count=0
    with (BASE/'TIMED_AUDIT_CHECKS.jsonl').open('x',encoding='utf-8') as ledger:
        def ck(name,value):
            entry=dict(name=name,passed=bool(value));checks.append(entry)
            ledger.write(json.dumps(entry,separators=(',',':'))+'\n');ledger.flush()
        ck('engineering_scope_fixed',reg['optimizer_steps']==0 and reg['head_seed']==20261012 and reg.get('scientific_family3_started') is False)
        # Every registered request is inspected once; failures are retained.
        for spec in reg['requests']:
            arm=spec['arm'];fault=spec['fault_stage'];tag=arm+'_fault%d'%fault;root=BASE/(tag+'_worker');q=root/'request'
            receipt=read(BASE/(tag+'_CALLER_RECEIPT.json'));ident=read(root/'INITIAL_PARAMETERS.json')
            ck(tag+':fixed_receipt_scope',receipt['arm']==arm and receipt['fault_stage']==fault and not receipt['live_installation'] and not receipt['full_request_budget_qualified'])
            s=receipt['start_clock'];e=receipt['end_clock'];deadline=s+8.
            ck(tag+':exact_elapsed_and_deadlines',receipt['elapsed']==e-s and receipt['child_deadline']==deadline and receipt['total_deadline']==s+10. and receipt['within_total_window']==(e<=s+10.))
            original_process_alive=False
            try:
                current=psutil.Process(receipt['child_pid'])
                original_process_alive=current.create_time()==receipt['child_identity']['create_time'] and current.cmdline()==receipt['child_identity']['cmdline']
            except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
            ck(tag+':actual_reap_within_window',receipt['worker_reaped'] and receipt['reap_clock']<=e<=s+10. and not original_process_alive)
            ck(tag+':returncode_not_invented',receipt['natural_exit']==(receipt['kill_clock'] is None) and (receipt['kill_clock'] is None or receipt['actual_worker_returncode']!=0))
            if receipt['kill_clock'] is not None:ck(tag+':kill_after_child_window',deadline<=receipt['kill_clock']<=receipt['reap_clock'])
            stages=sorted(root.glob('STAGE_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
            for p in stages:
                st=read(p);ck(tag+':'+p.stem,st['deadline']==deadline and s<=st['clock']<deadline)
                for f,h in st['evidence_files'].items():ck(tag+':stage_artifact:'+f,sha(root/f)==h)
                required_files={2:['request/FEATURES.npz'],3:['request/FEATURES.npz','request/INCUMBENT_C0.json','request/ENCODED.npz'],
                    4:['request/00.npz'],5:['request/CHAIN_STOP.json'],6:['request/C0_SELECTION.json']}
                ck(tag+':required_stage_evidence:'+p.stem,all(f in st['evidence_files'] for f in required_files.get(st['stage'],[])))
                if st['stage']>=7:ck(tag+':serialized_evidence:'+p.stem,(root/'SERIALIZED_RESPONSE.json').is_file())
                if st['stage']>=8:ck(tag+':published_evidence:'+p.stem,(root/'PUBLICATION.json').is_file())
            reached=receipt['actual_stage']
            ck(tag+':actual_stage_bound',reached==(read(stages[-1])['stage'] if stages else 0))
            if fault:
                if (root/'FAULT_DELAY_ENTERED.json').exists():
                    entered=read(root/'FAULT_DELAY_ENTERED.json')
                    ck(tag+':actual_fault_delay_reached',entered['stage']==fault and s<=entered['clock']<deadline)
                else:unexercised.append(dict(tag=tag,requested_stage=fault,actual_stage=reached))
            if arm=='pretrained_initial':params=pretrained
            elif arm=='random888_initial':
                cfg3=copy.deepcopy(cfg);cfg3.encoder_input_dim=3;fresh=cfg3.construct_model();params=copy.deepcopy(nnx.state(fresh,nnx.Param))
                k=np.asarray(params['init_proj']['kernel'].value);params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0));del fresh;gc.collect()
            else:raise ValueError('Unregistered initial arm')
            model=cfg.construct_model();nnx.update(model,params);pid=inventory(nnx.state(model,nnx.Param))
            source=digest(dict(arm=arm,backbone=pid,head=head_id,registration_sha256=sha(BASE/'EXECUTION_REGISTRATION.json'),optimizer_steps=0))
            ck(tag+':independent_full_initial_identity',ident==dict(backbone=pid,head=head_id,source_identity=source) and len(pid)==234 and sum(np.asarray(v.value).size for _,v in nnx.state(model,nnx.Param).flat_state())==76263424)
            incumbent=None
            if (q/'FEATURES.npz').exists():
                with np.load(q/'FEATURES.npz',allow_pickle=False) as z:ck(tag+':independent_full10D_exact',np.array_equal(features,z['features']))
            if (q/'INCUMBENT_C0.json').exists():
                incumbent=evaluate(data,row['initial'],contract,wave_h=0.);ck(tag+':original_incumbent_C0_exact',incumbent==read(q/'INCUMBENT_C0.json') and incumbent['hard'])
            encoded=None
            if (q/'ENCODED.npz').exists():
                encoded=np.asarray(model.encode(jnp.asarray(features,jnp.float32),attn_options={}))
                with np.load(q/'ENCODED.npz',allow_pickle=False) as z:ck(tag+':independent_encode_exact',np.array_equal(encoded,z['encoded']) and np.array_equal(features,z['features']))
            routes=copy.deepcopy(row['before']);remaining=row['remaining'][:];steps=[]
            for step in range(25):
                scorepath=q/('%02d.npz'%step);trpath=q/('%02d.json'%step)
                if not scorepath.exists():
                    ck(tag+':no_transition_without_scores:'+str(step),not trpath.exists());break
                actions,meta=action_state(routes,remaining);context,timestep,weights,scalars=anchors(actions,routes)
                with np.load(scorepath,allow_pickle=False) as z:
                    ck(tag+':anchors:'+str(step),np.array_equal(z['actions'],actions) and np.array_equal(z['context'],context) and np.array_equal(z['timestep'],timestep) and np.array_equal(z['features'],features) and all(np.array_equal(z[name],w) for name,w in zip(['route_weights','prefix_weights','suffix_weights'],weights)) and np.array_equal(z['slot_position_length'],scalars))
                    # New timed-request first scores are actual, never old00 reuse.
                    fresh=model.encode(jnp.asarray(features,jnp.float32),attn_options={})
                    latent=model.decode(fresh,jnp.asarray(timestep),jnp.asarray(context),attn_options={},target='embed')
                    if model.apply_final_norm:latent=model.final_norm(latent)
                    hj=latent[0].astype(jnp.float32);aa=jnp.asarray(actions)
                    xxj=jnp.concatenate([hj[aa[:,0]],hj[aa[:,3]],hj[aa[:,4]]]+[jnp.asarray(w)@hj for w in weights]+[jnp.asarray(scalars)],axis=-1)
                    expected=np.asarray(hout(jnp.tanh(hin(xxj))))[:,0];scores=z['scores'];h=z['latent'][0]
                    ck(tag+':full_forward_exact:'+str(step),encoded is not None and np.array_equal(np.asarray(fresh),encoded) and np.array_equal(np.asarray(latent),z['latent']) and np.array_equal(expected,scores))
                    ck(tag+':float32_finite:'+str(step),scores.dtype==np.float32 and scores.shape==(len(actions),) and np.isfinite(scores).all())
                    xx=np.concatenate([h[actions[:,0]],h[actions[:,3]],h[actions[:,4]]]+[w@h for w in weights]+[scalars],axis=1)
                    ref=(np.tanh(xx@hp[0]+hp[1])@hp[2]+hp[3])[:,0]
                    error=float(np.max(np.abs(ref.astype(np.float64)-scores.astype(np.float64))));largest=max(largest,error)
                    ck(tag+':NumPy_head_bound:'+str(step),error<=reg['numpy_head_max_abs_error']);forward_count+=1
                if not trpath.exists():partials.append(dict(tag=tag,step=step,scored_but_transition_not_committed=True));break
                tr=read(trpath);groups=[];offset=0
                for customer in remaining:
                    packed=[]
                    while offset<len(actions) and actions[offset,0]==customer:
                        packed.append(actions[offset].tolist()+[bool(meta[offset][0]),bool(meta[offset][1]),float(scores[offset]),meta[offset][2]]);offset+=1
                    eligible=sorted((a for a in packed if a[5] and a[6]),key=lambda a:(-a[7],a[8],a[1],a[2]))
                    groups.append(dict(customer=customer,rows=packed,best=eligible[0] if eligible else None,regret=None if len(eligible)==1 else eligible[0][7]-eligible[1][7] if eligible else 0.))
                feasible=[g for g in groups if g['best'] is not None]
                chosen=max(feasible,key=lambda g:(g['regret'] is None,0. if g['regret'] is None else g['regret'],-g['customer']))['best'] if feasible else None
                ck(tag+':competition:'+str(step),tr['before']==routes and tr['remaining']==remaining and tr['customers']==groups and tr['chosen']==chosen and tr['source_identity']==source and tr['learned_output_qualified'] is False and tr['binding_digest']==digest(dict(data=data,key=row['key'],step=step,before=routes,remaining=remaining,actions=actions.tolist())))
                if chosen is not None:routes[chosen[1]].insert(chosen[2],chosen[0]);remaining.remove(chosen[0])
                ck(tag+':transition:'+str(step),tr['after']==routes and tr['next_remaining']==remaining and tr['status']==('inserted' if chosen is not None else 'no_feasible_action') and retained(routes) and all([i for i in r if i not in row['mask']]==row['before'][v] for v,r in enumerate(routes)))
                steps.append(dict(file=trpath.name,sha256=sha(trpath)))
                if chosen is None:break
            if (q/'CHAIN_STOP.json').exists():ck(tag+':durable_chain_stop',read(q/'CHAIN_STOP.json')==dict(routes=routes,remaining=remaining,steps=steps,status='complete' if not remaining else 'no_feasible_action'))
            candidate=None;choose=False
            if (q/'C0_SELECTION.json').exists():
                candidate=evaluate(data,routes,contract,wave_h=0.) if not remaining else None
                choose=bool(candidate and candidate['hard'] and candidate['cost']<incumbent['cost'])
                ck(tag+':full_C0_selection',read(q/'C0_SELECTION.json')==dict(incumbent=incumbent,candidate=candidate,candidate_selected=choose,routes=routes,remaining=remaining))
            if receipt['selected']:
                raw=(BASE/(tag+'_SELECTED_RAW.json')).read_bytes();out=json.loads(raw.decode('utf-8'))
                ck(tag+':deadline_selection',s<=receipt['writer_clock']<=receipt['validation_clock']<=receipt['selection_clock']<deadline)
                ck(tag+':selected_raw_identity',sha(BASE/(tag+'_SELECTED_RAW.json'))==receipt['raw_identity'] and raw==(root/'SERIALIZED_RESPONSE.json').read_bytes())
                ck(tag+':selected_exact_C0_source_routes',out==read(q/'REPORT_BEFORE_PUBLICATION.json') and out['source_identity']==source and out['source_qualified'] and out['parameters_unchanged'] and out['optimizer_steps']==0 and out['learned_output_qualified'] is False and out['incumbent']==incumbent and out['candidate']==candidate and out['candidate_selected']==choose and out['routes']==routes and out['remaining']==remaining and out['steps']==steps and out['selected_routes']==receipt['selected_routes']==(routes if choose else row['initial']))
            else:ck(tag+':fallback_original',receipt['selected_routes']==row['initial'] and receipt['raw_identity'] is None and not (BASE/(tag+'_SELECTED_RAW.json')).exists())
            prod_start=read(root/'SOURCE_START.json')
            ends=[q/'SOURCE_END.json',root/'FAULT_SOURCE_AT_ENTRY.json']
            if not any(p.exists() for p in ends):missing_source_end.append(dict(tag=tag,scope='Killed request has no completed source-end or registered fault-entry snapshot; begin-only qualification cannot close full scientific workload.'))
            for p in ends:
                if p.exists():
                    pend=read(p)
                    ck(tag+':sealed_loaded_files:'+p.name,not pend['changed'] and not pend['new_files'])
                    for name,h in prod_start['files'].items():ck(tag+':file:'+p.name+':'+name,pend['files'].get(name)==h and sha(name)==h)
            ck(tag+':independent_params_unchanged',inventory(nnx.state(model,nnx.Param))==pid)
            del model;gc.collect()
        end=closure(extra,trees);changes=delta(start,end);save(BASE,'TIMED_AUDIT_SOURCE_END.json',dict(**end,**changes))
        ck('independent_loaded_closure',not changes['changed'] and not changes['new_files'])
        ck('unchanged_checkpoint',sha(reg['checkpoint'])==checkpoint_sha)
    failed=sum(not c['passed'] for c in checks)
    save(BASE,'INDEPENDENT_TIMED_REQUEST_AUDIT.json',dict(checks_count=len(checks),failed=failed,
        failure_names=[c['name'] for c in checks if not c['passed']],unexercised_faults=unexercised,partial_score_records=partials,
        missing_completed_source_end=missing_source_end,complete_all_request_dependency_closure=not missing_source_end,
        new_timed_evidence_forward_replays=forward_count,max_abs_numpy_head_error=largest,
        old_QA_repeated=False,all_registered_faults_exercised=not unexercised,
        raw_replay_qualified=failed==0,full_request_budget_qualified=False,formal_controls_qualified=False,
        full_scientific_loaded_closure_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        scope='Original shared Transformer/JAX/C0; independent current10D/parameters/head/anchors/local geometry/choices and actual persisted timed partial/full evidence. Timing/source markers do not prove native internal preemption, arbitrary caller/disk load, live installation or scientific controls.'))
    return int(failed>0)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'TIMED_AUDIT_FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
