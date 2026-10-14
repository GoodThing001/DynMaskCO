"""Unexecuted two-arm, fixed-first-observed-request integration runner.

Execution is blocked until a separate exact execution registration, independent
full chain checker, supervision and transport are sealed. No scientific data.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import copy, gc, hashlib, io, json, sys, time, traceback
from pathlib import Path
BASE = Path(__file__).resolve().parent
from action_head_runtime_v1 import sha, save, digest, closure, delta


def main():
    import psutil
    reg = json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if reg.get('execution_authorized') is not True:
        raise RuntimeError('Actual integration execution is not authorized by this draft')
    if reg.get('independent_full_chain_checker_sealed') is not True:
        raise RuntimeError('Independent complete-chain checker still required')
    process = psutil.Process(); process.cpu_affinity([15])
    if psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError('Unchanged 8GiB resource barrier')
    for name, h in reg['package_files'].items():
        if sha(BASE/name) != h:
            raise ValueError('Registered package identity: '+name)
    old = Path(reg['old_remote_root']); up = old/'upstream'; ext = old/'extension'
    for p in (up, old/'actual_checkpoint_engineering_v1', BASE,
              ext/'scripts/delivery', ext/'scripts/coldchain'):
        sys.path.insert(0, str(p))
    import numpy as np, jax, jax.numpy as jnp
    from flax import nnx
    from inspect_checkpoint import load_trusted
    from features_objective import make_features, extend_projection, UNITS
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from modules.functional import coord_normalize
    from delivery_actual_action_head_v1 import anchor_weights, construct_head
    from delivery_actual_score_chain_v1 import enumerate_step, binding_identity, consume_step
    row = json.loads((BASE/'CHAIN_INPUT.json').read_text(encoding='utf-8'))
    contract = DeliveryContract()
    if contract.identity() != reg['contract_hash'] or digest(row) != reg['input_digest']:
        raise ValueError('Exact fixed input/contract identity')
    checkpoint = Path(reg['checkpoint'])
    if sha(checkpoint) != reg['checkpoint_sha256']:
        raise ValueError('Original checkpoint identity')
    saved, original_sha = load_trusted(checkpoint, up)
    cfg = copy.deepcopy(saved['model_config']); cfg.dtype='float32'; cfg.encoder_input_dim=10; cfg.rngs=888
    pretrained = copy.deepcopy(saved['params'])
    pretrained['init_proj']['kernel'].value = jnp.asarray(extend_projection(pretrained['init_proj']['kernel'].value))
    del saved; gc.collect()
    head = construct_head()
    def inventory(params):
        return {str(k): dict(shape=list(np.asarray(v.value).shape), dtype=str(np.asarray(v.value).dtype),
                sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k, v in params.flat_state()}
    head_identity = inventory(nnx.state(head, nnx.Param))
    coordinate = np.asarray(coord_normalize(jnp.asarray(row['data']['coords'], jnp.float32)[None]))[0]
    features = make_features(row['data'], contract, coordinate, UNITS, contract.identity())
    contract.physics(); jax.devices(); list(psutil.process_iter(['pid']))
    buf=io.BytesIO(); np.savez(buf, x=np.zeros(1,np.float32)); buf.seek(0); np.load(buf).close()
    # No encoder or decoder forward is prewarmed; original initial full forwards
    # are reused as immutable first-step artifacts rather than executed again.
    prior = Path(reg['initial_component_remote_root'])
    artifacts = [prior/n for n in reg['initial_artifacts']]
    for p in artifacts:
        if sha(p) != reg['initial_artifacts'][p.name]:
            raise ValueError('Previously qualified initial artifact identity: '+p.name)
    extras = [BASE/n for n in reg['package_files']] + [BASE/'EXECUTION_REGISTRATION.json', checkpoint] + artifacts
    trees = [up, ext/'scripts/delivery', ext/'scripts/coldchain']
    start = closure(extras, trees); save(BASE, 'SOURCE_START.json', start)
    results=[]
    for arm in ('pretrained_initial','random888_initial'):
        if arm == 'pretrained_initial': params=pretrained
        else:
            cfg3=copy.deepcopy(cfg); cfg3.encoder_input_dim=3
            seeded=cfg3.construct_model(); params=copy.deepcopy(nnx.state(seeded,nnx.Param))
            params['init_proj']['kernel'].value=jnp.asarray(extend_projection(params['init_proj']['kernel'].value))
            del seeded; gc.collect()
        model=cfg.construct_model(); nnx.update(model,params)
        before_identity=inventory(nnx.state(model,nnx.Param))
        if before_identity != json.loads((prior/(arm+'_PARAMETER_IDENTITY.json')).read_text(encoding='utf-8')):
            raise ValueError('Exact original initial-arm parameters')
        source_identity=digest(dict(arm=arm, checkpoint_sha256=original_sha,
            backbone_identity=before_identity, head_identity=head_identity, optimizer_steps=0,
            protocol_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
        began=time.perf_counter()
        incumbent=evaluate(row['data'],row['initial'],contract,wave_h=0.)
        if not incumbent['hard']:
            raise ValueError('Fixed incumbent fullC0 hard failure; preserve original failure')
        # New prospectively specified cache: features are fixed across the mask
        # chain, but every changed temporary context requires a fresh decode.
        encoded=model.encode(jnp.asarray(features,jnp.float32),attn_options={})
        jax.block_until_ready(encoded)
        with (BASE/(arm+'_ENCODED.npz')).open('xb') as f:
            np.savez(f,encoded=np.asarray(encoded),features=features)
        routes=copy.deepcopy(row['before']); remaining=row['remaining'][:]; steps=[]
        for step in range(len(row['remaining'])):
            actions,groups=enumerate_step(row['data'],routes,remaining,row['partial'],contract)
            binding=anchor_weights(actions,routes,remaining,len(row['data']['demands']))
            tag=arm+'_%02d'%step
            if step==0:
                with np.load(prior/(arm+'_00.npz'),allow_pickle=False) as z:
                    if not np.array_equal(z['features'],features) or not np.array_equal(z['actions'],actions):
                        raise ValueError('First-step original action order/features')
                    for k,v in binding.items():
                        if not np.array_equal(v,z[k]):raise ValueError('First-step binding: '+k)
                    latent=z['latent'].copy(); scores=z['scores'].copy()
                origin=dict(kind='immutable_qualified_initial_component',file=arm+'_00.npz',
                            sha256=reg['initial_artifacts'][arm+'_00.npz'],new_decoder_forward=False)
            else:
                latent=model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),
                                    attn_options={},target='embed')
                if model.apply_final_norm:latent=model.final_norm(latent)
                scores=np.asarray(head(latent,binding)); latent=np.asarray(latent)
                hp={('head_'+str(k[0])+'_'+str(k[1])):np.asarray(v.value) for k,v in nnx.state(head,nnx.Param).flat_state()}
                with (BASE/(tag+'.npz')).open('xb') as f:
                    np.savez(f,latent=latent,scores=scores,features=features,**binding,**hp)
                origin=dict(kind='fresh_current_context_decode',file=tag+'.npz',
                            sha256=sha(BASE/(tag+'.npz')),new_decoder_forward=True)
            packet=dict(scores=scores,source='actual_initial_model_untrained',learned_output_qualified=False,
                binding_digest=binding_identity(row['data'],row['key'],step,routes,remaining,actions),
                source_identity=source_identity)
            transition=consume_step(row['data'],row['key'],step,routes,remaining,actions,groups,packet,source_identity)
            transition.update(step=step,score_origin=origin)
            save(BASE,tag+'.json',transition); steps.append(dict(file=tag+'.json',sha256=sha(BASE/(tag+'.json'))))
            routes=transition['after']; remaining=transition['next_remaining']
            if transition['chosen'] is None:break
        candidate=evaluate(row['data'],routes,contract,wave_h=0.) if not remaining else None
        choose=bool(candidate and candidate['hard'] and candidate['cost']<incumbent['cost'])
        unchanged=inventory(nnx.state(model,nnx.Param))==before_identity and inventory(nnx.state(head,nnx.Param))==head_identity
        result=dict(arm=arm,key=row['key'],steps=steps,routes=routes,remaining=remaining,
            status='complete' if not remaining else 'no_feasible_action',incumbent=incumbent,candidate=candidate,
            selected_routes=routes if choose else row['initial'],candidate_selected=choose,
            source_identity=source_identity,parameters_unchanged=unchanged,wall_seconds=time.perf_counter()-began,
            optimizer_steps=0,learned_output_qualified=False,whole_request_10second_qualified=False)
        save(BASE,arm+'_CHAIN_RESULT.json',result); results.append(result)
        del model,params,encoded; gc.collect()
    end=closure(extras,trees); changes=delta(start,end); save(BASE,'SOURCE_END.json',dict(**end,**changes))
    failed=bool(changes['changed'] or changes['new_files'] or sha(checkpoint)!=original_sha or
                any(not r['parameters_unchanged'] for r in results))
    save(BASE,'RESULT.json',dict(results=results,failed=int(failed),source_qualified=not changes['changed'] and not changes['new_files'],
        scope='Two fixed old input initial-model engineering chains; reuse two old initial score artifacts; new two encodes and at most48 context decodes. Complete C0 shared original implementation; no training or scientific claim.',
        full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,
        full_request_budget_qualified=False,scientific_family3_started=False,stable_online_gain=False))
    return int(failed)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
