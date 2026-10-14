"""Create new source-only replay drafts, preserve all prior frozen versions."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    source=(ROOT/'audit_delivery_actual_action_head_v1.py').read_text(encoding='utf-8')
    source=source.replace('Independent NumPy action scorer replay; no backbone forward or physics replay.', 'Independent checkpoint reload/full forward and NumPy head replay; no physics or update.')
    source=source.replace('import io,json,sys,traceback','import copy,gc,hashlib,io,json,sys,time,traceback')
    source=source.replace('    import numpy as np,psutil','    import numpy as np,psutil,jax,jax.numpy as jnp\n    from flax import nnx')
    source=source.replace("    psutil.Process().cpu_affinity([15])", "    psutil.Process().cpu_affinity([15])\n    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('unchanged independent 8GiB barrier')")
    source=source.replace("    old=Path(reg['old_remote_root']);ext=old/'extension'", "    old=Path(reg['old_remote_root']);ext=old/'extension';up=old/'upstream'\n    sys.path.insert(0,str(up));sys.path.insert(0,str(old/'actual_checkpoint_engineering_v1'))\n    from inspect_checkpoint import load_trusted,inspect")
    source=source.replace("    contract=DeliveryContract();physics=contract.physics()", """    contract=DeliveryContract();physics=contract.physics()
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
    jax.devices();normalized_coordinates(rows[0]['data']);gc.collect()""")
    source=source.replace("+[BASE/r['output_file'] for r in result['records']]", "+[BASE/r['output_file'] for r in result['records']]+[Path(reg['checkpoint']),BASE/'pretrained_initial_PARAMETER_IDENTITY.json',BASE/'random888_initial_PARAMETER_IDENTITY.json']")
    source=source.replace('    start=closure(extra,[])','    trees=[up,ext/\'scripts/delivery\',ext/\'scripts/coldchain\']\n    start=closure(extra,trees)')
    source=source.replace('    checks=[];largest=0.;head_bytes=None','    checks=[];largest=0.;head_bytes=None;replays=[];model=None;loaded_arm=None;parameter_identity=None')
    source=source.replace("    for record in result['records']:\n", """    ck('registered_cpu_backend',psutil.Process().cpu_affinity()==[15] and jax.default_backend()=='cpu')
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
""")
    old="""            # Coordinate normalization is saved actual upstream output; this audit
            # independently checks the other8 full10D fields and ordered head only.
            expected=np.zeros((1,n,10),np.float32);expected[:,:,:2]=z['features'][:,:,:2]"""
    assert old in source
    source=source.replace(old,"""            expected=np.zeros((1,n,10),np.float32)
            expected[:,:,:2]=normalized_coordinates(data)
            ck(tag+':independent_exact_coordinate_normalization',np.array_equal(expected[:,:,:2],z['features'][:,:,:2]) and np.isfinite(expected[:,:,:2]).all())""")
    source=source.replace("':eight_mapped_fields'","':all10_independent_fields'")
    source=source.replace("            hp=[z[k] for k in", """            began=time.perf_counter()
            encoded=model.encode(jnp.asarray(expected,jnp.float32),attn_options={})
            embedded=model.decode(encoded,jnp.asarray([context[0,1:].sum()/(2*(n-1))],jnp.float32),jnp.asarray(context),attn_options={},target='embed')
            if model.apply_final_norm:embedded=model.final_norm(embedded)
            reloaded=np.asarray(embedded)
            ck(tag+':fresh_complete_forward_exact',np.array_equal(reloaded,z['latent']) and np.isfinite(reloaded).all())
            replays.append(dict(arm=record['arm'],index=record['index'],seconds=time.perf_counter()-began,latent_sha256=hashlib.sha256(reloaded.tobytes()).hexdigest(),exact=np.array_equal(reloaded,z['latent'])))
            save(BASE,'REPLAY_'+record['arm']+'_%02d.json'%record['index'],replays[-1])
            hp=[z[k] for k in""")
    source=source.replace("            identity=[__import__", "            ck(tag+':independent_head_seed_initialization',all(np.array_equal(a,b) for a,b in zip(hp,expected_head)))\n            identity=[__import__")
    source=source.replace("    for n,h in reg['package_files'].items():", "    ck(loaded_arm+':unchanged_reloaded_backbone',inventory(nnx.state(model,nnx.Param))==parameter_identity)\n    del model;gc.collect()\n    ck('checkpoint_end_unchanged',sha(reg['checkpoint'])==checkpoint_sha)\n    for n,h in reg['package_files'].items():")
    source=source.replace('    end=closure(extra,[])','    end=closure(extra,trees)')
    source=source.replace('independent_backbone_forward=False,independent_coordinate_normalization=False','independent_backbone_forward=failed==0,independent_coordinate_normalization=failed==0,replays=replays,independence_scope="Fresh process independent checkpoint reload/seeded initialization/features/context/direct full backbone forward, exact latent identity and separate NumPy head arithmetic; shared original JAX model/runtime, not independent transformer or physical implementation"')
    out=ROOT/'audit_delivery_actual_action_head_v2.py'
    ast.parse(source)
    with out.open('x',encoding='utf-8') as f:f.write(source)
    supervisor=(ROOT/'supervise_delivery_action_head_v1.py').read_text(encoding='utf-8').replace('audit_delivery_actual_action_head_v1.py','audit_delivery_actual_action_head_v2.py')
    ast.parse(supervisor)
    with (ROOT/'supervise_delivery_action_head_v2.py').open('x',encoding='utf-8') as f:f.write(supervisor)
    print('New full independent replay drafts created; AST only; no model execution.')

if __name__=='__main__':main()
