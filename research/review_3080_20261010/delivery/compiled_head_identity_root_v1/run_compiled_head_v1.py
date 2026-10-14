"""Execute only the new head candidate; never backbone/C0/route solver."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import gc,hashlib,io,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity

def main():
    import numpy as np,psutil,jax,jax.numpy as jnp
    from flax import nnx
    from delivery_actual_action_head_v1 import construct_head
    from compiled_action_head_v1 import make_scorer
    native=prime_native_identity();psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Registered8GiB resource barrier')
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    entries=json.loads((BASE/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))
    head=construct_head();state=nnx.state(head,nnx.Param)
    values={str(k):np.asarray(v.value) for k,v in state.flat_state()}
    identity={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    if identity!=reg['head_identity']:raise ValueError('Original four-leaf parameter identity')
    pkeys=["('input', 'kernel')","('input', 'bias')","('output', 'kernel')","('output', 'bias')"]
    np.savez(BASE/'HEAD_PARAMETERS.npz',**{n:values[k] for n,k in zip(('k1','b1','k2','b2'),pkeys)})
    args=[jnp.asarray(values[k]) for k in pkeys];compiled=make_scorer()
    jax.devices();list(psutil.process_iter(['pid']))
    temp=io.BytesIO();np.savez(temp,x=np.zeros(1,np.float32));temp.seek(0);np.load(temp).close()
    extras=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json']
    extras.extend(Path(x['path']) for x in entries)
    start=closure(extras,[]);save(BASE,'SOURCE_START.json',start)
    records=[]
    for index,entry in enumerate(entries):
        p=Path(entry['path'])
        if sha(p)!=entry['sha256']:raise ValueError('Saved input byte identity')
        with np.load(p,allow_pickle=False) as z:
            bound={k:z[k].copy() for k in ('latent','actions','route_weights','prefix_weights','suffix_weights','slot_position_length','scores')}
        h=bound['latent'][0].astype(np.float32);actions=bound['actions']
        began=time.perf_counter()
        out=compiled(jnp.asarray(h),jnp.asarray(actions),*[jnp.asarray(bound[k],jnp.float32) for k in ('route_weights','prefix_weights','suffix_weights','slot_position_length')],*args)
        jax.block_until_ready(out);actual=np.asarray(out)
        ended=time.perf_counter()
        equal=np.array_equal(actual,bound['scores']) and actual.dtype==bound['scores'].dtype
        err=float(np.max(np.abs(actual.astype(np.float64)-bound['scores'].astype(np.float64))))
        name='%02d_OUTPUT.npz'%index;np.savez(BASE/name,scores=actual)
        record=dict(index=index,arm=entry['arm'],step=entry['step'],input_path=str(p),input_sha256=entry['sha256'],
            output_file=name,output_sha256=sha(BASE/name),actions=len(actions),
            began_hex=began.hex(),ended_hex=ended.hex(),elapsed=ended-began,
            exact_float32_scores_equal=equal,max_abs_score_difference=err,
            compile_and_sync_and_host_in_operation=True,no_sample_warmup=True)
        save(BASE,'%02d_RECORD.json'%index,record);records.append(record)
    end=closure(extras,[]);changes=delta(start,end);save(BASE,'SOURCE_END.json',dict(**end,**changes))
    if changes['changed'] or changes['new_files']:raise ValueError('Actual source byte closure')
    result=dict(records=records,count=len(records),exact_equal_count=sum(x['exact_float32_scores_equal'] for x in records),
        strict_output_identity_qualified=all(x['exact_float32_scores_equal'] for x in records),
        source_qualified=True,source_module_additions=sorted(set(end['module_files'])-set(start['module_files'])),
        candidate_attempts=1,original_backbone_executed=False,old_QA_repeated=False,
        no_solver_C0_or_training=True,full_request_budget_qualified=False,scientific_family3_started=False,stable_online_gain=False)
    save(BASE,'RESULT.json',result)
    return 0
if __name__=='__main__':
    code=1
    try:code=main()
    except BaseException:save(BASE,'FAILURE.json',dict(traceback=traceback.format_exc()))
    raise SystemExit(code)
