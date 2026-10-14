"""Prospectively seal one compiled-head candidate over all saved inputs."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'research/review_3080_20261010'
PRE=TREE/'delivery/score_split_timed_execution_root_v1'
OUT=TREE/'delivery/compiled_head_identity_root_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    if isinstance(v,str):
        if p.suffix=='.py':ast.parse(v,filename=str(p))
        with p.open('x',encoding='utf-8',newline='\n') as f:f.write(v)
    else:
        with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


CANDIDATE='''"""Single compiled expression schedule; no warmup or model/control claims."""
import jax
import jax.numpy as jnp

def make_scorer():
    # Barriers explicitly retain intermediate arithmetic boundaries; exact
    # agreement is an observed acceptance gate, not assumed from this source.
    barrier=jax.lax.optimization_barrier
    def score(h,actions,route,prefix,suffix,scalars,k1,b1,k2,b2):
        parts=[barrier(h[actions[:,0]]),barrier(h[actions[:,3]]),barrier(h[actions[:,4]])]
        parts.extend([barrier(route@h),barrier(prefix@h),barrier(suffix@h)])
        parts.append(scalars)
        x=barrier(jnp.concatenate(parts,axis=-1))
        hidden=barrier(barrier(x@k1)+b1)
        activated=barrier(jnp.tanh(hidden))
        return barrier(barrier(activated@k2)+b2)[:,0]
    return jax.jit(score)
'''

PRODUCER='''"""Execute only the new head candidate; never backbone/C0/route solver."""
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
'''

AUDIT='''"""Independent new candidate artifacts; no candidate, old head or backbone import."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import hashlib,io,json,time,traceback
from fractions import Fraction
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    import numpy as np,psutil
    native=prime_native_identity();psutil.Process().cpu_affinity([15])
    reg=read(BASE/'EXECUTION_REGISTRATION.json');entries=read(BASE/'SAVED_HEAD_INPUTS.json');result=read(BASE/'RESULT.json')
    temp=io.BytesIO();np.savez(temp,x=np.zeros(1,np.float32));temp.seek(0);np.load(temp).close()
    extras=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json',BASE/'HEAD_PARAMETERS.npz']
    extras.extend(Path(x['path']) for x in entries)
    start=closure(extras,[]);save(BASE,'TIMED_AUDIT_SOURCE_START.json',start)
    checks=[];gates=[]
    def ck(name,test):checks.append(dict(name=name,passed=bool(test)))
    with np.load(BASE/'HEAD_PARAMETERS.npz',allow_pickle=False) as z:
        params={k:z[k].copy() for k in ('k1','b1','k2','b2')}
    names=["('input', 'kernel')","('input', 'bias')","('output', 'kernel')","('output', 'bias')"]
    for short,long in zip(('k1','b1','k2','b2'),names):
        v=params[short];info=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
        ck('actual_parameter:'+short,info==reg['head_identity'][long])
    largest=0.;durations={}
    for i,entry in enumerate(entries):
        rec=read(BASE/('%02d_RECORD.json'%i));source=Path(entry['path'])
        ck('input_hash:%d'%i,sha(source)==entry['sha256']==rec['input_sha256'])
        ck('output_hash:%d'%i,sha(BASE/rec['output_file'])==rec['output_sha256'])
        with np.load(source,allow_pickle=False) as z:
            h=z['latent'][0].astype(np.float32);act=z['actions'];expected=z['scores'];weights=[z[k] for k in ('route_weights','prefix_weights','suffix_weights')];scalar=z['slot_position_length']
        with np.load(BASE/rec['output_file'],allow_pickle=False) as z:actual=z['scores']
        equal=np.array_equal(actual,expected) and actual.dtype==expected.dtype
        error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
        ck('actual_original_exact_gate:%d'%i,rec['exact_float32_scores_equal']==equal and rec['max_abs_score_difference']==error)
        if not equal:gates.append('strict_original_score_identity:%d'%i)
        # Independent NumPy arithmetic is a second, explicitly non-bitwise check.
        x=np.concatenate([h[act[:,0]],h[act[:,3]],h[act[:,4]],*[w@h for w in weights],scalar],axis=-1)
        reference=(np.tanh(x@params['k1']+params['b1'])@params['k2']+params['b2'])[:,0]
        err=float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64))));largest=max(largest,err)
        ck('new_candidate_NumPy_head:%d'%i,err<=reg['numpy_head_max_abs_error'])
        ck('float32_finite_shape:%d'%i,actual.dtype==np.float32 and actual.shape==(len(act),) and np.isfinite(actual).all())
        began=float.fromhex(rec['began_hex']);ended=float.fromhex(rec['ended_hex'])
        ck('exact_operation_clock:%d'%i,ended>=began and ended-began==rec['elapsed'] and rec['compile_and_sync_and_host_in_operation'] and rec['no_sample_warmup'])
        value=Fraction.from_float(ended)-Fraction.from_float(began)
        durations[entry['arm']]=durations.get(entry['arm'],Fraction(0))+value
    ck('all_original_score_gate',result['strict_output_identity_qualified']==(not gates) and result['exact_equal_count']==len(entries)-len(gates))
    ps,pe=read(BASE/'SOURCE_START.json'),read(BASE/'SOURCE_END.json')
    ck('producer_actual_file_closure',ps['files']==pe['files'] and not pe['changed'] and not pe['new_files'])
    for name,h in ps['files'].items():ck('actual_source:'+name,sha(Path(name))==h)
    end=closure(extras,[]);changes=delta(start,end);save(BASE,'TIMED_AUDIT_SOURCE_END.json',dict(**end,**changes))
    ck('independent_actual_loaded_closure',not changes['changed'] and not changes['new_files'])
    save(BASE,'INDEPENDENT_TIMED_REQUEST_AUDIT.json',dict(checks_count=len(checks),failed=sum(not c['passed'] for c in checks),
        failure_names=[c['name'] for c in checks if not c['passed']],checks=checks,
        registered_barrier_failure_names=gates,strict_output_identity_qualified=not gates,
        max_abs_numpy_head_error=largest,completed_operation_exact_seconds={k:[v.numerator,v.denominator] for k,v in durations.items()},
        completed_operation_seconds={k:float(v) for k,v in durations.items()},
        source_module_additions=sorted(set(end['module_files'])-set(start['module_files'])),
        original_backbone_executed=False,old_QA_repeated=False,full_request_budget_qualified=False,
        formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False))
    return 1 if any(not c['passed'] for c in checks) else 0
if __name__=='__main__':
    code=1
    try:code=main()
    except BaseException:save(BASE,'TIMED_AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
    raise SystemExit(code)
'''


def main():
    old=json.loads((PRE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    assert json.loads((PRE/'STAGE_ADJUDICATION.json').read_text(encoding='utf-8'))['independent_failures']==0
    for n,h in old['package_files'].items():assert sha(PRE/n)==h
    OUT.mkdir(exist_ok=False)
    copies=['action_head_runtime_v1.py','timed_evidence_io_v1.py','startup_native_identity_v1.py','delivery_actual_action_head_v1.py']
    for n in copies:shutil.copy2(PRE/n,OUT/n)
    entries=[]
    remote=Path(r'C:\Users\zh\project\MASKCO-Main')/old['remote_directory']
    for arm in ('pretrained_initial','random888_initial'):
        for step in range(17):
            relative=Path(arm+'_fault0_worker/request')/('%02d.npz'%step)
            p=PRE/'downloaded'/relative
            entries.append(dict(arm=arm,step=step,path=str(remote/relative),sha256=sha(p)))
    ident=json.loads((PRE/'downloaded/pretrained_initial_fault0_worker/INITIAL_PARAMETERS.json').read_text(encoding='utf-8'))['head']
    assert ident==json.loads((PRE/'downloaded/random888_initial_fault0_worker/INITIAL_PARAMETERS.json').read_text(encoding='utf-8'))['head']
    put(OUT/'SAVED_HEAD_INPUTS.json',entries)
    put(OUT/'compiled_action_head_v1.py',CANDIDATE)
    put(OUT/'run_compiled_head_v1.py',PRODUCER)
    put(OUT/'audit_compiled_head_v1.py',AUDIT)
    renames={'supervise_delivery_score_split_v1':'supervise_compiled_head_v1','run_delivery_score_split_v1':'run_compiled_head_v1','audit_delivery_score_split_v1':'audit_compiled_head_v1','score_split_remote_launcher_v1':'compiled_head_remote_launcher_v1','score_split_remote_status_v1':'compiled_head_remote_status_v1'}
    def rename(s):
        for a,b in renames.items():s=s.replace(a,b)
        return s
    for original in ('supervise_delivery_score_split_v1.py','score_split_remote_launcher_v1.py','score_split_remote_status_v1.py'):
        put(OUT/(renames[original[:-3]]+'.py'),rename((PRE/original).read_text(encoding='utf-8')))
    tp=ROOT/'tools/transport_delivery_compiled_head_v1.py'
    put(tp,rename((ROOT/'tools/transport_delivery_score_split_v1.py').read_text(encoding='utf-8')).replace('score_split_timed_execution_root_v1','compiled_head_identity_root_v1'))
    put(ROOT/'tools/inspect_delivery_compiled_head_v1.py',rename((ROOT/'tools/inspect_delivery_score_split_v1.py').read_text(encoding='utf-8')).replace('transport_delivery_score_split_v1','transport_delivery_compiled_head_v1'))
    names=copies+['SAVED_HEAD_INPUTS.json','compiled_action_head_v1.py','run_compiled_head_v1.py','audit_compiled_head_v1.py','supervise_compiled_head_v1.py','compiled_head_remote_launcher_v1.py','compiled_head_remote_status_v1.py']
    reg={k:v for k,v in old.items() if k not in ('package_files','transport_tool_sha256','phase_range','requests','actual_phase_design_sha256')}
    reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_SINGLE_COMPILED_HEAD_STRICT_IDENTITY_ENGINEERING',
        prior_registration_sha256=sha(PRE/'EXECUTION_REGISTRATION.json'),preparation_tool_sha256=sha(Path(__file__)),
        package_files={n:sha(OUT/n) for n in names},transport_tool_sha256=sha(tp),head_identity=ident,
        remote_directory='research/review_3080_20261010/compiled_head_identity_root_v1',stage_timeout_seconds=300,
        stage_execution=['run_compiled_head_v1.py','audit_compiled_head_v1.py'],
        inputs_count=34,candidate_attempts=1,maximum_candidate_calls=34,no_backbone_execution=True,
        selection_rule='All34 saved states, both arms step0..16, original order. One fixed optimization-barrier jit candidate, one cold call per row, no retry/alternative choice/warmup. Strict exact float32 output identity is the deployment eligibility gate; any mismatch retains FAIL. NumPy2e-5 is diagnostic only, never a substitute.',
        no_model_science_or_whole_request_execution=True,scientific_family3_started=False,
        numeric_contract_unchanged=True,full_request_budget_qualified=False,learned_output_qualified=False,
        completed_diagnostic_exit0_not_strict_candidate_pass=True)
    put(OUT/'EXECUTION_REGISTRATION.json',reg)
    put(OUT/'LOCAL_EXECUTION_BINDING.json',dict(registration_sha256=sha(OUT/'EXECUTION_REGISTRATION.json'),transport_tool_sha256=sha(tp),package_files=reg['package_files'],shared_MD_upload_authorized=False))
    print(json.dumps(dict(root=str(OUT),registration_sha256=sha(OUT/'EXECUTION_REGISTRATION.json'),candidate_executed=False,science_started=False)))


if __name__=='__main__':main()
