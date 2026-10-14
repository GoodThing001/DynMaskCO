"""Independent new candidate artifacts; no candidate, old head or backbone import."""
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
