"""Independent NumPy action scorer replay; no backbone forward or physics replay."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import io,json,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,digest,closure,delta

def main():
    import numpy as np,psutil
    psutil.Process().cpu_affinity([15])
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    result=json.loads((BASE/'RESULT.json').read_text(encoding='utf-8'))
    rows=json.loads((BASE/'INPUTS.json').read_text(encoding='utf-8'))
    # NPZ import path initialization before actual audit closure.
    b=io.BytesIO();np.savez(b,x=np.zeros(1));b.seek(0);np.load(b).close()
    old=Path(reg['old_remote_root']);ext=old/'extension'
    sys.path.insert(0,str(ext/'scripts/delivery'));sys.path.insert(0,str(ext/'scripts/coldchain'))
    from delivery_contract import DeliveryContract
    contract=DeliveryContract();physics=contract.physics()
    list(psutil.process_iter(['pid']))
    extra=[BASE/n for n in reg['package_files']]+[BASE/'EXECUTION_REGISTRATION.json',BASE/'RESULT.json',BASE/'SOURCE_START.json',BASE/'SOURCE_END.json']+[BASE/r['output_file'] for r in result['records']]
    start=closure(extra,[]);save(BASE,'AUDIT_SOURCE_START.json',start)
    checks=[];largest=0.;head_bytes=None
    def ck(name,v):checks.append(dict(name=name,passed=bool(v)))
    ck('fixed60_untrained_records',len(result['records'])==60 and result['optimizer_steps']==0 and not result['formal_controls_qualified'])
    ck('fixed_all30_order',digest([r['key'] for r in rows])==reg['ordered_keys_digest'])
    for record in result['records']:
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
            # Coordinate normalization is saved actual upstream output; this audit
            # independently checks the other8 full10D fields and ordered head only.
            expected=np.zeros((1,n,10),np.float32);expected[:,:,:2]=z['features'][:,:,:2]
            expected[0,:,2]=np.asarray(data['demands'],np.float32)/contract.capacity
            for col,field in enumerate(['tw_start','tw_end','service_time','reveal_time','ready_time'],3):expected[0,:,col]=np.asarray(data[field],np.float32)/contract.horizon_h
            classes=np.asarray(data['temp_class'],int)
            expected[0,:,8]=np.asarray(physics.thermal.target_temperature_c,np.float32)[classes]/20.
            expected[0,:,9]=np.asarray(physics.quality.initial_value,np.float32)[classes]/9.
            expected[0,0,3:]=0
            ck(tag+':eight_mapped_fields',np.array_equal(expected,z['features']))
            hp=[z[k] for k in ['head_input_kernel','head_input_bias','head_output_kernel','head_output_bias']]
            identity=[__import__('hashlib').sha256(a.tobytes()).hexdigest() for a in hp]
            if head_bytes is None:head_bytes=identity
            ck(tag+':same_head_all60',identity==head_bytes)
            x=np.concatenate([h[actions[:,0]],h[actions[:,3]],h[actions[:,4]]]+[w@h for w in weights]+[scalars],axis=1)
            reference=(np.tanh(x@hp[0]+hp[1])@hp[2]+hp[3])[:,0]
            error=float(np.max(np.abs(reference.astype(np.float64)-z['scores'].astype(np.float64))));largest=max(largest,error)
            ck(tag+':independent_numpy_score',error<=reg['numpy_head_max_abs_error'] and np.isfinite(reference).all())
    for n,h in reg['package_files'].items():ck('package:'+n,sha(BASE/n)==h)
    production_start=json.loads((BASE/'SOURCE_START.json').read_text(encoding='utf-8'));production_end=json.loads((BASE/'SOURCE_END.json').read_text(encoding='utf-8'))
    for p,h in production_start['files'].items():ck('production_source:'+p,production_end['files'].get(p)==h and sha(p)==h)
    ck('production_no_new_files',not production_end['new_files'])
    end=closure(extra,[]);change=delta(start,end);save(BASE,'AUDIT_SOURCE_END.json',dict(**end,**change))
    ck('audit_closure',not change['changed'] and not change['new_files'])
    failed=sum(not c['passed'] for c in checks)
    save(BASE,'INDEPENDENT_ACTION_HEAD_AUDIT.json',dict(checks=checks,checks_count=len(checks),failed=failed,max_abs_head_error=largest,scorer_replay_qualified=failed==0,independent_backbone_forward=False,independent_coordinate_normalization=False,full_C0_qualified=False,full_budget_qualified=False,scientific_model_qualified=False,stable_online_gain=False))
    return int(failed>0)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False));raise
