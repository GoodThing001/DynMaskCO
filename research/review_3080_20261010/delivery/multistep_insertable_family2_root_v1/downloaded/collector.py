"""Registered fresh family-2 data only; no model training or confirmation."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'multistep_full_gradient_root_v2'
OLD=BASE.parent/'remote_finite_research_root_v2';EXT=OLD/'extension'
for p in (BASE,PRIOR,OLD/'upstream',OLD/'actual_checkpoint_engineering_v1',EXT/'scripts/delivery',EXT/'scripts/coldchain',EXT/'scripts/research_20261010'):sys.path.insert(0,str(p))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    import numpy as np,psutil
    from retained_gls import solve,adjacency
    from delivery_maskco import partial_adjacency
    from delivery_equal_schedule import generate,make_schedule,reconstruct
    from delivery_evaluator import evaluate
    from delivery_contract import DeliveryContract
    from ortools.constraint_solver import pywrapcp
    from full_gradient import closure
    psutil.Process().cpu_affinity([15]);pywrapcp.DefaultRoutingSearchParameters()
    reg=read(BASE/'RESEARCH_REGISTRATION.json');bind=read(BASE/'EXTERNAL_REGISTRATION_BINDING.json')
    if sha(BASE/'RESEARCH_REGISTRATION.json')!=bind['registration_sha256']:raise ValueError('Registration binding')
    if sha(__file__)!=reg['collector_sha256'] or sha(BASE/'retained_gls.py')!=reg['teacher_sha256']:raise ValueError('Actual source identity')
    if reg['family_index']!=2 or reg['train_instances']!=64 or reg['states_per_instance']!=4:raise ValueError('Finite scientific scope')
    contract=DeliveryContract()
    if contract.identity()!=reg['contract_sha256']:raise ValueError('Actual contract identity')
    reference=read(BASE/'QUALIFIED_SERVER_REFERENCE.json')['files']
    for path,h in reference.items():
        if sha(path)!=h:raise ValueError('Actual qualified source/runtime changed: '+path)
    closure() # Initialize inventory imports before the first scientific observation.
    def sealed():
        out=closure()
        for n in ('collector.py','retained_gls.py','RESEARCH_REGISTRATION.json','EXTERNAL_REGISTRATION_BINDING.json','QUALIFIED_SERVER_REFERENCE.json','supervisor.py'):out[str((BASE/n).resolve())]=sha(BASE/n)
        return out
    start=sealed();save('SOURCE_START.json',{'files':start,'cpu':psutil.Process().cpu_affinity(),'python':sys.version,'registration_sha256':bind['registration_sha256'],'reference_files_rechecked':len(reference),'before_fresh_data':True})
    checks=[];counts={'states':0,'positive':0,'negative':0,'small':0,'oracle_exact':0,'teacher_candidates':0};begin=time.perf_counter()
    def ck(n,v):
        checks.append({'name':n,'pass':bool(v)})
        return bool(v)
    def legal(route,initial,mask,partial,n):
        if len(route)!=contract.vehicles or sorted(o for r in route for o in r)!=list(range(1,n)):return False
        loc={o:(v,k) for v,r in enumerate(route) for k,o in enumerate(r)};unmasked=set(range(1,n))-set(mask)
        return all(all(loc[o][0]==v for o in r if o in unmasked) and all(loc[b][1]>loc[a][1] for a,b in zip([o for o in r if o in unmasked],[o for o in r if o in unmasked][1:])) for v,r in enumerate(initial)) and bool(np.all(adjacency(route,n)[partial==1]==1))
    try:
        with (BASE/'STATES.jsonl').open('x',encoding='utf-8') as f:
            for instance_id in range(64):
                data,incumbent=generate(reg['seeds']['train'],instance_id)
                masks=make_schedule(51,4,reg['seeds']['teacher_mask'],instance_id,fraction=.5)
                for step,mask in enumerate(masks):
                    t=time.perf_counter();deadline=t+10.;initial=copy.deepcopy(incumbent);part=partial_adjacency(initial,set(mask),51)
                    with (BASE/'REQUESTS_STARTED.jsonl').open('a',encoding='utf-8') as journal:
                        journal.write(json.dumps({'instance_id':instance_id,'step':step,'data':data,'initial':initial,'mask':mask},allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                    before=evaluate(data,initial,contract,deadline=deadline)
                    if not before['hard'] or not before['complete']:raise ValueError('Unqualified initial; preserve, no screen/resample')
                    teacher=solve(data,initial,part.tolist(),contract,deadline,mask,search_s=3.)
                    hroute,remaining,insertions,status=reconstruct(data,initial,contract,set(mask),part,None,0.,deadline,time.perf_counter)
                    hreport=evaluate(data,hroute,contract,deadline=deadline) if not remaining else None
                    # Offline reachability probe is diagnostic, never a deployed treatment.
                    ideal=adjacency(teacher['routes'],51).astype(float)
                    oracle,orem,oins,ostatus=reconstruct(data,initial,contract,set(mask),part,ideal,1.,deadline,time.perf_counter)
                    oreport=evaluate(data,oracle,contract,deadline=deadline) if not orem else None
                    ontime=time.perf_counter()<deadline
                    accept=bool(ontime and not remaining and hreport['hard'] and hreport['complete'] and legal(hroute,initial,mask,part,51) and hreport['cost']<before['cost'])
                    next_routes=copy.deepcopy(hroute if accept else initial)
                    delta=before['cost']-teacher['report']['cost'];label=0 if abs(delta)<=1e-8 else (1 if delta>0 else -1)
                    exact=not orem and oracle==teacher['routes'] and oreport['hard'] and oreport['complete']
                    key=str((instance_id,step))
                    teacher_legal=legal(teacher['routes'],initial,mask,part,51)
                    ck(key+':full_C0_and_action',teacher_legal and teacher['report']['hard'] and teacher['report']['complete'])
                    ck(key+':request_within_registered_cap',ontime)
                    ck(key+':H_complete_legal',not remaining and hreport['hard'] and hreport['complete'] and legal(hroute,initial,mask,part,51))
                    # An unreachable target remains in the raw file; training is gated off.
                    ck(key+':ideal_input_exact_deployment_reachability',exact)
                    row={'instance_id':instance_id,'step':step,'request':{'data':data,'initial':initial,'mask':mask},'partial':part.tolist(),'teacher':teacher,'initial_report':before,'label':label,'delta_cost':delta,'deployment':{'routes':hroute,'report':hreport,'remaining':remaining,'insertions':insertions,'status':status,'accepted':accept,'next_routes':next_routes},'oracle':{'routes':oracle,'report':oreport,'remaining':orem,'insertions':oins,'status':ostatus,'exact_target':bool(exact)},'wall_s':time.perf_counter()-t,'timed_out':not ontime,'trajectory_driver':'nonlearning_H_not_teacher_or_oracle'}
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    counts['states']+=1;counts[{1:'positive',-1:'negative',0:'small'}[label]]+=1;counts['oracle_exact']+=int(exact);counts['teacher_candidates']+=len(teacher['all_candidates'])
                    incumbent=next_routes
                    with (BASE/'progress.json').open('w',encoding='utf-8') as p:json.dump({'counts':counts,'instance_id':instance_id,'step':step,'elapsed_s':time.perf_counter()-begin},p)
                    if not teacher_legal or not teacher['report']['hard']:raise ValueError('Hard teacher failure; no policy or data repair in this run')
    finally:
        end=sealed();changed=[p for p,h in start.items() if end.get(p)!=h];extra=sorted(set(end)-set(start))
        ck('actual_environment_closed',not changed and not extra)
        for path,h in reference.items():ck('actual_reference_end:'+path,sha(path)==h)
        save('SOURCE_END.json',{'files':end,'changed':changed,'new_files':extra})
    ck('all256_registered_states',counts['states']==256)
    failed=sum(not c['pass'] for c in checks)
    save('RESULT.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'counts':counts,'independent_instances':64,'request_s':10,'teacher_search_s':3,'scientific_teacher_producer_checks_passed':failed==0,'independent_data_acceptance':False,'training_execution_enabled':False,'confirmation_execution_enabled':False,'scientific_training_qualified':False,'stable_MaskCO_gain':False})
    print(json.dumps({'counts':counts,'failed':failed}),flush=True);return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('FAILURE.json',{'traceback':traceback.format_exc(),'training_execution_enabled':False,'no_resampling':True});raise
