"""Actual server teacher/deployment boundary on fixed OLD derived fixtures only."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;PRIOR=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';EXT=OLD/'extension'
for p in (BASE,PRIOR,OLD/'upstream',OLD/'actual_checkpoint_engineering_v1',EXT/'scripts/delivery',EXT/'scripts/coldchain',EXT/'scripts/research_20261010'):sys.path.insert(0,str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    import numpy as np,psutil
    from retained_gls import solve,adjacency
    from delivery_maskco import partial_adjacency
    from delivery_equal_schedule import reconstruct
    from delivery_evaluator import evaluate
    from delivery_contract import DeliveryContract
    from ortools.constraint_solver import pywrapcp
    from full_gradient import closure
    psutil.Process().cpu_affinity([15]);pywrapcp.DefaultRoutingSearchParameters();reg=read(BASE/'ENGINEERING_REGISTRATION.json');checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    for name,h in reg['inputs'].items():ck('actual_registered_input:'+name,sha(BASE/name)==h)
    ck('registered_program',sha(__file__)==reg['program_sha256']);ck('original_source_binding',sha(PRIOR/'STATES.jsonl')==reg['prior_states_sha256'])
    requests=read(BASE/'REQUESTS.json');contract=DeliveryContract();ck('actual_same_contract',contract.identity()==reg['contract_hash'])
    def own_closure():
        out=closure()
        for n in ('teacher.py','retained_gls.py','REQUESTS.json','ENGINEERING_REGISTRATION.json'):out[str((BASE/n).resolve())]=sha(BASE/n)
        return out
    closure() # Prime closure-imported packages before inventory; no input-dependent work.
    start=own_closure();save('SOURCE_START.json',{'files':start,'python':sys.version,'cpu':psutil.Process().cpu_affinity()});answers=[]
    with (BASE/'ANSWERS.jsonl').open('x',encoding='utf-8') as f:
        for request in requests:
            key=request['key'];data=request['data'];initial=request['initial'];mask=request['mask'];t=time.perf_counter();deadline=t+reg['request_cap_s'];n=len(data['demands'])
            partial=partial_adjacency(initial,set(mask),n);initial_report=evaluate(data,initial,contract,deadline=deadline)
            teacher=solve(data,initial,partial.tolist(),contract,deadline,mask,search_s=reg['teacher_search_cap_s'])
            routes,remaining,insertions,status=reconstruct(data,initial,contract,set(mask),partial,None,0.,deadline,time.perf_counter)
            report=evaluate(data,routes,contract,deadline=deadline);done=time.perf_counter()
            ck(key+':bounded_actual_teacher_and_deployment',done<deadline and initial_report['hard'] and initial_report['complete'] and not remaining and report['hard'] and report['complete'])
            chosen=initial;best=initial_report['cost'];unmasked=set(range(1,n))-set(mask)
            def inserted(route):
                if len(route)!=contract.vehicles:return False
                flat=[o for r in route for o in r]
                if sorted(flat)!=list(range(1,n)):return False
                loc={o:(v,j) for v,r in enumerate(route) for j,o in enumerate(r)}
                return all(all(loc[o][0]==v for o in r if o in unmasked) and all(loc[b][1]>loc[a][1] for a,b in zip([o for o in r if o in unmasked],[o for o in r if o in unmasked][1:])) for v,r in enumerate(initial)) and bool(np.all(adjacency(route,n)[partial==1]==1))
            for j,candidate in enumerate(teacher['all_candidates']):
                label=key+':candidate:'+str(j);c=candidate['report'];valid=inserted(candidate['routes'])
                ck(label+':same_vehicle_order_and_retained_edges',valid)
                accept=valid and c['hard'] and c['cost']<best-1e-12
                ck(label+':actual_fullC0_selection_rule',accept==candidate['accepted'])
                if accept:chosen=candidate['routes'];best=c['cost']
            ck(key+':final_exact_selection',chosen==teacher['routes'] and best==teacher['report']['cost'])
            ck(key+':nonlearning_actual_deployment_retention',inserted(routes))
            if not mask:ck(key+':zero_mask_identity',teacher['routes']==initial and routes==initial)
            row={'request':request,'teacher':teacher,'partial':partial.tolist(),'deployment_routes':routes,'deployment_report':report,'remaining':remaining,'insertions':insertions,'status':status,'wall_s':done-t,'initial_report':initial_report,'source_qualified':True,'scientific_gain':False}
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());answers.append(row)
    end=own_closure();changed=[p for p,h in start.items() if end.get(p)!=h];extra=sorted(set(end)-set(start))
    ck('actual_environment_closed',not changed and not extra);save('SOURCE_END.json',{'files':end,'changed':changed,'new_files':extra})
    failed=sum(not c['pass'] for c in checks)
    save('RESULT.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'requests':len(answers),'candidates':sum(len(a['teacher']['all_candidates']) for a in answers),'source_files':len(start),'actual_server_teacher_boundary_qualified':failed==0,'scientific_training_qualified':False,'learning_trajectory_distribution_qualified':False,'dynamic_online_qualified':False,'stable_MaskCO_gain':False})
    print(json.dumps({'checks':len(checks),'failed':failed,'requests':len(answers)}));return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('FAILURE.json',{'traceback':traceback.format_exc(),'scientific_gain':False});raise
