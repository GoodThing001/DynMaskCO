"""New process replays every stored server candidate and actual empty-slot rules."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent/'server_teacher_boundary_root_v1';PRIOR=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';EXT=OLD/'extension'
for p in (SOURCE,PRIOR,OLD/'upstream',OLD/'actual_checkpoint_engineering_v1',EXT/'scripts/delivery',EXT/'scripts/coldchain',EXT/'scripts/research_20261010'):sys.path.insert(0,str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    import numpy as np,psutil
    from delivery_evaluator import evaluate
    from delivery_contract import DeliveryContract
    from full_gradient import closure
    from ortools.constraint_solver import pywrapcp
    psutil.Process().cpu_affinity([15]);pywrapcp.DefaultRoutingSearchParameters();ar=read(BASE/'AUDIT_REGISTRATION.json');reg=read(SOURCE/'ENGINEERING_REGISTRATION.json');raw=read(SOURCE/'RESULT.json');checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('independent_program_registered',sha(__file__)==ar['program_sha256'])
    for n,h in ar['inputs'].items():ck('bound_input:'+n,sha(SOURCE/n)==h)
    ck('producer_all4_completed',raw['failed']==0 and raw['requests']==4 and raw['actual_server_teacher_boundary_qualified'])
    ck('actual_original_states',sha(PRIOR/'STATES.jsonl')==reg['prior_states_sha256'])
    requests=read(SOURCE/'REQUESTS.json');rows=[json.loads(s) for s in (SOURCE/'ANSWERS.jsonl').read_text().splitlines()]
    ck('all4_order_retained',[r['request'] for r in rows]==requests and [q['key'] for q in requests]==['observed_second_H_state','derived_empty_slots_partial','derived_empty_slots_full_mask','derived_empty_slots_zero_mask'])
    prior=json.loads((PRIOR/'STATES.jsonl').read_text().splitlines()[1])['request'];ids=[0]+sorted(o for route in prior['initial'][:2] for o in route);mapping={o:i for i,o in enumerate(ids)}
    data={k:[v[i] for i in ids] for k,v in prior['data'].items()};initial=[[],[mapping[o] for o in prior['initial'][0]],[],[mapping[o] for o in prior['initial'][1]],[]]
    ck('observed_second_state_exact',requests[0]['data']==prior['data'] and requests[0]['initial']==prior['initial'] and requests[0]['mask']==prior['mask'])
    ck('actual18_derived_orders_and_declared_empty_slots',len(ids)-1==18 and all(q['data']==data and q['initial']==initial for q in requests[1:]))
    ck('declared_boundary_masks',[q['mask'] for q in requests[1:]]==[list(range(1,len(ids),2)),list(range(1,len(ids))),[]])
    start=read(SOURCE/'SOURCE_START.json');end=read(SOURCE/'SOURCE_END.json');ck('actual_original_closure_start_end',start['files']==end['files'] and not end['changed'] and not end['new_files'])
    for p,h in start['files'].items():ck('actual_source_start:'+p,sha(p)==h)
    def own_closure():
        out=closure()
        for p in (Path(__file__),BASE/'AUDIT_REGISTRATION.json'):out[str(p.resolve())]=sha(p)
        return out
    own_start=own_closure();save('AUDIT_SOURCE_START.json',{'files':own_start,'python':sys.version,'cpu':psutil.Process().cpu_affinity()});contract=DeliveryContract();ck('actual_contract_identity',contract.identity()==reg['contract_hash'])
    def adjacency(routes,n):
        a=np.zeros((n,n),np.int8)
        for r in routes:
            path=[0]+r+[0]
            for x,y in zip(path,path[1:]):
                if x!=y:a[x,y]=a[y,x]=1
        return a
    summaries=[]
    for row in rows:
        q=row['request'];key=q['key'];mask=set(q['mask']);n=len(q['data']['demands']);unmasked=set(range(1,n))-mask;partial=adjacency(q['initial'],n)
        for o in mask:partial[o,:]=0;partial[:,o]=0
        ck(key+':actual_partial',np.array_equal(partial,np.asarray(row['partial'])))
        initial_report=evaluate(q['data'],q['initial'],contract);ck(key+':complete_initial_report_exact',initial_report==row['initial_report']==row['teacher']['initial_report'])
        selected=q['initial'];best=initial_report['cost']
        for j,c in enumerate(row['teacher']['all_candidates']):
            label=key+':candidate:'+str(j);routes=c['routes'];report=evaluate(q['data'],routes,contract)
            ck(label+':independent_fullC0_exact',report==c['report'])
            ck(label+':all5_actual_vehicle_slots',len(routes)==contract.vehicles==5)
            ck(label+':all_orders_once',sorted(o for route in routes for o in route)==list(range(1,n)))
            locations={o:(v,i) for v,r in enumerate(routes) for i,o in enumerate(r)}
            ck(label+':same_unmasked_vehicle',all(locations[o][0]==v for v,r in enumerate(q['initial']) for o in r if o in unmasked))
            ck(label+':strict_unmasked_order',all(all(locations[b][1]>locations[a][1] for a,b in zip([o for o in r if o in unmasked],[o for o in r if o in unmasked][1:])) for r in q['initial']))
            ck(label+':retained_edges',bool(np.all(adjacency(routes,n)[partial==1]==1)))
            accept=report['hard'] and report['cost']<best-1e-12;ck(label+':fullC0_chosen',accept==c['accepted'])
            if accept:selected=routes;best=report['cost']
        ck(key+':selected_all_candidates_exact',selected==row['teacher']['routes'] and best==row['teacher']['report']['cost'])
        ck(key+':final_fullC0_exact',evaluate(q['data'],selected,contract)==row['teacher']['report'])
        ck(key+':actual_nonlearning_fullC0',evaluate(q['data'],row['deployment_routes'],contract)==row['deployment_report'] and row['deployment_report']['hard'] and row['deployment_report']['complete'] and row['remaining']==[])
        ck(key+':normal_request_in_registered_cap',0<row['wall_s']<reg['request_cap_s']==10.)
        if not mask:ck(key+':zero_mask_fixed_identity',selected==row['deployment_routes']==q['initial'])
        summaries.append({'key':key,'nodes':n,'initial_empty_slots':[i for i,r in enumerate(q['initial']) if not r],'candidates':len(row['teacher']['all_candidates']),'wall_s':row['wall_s'],'science':False})
    for p,h in start['files'].items():ck('actual_source_end:'+p,sha(p)==h)
    for n,h in ar['inputs'].items():ck('bound_input_end:'+n,sha(SOURCE/n)==h)
    own_end=own_closure();changed=[p for p,h in own_start.items() if own_end.get(p)!=h];extra=sorted(set(own_end)-set(own_start));ck('independent_actual_closure',not changed and not extra);save('AUDIT_SOURCE_END.json',{'files':own_end,'changed':changed,'new_files':extra})
    ck('honest_scope',not raw['scientific_training_qualified'] and not raw['learning_trajectory_distribution_qualified'] and not raw['dynamic_online_qualified'] and not raw['stable_MaskCO_gain'])
    failed=sum(not c['pass'] for c in checks);save('INDEPENDENT_SERVER_TEACHER_BOUNDARY.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'summaries':summaries,'source_files':len(own_start),'actual_server_teacher_boundary_qualified':failed==0,'scientific_training_qualified':False,'learning_distribution_qualified':False,'online_gain':False,'shared_physical_evaluator':True,'actual18_not20_derived_orders':True})
    print(json.dumps({'checks':len(checks),'failed':failed}));return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('AUDIT_FAILURE.json',{'traceback':traceback.format_exc()});raise
