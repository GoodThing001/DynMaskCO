"""Separate process acceptance of complete fresh family2 raw data; no search/train."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent/'multistep_insertable_family2_root_v1';PRIOR=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';EXT=OLD/'extension'
for p in (SOURCE,PRIOR,OLD/'upstream',OLD/'actual_checkpoint_engineering_v1',EXT/'scripts/delivery',EXT/'scripts/coldchain',EXT/'scripts/research_20261010'):sys.path.insert(0,str(p))
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
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from full_gradient import closure
    from ortools.constraint_solver import pywrapcp
    psutil.Process().cpu_affinity([15]);pywrapcp.DefaultRoutingSearchParameters();ar=read(BASE/'AUDIT_REGISTRATION.json');reg=read(SOURCE/'RESEARCH_REGISTRATION.json');producer=read(SOURCE/'RESULT.json');checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('registered_independent_program',sha(__file__)==ar['program_sha256'])
    for n,h in ar['inputs'].items():ck('bound_start:'+n,sha(SOURCE/n)==h)
    ck('external_registration_exact',sha(SOURCE/'RESEARCH_REGISTRATION.json')==read(SOURCE/'EXTERNAL_REGISTRATION_BINDING.json')['registration_sha256']==ar['research_registration_sha256'])
    ck('producer_registered_barrier_passed',read(SOURCE/'RUN_EXIT.json')['exit_code']==0 and producer['failed']==0)
    source_start=read(SOURCE/'SOURCE_START.json');source_end=read(SOURCE/'SOURCE_END.json');ck('producer_actual_closure',source_start['files']==source_end['files'] and not source_end['changed'] and not source_end['new_files'])
    for p,h in source_start['files'].items():ck('actual_source_start:'+p,sha(p)==h)
    closure()
    def own_closure():
        out=closure()
        for p in (Path(__file__),BASE/'AUDIT_REGISTRATION.json'):out[str(p.resolve())]=sha(p)
        return out
    audit_start=own_closure();save('AUDIT_SOURCE_START.json',{'files':audit_start,'python':sys.version,'cpu':[15]})
    started=[json.loads(s) for s in (SOURCE/'REQUESTS_STARTED.jsonl').read_text(encoding='utf-8').splitlines()]
    # Full candidate reports can exceed1GiB. Stream one state; never materialize all raw rows.
    raw_file=(SOURCE/'STATES.jsonl').open(encoding='utf-8');rows=(json.loads(s) for s in raw_file);observed_keys=[]
    keys=[(i,s) for i in range(64) for s in range(4)]
    ck('all256_started_requests_retained',len(started)==256 and [(r['instance_id'],r['step']) for r in started]==keys)
    contract=DeliveryContract();ck('actual_contract',contract.identity()==reg['contract_sha256'])
    def adjacency(routes):
        out=np.zeros((51,51),np.int8)
        for route in routes:
            path=[0]+route+[0]
            for a,b in zip(path,path[1:]):
                if a!=b:out[a,b]=out[b,a]=1
        return out
    def structural(routes,initial,mask,partial):
        if len(routes)!=5 or sorted(o for r in routes for o in r)!=list(range(1,51)):return False
        locations={o:(v,k) for v,r in enumerate(routes) for k,o in enumerate(r)};kept=set(range(1,51))-set(mask)
        return all(all(locations[o][0]==v for o in r if o in kept) and all(locations[b][1]>locations[a][1] for a,b in zip([o for o in r if o in kept],[o for o in r if o in kept][1:])) for v,r in enumerate(initial)) and bool(np.all(adjacency(routes)[partial==1]==1))
    summaries=[];candidate_count=0;categories={1:0,-1:0,0:0};next_routes={};expected_data={};masks={}
    mismatch_counts={'exact_route_false':0,'same_adjacency_different_route':0,'different_adjacency':0,'oracle_better_fullC0_than_teacher':0,'oracle_equal_fullC0_to_teacher':0,'oracle_worse_fullC0_than_teacher':0,'oracle_incomplete':0,'H_incomplete':0}
    for i in range(64):
        rng=np.random.default_rng(np.random.SeedSequence([reg['seeds']['train'],i]))
        expected_data[i]={'coords':np.r_[np.array([[5.,5.]]),rng.uniform(0,10,(50,2))].tolist(),'demands':np.r_[0,rng.integers(1,6,50)].tolist(),'temp_class':np.r_[0,rng.integers(0,3,50)].tolist(),'reveal_time':[0.]*51,'ready_time':[0.]*51,'tw_start':[0.]*51,'tw_end':[8.]*51,'service_time':[0.]+[.03]*50}
        mrng=np.random.default_rng(np.random.SeedSequence([reg['seeds']['teacher_mask'],i]));masks[i]=[sorted(map(int,mrng.choice(np.arange(1,51),25,replace=False))) for _ in range(4)]
        next_routes[i]=[list(range(1+10*k,11+10*k)) for k in range(5)]
    for index,row in enumerate(rows):
        i,s=row['instance_id'],row['step'];observed_keys.append((i,s));key=str((i,s));q=row['request'];data,initial,mask=q['data'],q['initial'],q['mask']
        ck(key+':fixed_fresh_dataset_exact',data==expected_data[i]);ck(key+':fixed_mask_exact',mask==masks[i][s]);ck(key+':exact_H_driven_state',initial==next_routes[i])
        ck(key+':started_input_preserved',index<len(started) and q=={k:started[index][k] for k in ('data','initial','mask')})
        part=adjacency(initial)
        for o in mask:part[o,:]=0;part[:,o]=0
        ck(key+':actual_partial',np.array_equal(part,row['partial']));before=evaluate(data,initial,contract)
        ck(key+':initial_fullC0_exact',before==row['initial_report']==row['teacher']['initial_report'] and before['hard'] and before['complete'])
        selected=initial;best=before['cost']
        for j,c in enumerate(row['teacher']['all_candidates']):
            name=key+':candidate:'+str(j);report=evaluate(data,c['routes'],contract);candidate_count+=1
            ck(name+':fullC0_exact',report==c['report']);ck(name+':insertable_action',structural(c['routes'],initial,mask,part))
            accept=report['hard'] and report['cost']<best-1e-12;ck(name+':strict_fullC0_choice',accept==c['accepted'])
            if accept:selected=c['routes'];best=report['cost']
        ck(key+':teacher_full_selection',selected==row['teacher']['routes'] and best==row['teacher']['report']['cost'] and evaluate(data,selected,contract)==row['teacher']['report'])
        delta=before['cost']-best;lab=0 if abs(delta)<=1e-8 else (1 if delta>0 else -1);categories[lab]+=1
        ck(key+':label_fullC0_not_distance',row['label']==lab and row['delta_cost']==delta)
        for name in ('deployment','oracle'):
            block=row[name];routes=block['routes'];report=evaluate(data,routes,contract)
            if block['remaining']:
                ck(key+':'+name+':incomplete_original_retained',block['report'] is None and not report['complete'] and len(set(block['remaining']))==len(block['remaining']) and set(block['remaining'])<=set(mask))
                if name=='deployment':mismatch_counts['H_incomplete']+=1;ck(key+':H_complete_legal',False)
                else:mismatch_counts['oracle_incomplete']+=1
            else:
                ck(key+':'+name+':actual_fullC0',report==block['report'] and report['hard'] and report['complete'])
                ck(key+':'+name+':insertable_action',structural(routes,initial,mask,part))
            replay=[[o for o in r if o not in set(mask)] for r in initial];orders=[]
            for event in block['insertions']:
                customer,vehicle,at=event['customer'],event['vehicle'],event['position'];orders.append(customer)
                if not (type(customer)is int and type(vehicle)is int and type(at)is int and 0<=vehicle<5 and 0<=at<=len(replay[vehicle])):raise ValueError('Recorded insertion boundary')
                replay[vehicle].insert(at,customer)
            ck(key+':'+name+':complete_insertion_log',sorted(orders+block['remaining'])==mask and len(set(orders+block['remaining']))==len(mask) and replay==routes)
        d=row['deployment'];accept=not row['timed_out'] and not d['remaining'] and d['report'] is not None and d['report']['hard'] and d['report']['complete'] and d['report']['cost']<before['cost']
        ck(key+':H_next_selected_without_teacher',d['accepted']==accept and d['next_routes']==(d['routes'] if accept else initial) and row['trajectory_driver']=='nonlearning_H_not_teacher_or_oracle');next_routes[i]=d['next_routes']
        ck(key+':exact_ideal_target_reachability',row['oracle']['routes']==selected and row['oracle']['exact_target'] is True)
        ck(key+':actual_request_cap',not row['timed_out'] and 0<row['wall_s']<10)
        oracle_exact=row['oracle']['routes']==selected;same_adj=bool(np.array_equal(adjacency(row['oracle']['routes']),adjacency(selected)));oracle_gap=row['oracle']['report']['cost']-best if row['oracle']['report'] is not None else None
        if not oracle_exact:
            mismatch_counts['exact_route_false']+=1;mismatch_counts['same_adjacency_different_route' if same_adj else 'different_adjacency']+=1
            if oracle_gap is not None:mismatch_counts['oracle_equal_fullC0_to_teacher' if abs(oracle_gap)<=1e-8 else ('oracle_better_fullC0_than_teacher' if oracle_gap<0 else 'oracle_worse_fullC0_than_teacher')]+=1
        summaries.append({'key':[i,s],'label':lab,'candidates':len(row['teacher']['all_candidates']),'H_accepted':accept,'oracle_exact':oracle_exact,'oracle_same_undirected_adjacency':same_adj,'oracle_minus_teacher_fullC0':oracle_gap,'oracle_fullC0':row['oracle']['report']['cost'] if row['oracle']['report'] is not None else None,'teacher_fullC0':best,'initial_fullC0':before['cost']})
    raw_file.close();ck('all256_unique_ordered_keys',observed_keys==keys)
    ck('all_candidate_counts_retained',candidate_count==producer['counts']['teacher_candidates'])
    for p,h in source_start['files'].items():ck('actual_source_end:'+p,sha(p)==h)
    for n,h in ar['inputs'].items():ck('bound_end:'+n,sha(SOURCE/n)==h)
    audit_end=own_closure();changed=[p for p,h in audit_start.items() if audit_end.get(p)!=h];extra=sorted(set(audit_end)-set(audit_start));ck('independent_actual_closure',not changed and not extra);save('AUDIT_SOURCE_END.json',{'files':audit_end,'changed':changed,'new_files':extra})
    failed=sum(not c['pass'] for c in checks)
    failed_names=[c['name'] for c in checks if not c['pass']]
    barrier_names=[n for n in failed_names if n=='producer_registered_barrier_passed' or n.endswith(':exact_ideal_target_reachability') or n.endswith(':H_complete_legal')]
    save('INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'registered_barrier_failures':barrier_names,'raw_replay_failure_names':[n for n in failed_names if n not in barrier_names],'original_producer_failed':producer['failed'],'summaries':summaries,'counts':{'states':len(observed_keys),'candidates':candidate_count,'positive':categories[1],'negative':categories[-1],'small':categories[0]},'posthoc_descriptive_mismatch_diagnosis':mismatch_counts,'posthoc_diagnosis_does_not_change_registered_exact_target_barrier':True,'independent_instances':64,'source_files':len(audit_start),'scientific_teacher_data_qualified':failed==0,'actual_training_qualified':False,'training_execution_enabled':False,'learning_policy_distribution_qualified':False,'stable_snapshot_gain':False,'stable_online_gain':False,'shared_physical_evaluator':True,'teacher_search_rerun':False,'new_optimizer_steps':0,'specific_binary_probe_failure_not_general_unreachability_proof':True})
    print(json.dumps({'checks':len(checks),'failed':failed,'candidates':candidate_count}));return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('AUDIT_FAILURE.json',{'traceback':traceback.format_exc(),'training_execution_enabled':False});raise
