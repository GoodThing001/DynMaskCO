"""Independent regeneration, recurrence, raw C0 replay and descriptive paired DEV estimates."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
D=ROOT/'research/review_3080_20261010/delivery';LOCAL=D/'remote_fresh_development_root_v1'
B=LOCAL/'downloaded';REG=D/'fresh_development_root_v1/DEVELOPMENT_REGISTRATION.json'
for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(E/'scripts'/sub))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import generate,make_schedule
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def adjacency(routes,removed):
    a=np.zeros((51,51),np.float32)
    for route in routes:
        sequence=[0]+route+[0]
        for u,v in zip(sequence,sequence[1:]):
            if u!=v and u not in removed and v not in removed:a[u,v]=a[v,u]=1.
    return a
def main():
    checks=[]
    def ck(name,v):checks.append({'name':name,'pass':bool(v)})
    def close(name,a,b):ck(name,math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8))
    result=read(B/'RESULT.json');reg=read(REG);c=DeliveryContract()
    ck('registered_before_actual_generation',sha(REG)==result['registration_sha256'])
    ck('actual_program_upload_and_independent_read',read(LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json')['source_sha256']['development.py']==reg['program_sha256'] and read(B/'ACTUAL_END_SOURCE_CHECK.json')['development.py'])
    ck('local_execution_source_same_bytes',sha(REG.parent/'development.py')==reg['program_sha256'])
    ck('all_actual_frozen_sources_unchanged',all(read(B/'ACTUAL_END_SOURCE_CHECK.json').values()))
    ck('own_supervisor_exit0_bound',read(B/'RUN_EXIT.json')['exit_code']==0 and read(B/'RUN_EXIT.json')['wall_seconds']<1800)
    for key,expected in reg['source_sha256'].items():
        group,name=key.split('/',1);ck('local_source:'+key,sha((ROOT/'MASKCO_code' if group=='upstream' else E)/name)==expected)
    ck('actual_end_source_identity',result['source_end']==reg['source_sha256'])
    training=D/'remote_finite_research_root_v2/training_downloaded/ACCEPTED_FIXED_TRAINING.json'
    budget=D/'remote_final_model_budget_bundle_root_v3/downloaded/ACCEPTED_FINAL_MODEL_BUDGET.json'
    ck('exact_independent_final128_acceptance',sha(training)==reg['actual_fixed_training_acceptance_sha256'] and read(training)['failed']==0)
    ck('exact_independent_parent_worker_budget_acceptance',sha(budget)==reg['actual_budget_acceptance_sha256'] and read(budget)['independent_acceptance'])
    science=D/'finite_training_research_root_v2/RESEARCH_REGISTRATION.json'
    ck('same_science_registration',sha(science)==reg['science_registration_sha256'])
    ck('six_registered_arm_mapping',set(reg['arm_names'].values())==set(read(science)['deployment_arms']))
    ck('registered_fresh16',reg['data_seed']==read(science)['seeds']['development'] and reg['instance_ids']==list(range(16)))
    rng=np.random.default_rng(reg['arm_order_seed']);base=list(rng.permutation(reg['arms']));orders=[]
    for block in range(3):
        for shift in map(int,rng.permutation(6)):orders.append(base[shift:]+base[:shift])
    ck('independent_balanced_arm_order_regeneration',orders[:16]==reg['balanced_arm_order'])
    for arm in reg['arms']:
        counts=[sum(row[j]==arm for row in orders[:16]) for j in range(6)]
        ck('positions_balanced:'+arm,max(counts)-min(counts)<=1)
    ck('96_unique_complete_arm_instance_pairs',len(result['records'])==96 and len({(r['arm'],r['instance_id']) for r in result['records']})==96)
    ck('registered_actual_trial_sequence',[(r['instance_id'],r['arm']) for r in result['records']]==[(i,arm) for i,order in enumerate(orders[:16]) for arm in order])
    ck('all_owned_workers_reaped',all(row['reaped'] for row in read(B/'WORKERS_REAPED.json')))
    costs={arm:[] for arm in reg['arms']};timeouts={arm:0 for arm in reg['arms']};walls={arm:[] for arm in reg['arms']};candidate_count=0
    for r in result['records']:
        tag=r['arm']+':'+str(r['instance_id'])+':'
        data,initial=generate(reg['data_seed'],r['instance_id']);fixture=read(B/('fixture_'+str(r['instance_id'])+'.json'))
        ck(tag+'independent_instance_data',fixture['data']==data and fixture['incumbent_routes']==initial)
        ck(tag+'raw_identical_combined',read(B/(r['arm']+'_'+str(r['instance_id'])+'.json'))==r)
        ck(tag+'charged_parent_and_worker_CPU15',r['actual_charged_parent_cpu_affinity']==[15] and r['ready']['actual_cpu_affinity']==[15])
        ck(tag+'actual_contract',r['ready']['contract_hash']==c.identity())
        if r['arm'] not in ('nonlearning','gls'):
            provider=r['ready']['provider']
            ck(tag+'full10D_original_weight_and_dtype',provider['original_checkpoint_sha256']==reg['original_checkpoint_sha256'] and provider['encoder_input_dim']==10 and provider['actual_dtype']=='float32' and provider['complete_encoder_decoder'] is True)
            ck(tag+'actual_random_initialization_seed',provider['random_seed']==(42 if r['arm']=='random' else None))
        close(tag+'10s_absolute_deadline',r['deadline'],r['started']+10.)
        close(tag+'parent_wall',r['wall_s'],r['submitted']-r['started'])
        walls[r['arm']].append(r['wall_s']);timeouts[r['arm']]+=int(r['timeout'])
        a=r['actual_result'];final=evaluate(data,r['final_routes'],c)
        ck(tag+'final_full_hard',final['hard'] is True and final['complete'] is True)
        for key in ('cost','distance_km','energy_kwh','quality_loss_value'):close(tag+'final_'+key,final[key],r['final_report'][key])
        ck(tag+'manifest_trace_exact',final['traces']==r['final_report']['traces'] and final['delivered']==r['final_report']['delivered'])
        if r['own_incumbent_fallback']:
            ck(tag+'timeout_no_screening_fallback',r['timeout'] and r['final_routes']==initial)
        else:
            ck(tag+'submission_inside_cap',r['submitted']<=r['deadline'] and r['wall_s']<=10 and a is not None and a['timeout'] is False)
            ck(tag+'inside_child_clock',r['started']<=a['worker_start']<=a['worker_end']<=r['submitted'])
            ck(tag+'accepted_routes_are_child_result',r['final_routes']==a['routes'])
        if r['arm'] in ('paired_objective','reconstruction_control'):
            ck(tag+'actual_fixed_final_sha',r['ready']['provider']['actual_saved_weight_sha256']==reg['fixed_final_weights'][r['arm']]['sha256'] and r['ready']['provider']['research_model'] is True)
        if a is not None and r['arm']!='gls':
            schedule=make_schedule(51,8,reg['mask_seed'],r['instance_id'],fraction=.5);current=initial
            ck(tag+'all8_planned_rows',a['planned_attempts']==8 and len(a['attempts'])==8)
            if not r['timeout']:
                ck(tag+'eight_actual_calls_and_complete_candidates',a['actual_attempts']==a['complete_candidates']==a['hard_candidates']==8 and a['decode_calls']==(0 if r['arm']=='nonlearning' else 8))
            for j,(row,mask) in enumerate(zip(a['attempts'],schedule),1):
                label=tag+str(j)+':'
                ck(label+'same_registered_mask',row['step']==j and row['removed']==mask and row['mask_sha256']==digest(mask))
                ck(label+'incumbent_before',row['incumbent_before']==current)
                adj=adjacency(current,set(mask));ck(label+'independent_adj_bytes',row['adjacency_sha256']==hashlib.sha256(adj.tobytes()).hexdigest())
                before=evaluate(data,current,c);close(label+'before_cost',before['cost'],row['cost_before'])
                if row['candidate_routes'] is not None and not row['remaining']:
                    candidate=evaluate(data,row['candidate_routes'],c);candidate_count+=1
                    ck(label+'candidate_complete_hard',candidate['hard'] and candidate['complete'])
                    ck(label+'preserved_edges',np.all(adjacency(row['candidate_routes'],set())[adj>0]>0))
                    for key in ('cost','distance_km','energy_kwh','quality_loss_value'):close(label+key,candidate[key],row['candidate_report'][key])
                    ck(label+'candidate_trace',candidate['traces']==row['candidate_report']['traces'])
                    if row['accepted']:
                        ck(label+'strict_improvement',candidate['cost']<before['cost']);current=row['candidate_routes']
                else:ck(label+'unfinished_never_accepted',row['accepted'] is False)
                ck(label+'incumbent_after',row['incumbent_after']==current)
                close(label+'after_cost',evaluate(data,current,c)['cost'],row['cost_after'])
            if not r['timeout']:ck(tag+'final_recurrence',a['routes']==current)
        costs[r['arm']].append((r['instance_id'],final['cost']))
    arrays={arm:np.asarray([v for i,v in sorted(rows)]) for arm,rows in costs.items()}
    paired={};rng=np.random.default_rng(read(science)['seeds']['bootstrap']);index=rng.integers(0,16,(100000,16))
    for control in ('random','nonlearning','gls','reconstruction_control'):
        delta=(arrays[control]-arrays['paired_objective'])/arrays[control]
        ci=np.quantile(delta[index].mean(1),[.025,.975])
        paired[control]={'mean_relative_gain':float(delta.mean()),'descriptive95_CI':ci.tolist(),'raw_paired_relative_gains':delta.tolist(),
          'registered_target_mean':0. if control=='reconstruction_control' else .01,'formal_confirmation_gate':False}
    report={'registered_scope':'16fresh_snapshot_development_diagnostic','checks_count':len(checks),
      'failed':sum(not x['pass'] for x in checks),'checks':checks,'independent_acceptance':all(x['pass'] for x in checks),
      'candidate_routes_replayed':candidate_count,'final_routes_replayed':96,'all96_final_hard':all(x['pass'] for x in checks if x['name'].endswith('final_full_hard')),
      'mean_costs':{arm:float(values.mean()) for arm,values in arrays.items()},'costs_by_instance':{arm:values.tolist() for arm,values in arrays.items()},
      'timeouts_all_counted':timeouts,'max_actual_parent_wall':{arm:max(values) for arm,values in walls.items()},
      'paired_descriptive':paired,'descriptive_bootstrap_repetitions':100000,'confirmation_adjudicated':False,
      'checkpoint_or_hyperparameter_selection':False,'stable_snapshot_gain':False,'MaskCO_core_online_gain':False,
      'limits':['New16 diagnostics only; not registered3x40 confirmation','Fixed8 opportunity/control ceiling10s not equal consumed time',
      'Distance proposing GLS does not prove C0 optimality','C0 production replay shared; physical parameters uncalibrated',
      'Random untrained control cannot isolate pretrained initialization versus matched random-trained models']}
    with (B/'INDEPENDENT_DEVELOPMENT_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({key:report[key] for key in ('failed','checks_count','mean_costs','timeouts_all_counted','paired_descriptive')},allow_nan=False))
    return int(report['failed']>0)
if __name__=='__main__':sys.exit(main())
