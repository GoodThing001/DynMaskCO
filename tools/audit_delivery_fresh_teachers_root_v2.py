"""Complete256 state replay and labels before fixed final training; never collect."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
R=ROOT/'research/review_3080_20261010/delivery';SOURCE=R/'finite_training_research_root_v2'
B=R/'remote_finite_research_root_v2/teacher_downloaded'
for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(E/'scripts'/sub))
from delivery_contract import DeliveryContract
from delivery_equal_schedule import generate,make_schedule
from delivery_evaluator import evaluate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def adjacency(routes,removed):
    a=np.zeros((51,51),np.float32)
    for route in routes:
        chain=[0]+route+[0]
        for u,v in zip(chain,chain[1:]):
            if u!=v and u not in removed and v not in removed:a[u,v]=a[v,u]=1
    return a
def main():
    reg=load(SOURCE/'RESEARCH_REGISTRATION.json');summary=load(B/'TEACHER_SUMMARY.json')
    archive=B/'TEACHER_STATES.jsonl';rows=[json.loads(s) for s in archive.read_text().splitlines()]
    c=DeliveryContract();checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    def close(n,x,y):ck(n,math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8))
    ck('registered_before_generation_bytes',sha(SOURCE/'RESEARCH_REGISTRATION.json')==summary['registration_sha256']=='1f19554a2790d258b7fca4f081aeae38b7250acab817142b08825cf91ab18af1')
    ck('actual_outer_exit0',load(B/'teacher_RUN_EXIT.json')['exit_code']==0)
    ck('actual_end_independent_source_read',all(load(B/'INDEPENDENT_ACTUAL_END_SOURCE_CHECK.json').values()))
    ck('whole_archive_identity',sha(archive)==summary['teacher_archive_sha256'])
    ck('all256_unique_keys_no_screening',len(rows)==256 and {tuple(x['state_key']) for x in rows}=={(i,j) for i in reg['train_instance_ids'] for j in range(4)})
    ck('actual_same_contract',summary['contract_hash']==reg['contract_hash']==c.identity())
    ck('new_registered_seeds',summary['dataset_seed']==reg['seeds']['train'] and summary['mask_seed']==reg['seeds']['teacher_mask'])
    ck('64_instance_summaries',len(summary['instances'])==64 and [x['instance_id'] for x in summary['instances']]==list(range(64)))
    for inst in summary['instances']:
        ck('teacher_all_generation_within10:'+str(inst['instance_id']),inst['states']==4 and 0<=inst['wall_s']<=10)
    categories=dict.fromkeys(('positive','negative','small','invalid'),0)
    for row in rows:
        p=str(row['state_key'])+':'
        data,initial=generate(reg['seeds']['train'],row['instance_id'])
        mask=make_schedule(51,4,reg['seeds']['teacher_mask'],row['instance_id'],.5)[row['mask_ordinal']]
        ck(p+'actual_record_hash',row['state_sha256']==digest({k:v for k,v in row.items() if k!='state_sha256'}))
        ck(p+'regenerated_full_data',data==row['data'] and digest(data)==row['data_sha256'])
        ck(p+'registered_original_routes_and_mask',initial==row['incumbent_routes'] and mask==row['mask'])
        adj=adjacency(initial,set(mask));ck(p+'independent_original_unmasked_adjacency',np.array_equal(adj,np.asarray(row['partial_adjacency'],np.float32)))
        before=evaluate(data,initial,c);candidate=evaluate(data,row['candidate_routes'],c)
        ck(p+'initial_complete_hard',before['hard'] is True and before['complete'] is True and row['initial_report']['hard'] is True)
        for key in ('cost','distance_km','energy_kwh','quality_loss_value'):
            close(p+'initial_'+key,before[key],row['initial_report'][key])
        ck(p+'original_manifest_and_trace',before['delivered']==row['initial_report']['delivered'] and before['traces']==row['initial_report']['traces'])
        preserved=np.all(adjacency(row['candidate_routes'],set())[adj>0]>0)
        ck(p+'retained_edge_boolean',bool(preserved)==row['retained_edges_preserved'])
        if row['candidate_report']['hard'] is True:
            ck(p+'claimed_hard_candidate_replay',candidate['hard'] is True and candidate['complete'] is True)
            for key in ('cost','distance_km','energy_kwh','quality_loss_value'):close(p+'candidate_'+key,candidate[key],row['candidate_report'][key])
            ck(p+'candidate_manifest_and_trace',candidate['delivered']==row['candidate_report']['delivered'] and candidate['traces']==row['candidate_report']['traces'])
        # Invalid or late candidates remain invalid even if an after-budget replay is hard.
        valid=row['candidate_report']['hard'] is True and row['candidate_report']['complete'] is True and row['status']=='complete' and not row['late']
        delta=before['cost']-candidate['cost'] if valid else None
        sign=0 if delta is None or abs(delta)<=1e-8 else (1 if delta>0 else -1)
        category='invalid' if not valid else ('small' if sign==0 else 'positive' if sign>0 else 'negative')
        categories[category]+=1
        ck(p+'registered_pair_validity',valid==row['teacher_pair_valid'])
        ck(p+'independent_label_and_category',type(row['preference_sign']) is int and sign==row['preference_sign'] and category==row['category'])
        target=row['candidate_routes'] if valid and delta>1e-8 else initial
        ck(p+'full_C0_reconstruction_target',target==row['CE_target_routes'])
        ce=evaluate(data,target,c);ck(p+'CE_full_complete_hard',ce['hard'] is True and ce['complete'] is True and row['CE_target_full_hard'] is True)
        close(p+'CE_cost',ce['cost'],row['CE_target_certified_report']['cost'])
        ck(p+'retain_invalid_negative_small',row['all_states_retained'] is True)
        ck(p+'local_teacher_ceiling_or_retained_late',row['late'] is True or 0<=row['wall_s']<=2.5)
    for category in categories:
        ck('category_summary:'+category,categories[category]==sum(x['categories'][category] for x in summary['instances']))
    audit={'gate_name':'fresh256_teacher_acceptance','independent_acceptance':all(x['pass'] for x in checks),
        'registration_sha256':sha(SOURCE/'RESEARCH_REGISTRATION.json'),'teacher_archive_sha256':sha(archive),
        'states':len(rows),'unique_instances':64,'categories':categories,'contract_hash':c.identity(),
        'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'checks':checks,
        'limits':['Frozen C0 production evaluator replay; not independent physical implementation','Single masked regret2 proposals; not optimal teacher or deployment gain','All negative/small/invalid retained; no filtered dataset'],
        'training_stage_may_start':all(x['pass'] for x in checks),'confirmation_may_start':False,'dynamic_online_gain':False}
    with (B/'ACCEPTED_TEACHER_STATES.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2,allow_nan=False)
    print(json.dumps({k:audit[k] for k in ('states','categories','checks_count','failed','training_stage_may_start')}))
    return int(audit['failed']>0)
if __name__=='__main__':sys.exit(main())
