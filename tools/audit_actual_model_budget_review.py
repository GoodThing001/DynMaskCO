"""Replay recorded snapshot candidates, clocks and pinned bytes; no reruns."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
R=ROOT/'research/review_3080_20261010/delivery'
B=R/'remote_model_budget_bundle_root_v1/downloaded'
for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(E/'scripts'/sub))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import generate,make_schedule

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def adjacency(routes,removed):
    a=np.zeros((51,51),np.float32)
    for route in routes:
        chain=[0]+route+[0]
        for u,v in zip(chain,chain[1:]):
            if u!=v and u not in removed and v not in removed:a[u,v]=a[v,u]=1.
    return a

def main():
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    def close(n,a,b):ck(n,math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8))
    result=load(B/'RESULT.json');regp=R/'model_budget_integration_root_v1/BUDGET_ENGINEERING_REGISTRATION.json';reg=load(regp)
    ck('registered_actual_bytes',sha(regp)==result['registration_sha256']=='3ba23fe74d0d63dd404281db7a6465818f516fd24b7092d476a0c54c0509428f')
    ck('actual_program_bytes',sha(regp.parent/'budget_qa.py')==reg['program_sha256'])
    ck('supervisor_exit0_within600',load(B/'RUN_EXIT.json')['exit_code']==0 and load(B/'RUN_EXIT.json')['wall_seconds']<600)
    ck('remote_end_source_independently_read',all(load(B/'ACTUAL_END_SOURCE_CHECK.json').values()))
    ck('actual_end_equals_registered_source',result['source_end']==reg['source_sha256'])
    for k,v in reg['source_sha256'].items():
        base,rel=k.split('/',1);ck('local_frozen:'+k,sha((ROOT/'MASKCO_code' if base=='upstream' else E)/rel)==v)
    teacherp=R/'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl'
    ck('actual_teacher_archive',sha(teacherp)==reg['teacher_archive_sha256'])
    teachers=[json.loads(x) for x in teacherp.read_text().splitlines()]
    c=DeliveryContract();records=result['records'];ck('14_unique_arm_fixture_pairs',len(records)==14 and len({(x['arm'],x['instance_id']) for x in records})==14)
    ck('exact_seven_arms',set(x['arm'] for x in records)==set(reg['arms']))
    ck('only_two_observed_engineering_fixtures',set(x['instance_id'] for x in records)=={900000,900001})
    weights=load(R/'remote_finite_training_bundle_v1/downloaded/INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json')
    for arm in ('reconstruction_control','paired_objective'):
        ck('actual_saved_weight_identity:'+arm,reg['engineering_weights'][arm]['sha256']==weights[arm+'_FIXED_FINAL.pkl']['sha256'] and weights[arm+'_FIXED_FINAL.pkl']['full_remote_bytes_read'] is True)
    costs={};candidate_count=0
    for arm in reg['arms']:
        ck(arm+':owned_worker_reaped',load(B/(arm+'_worker_end.json'))['reaped'] is True)
        costs[arm]=[]
    for r in records:
        tag=r['arm']+':'+str(r['instance_id'])+':'
        ck(tag+'downloaded_raw_equals_combined',load(B/(r['arm']+'_'+str(r['instance_id'])+'.json'))==r)
        data,initial=generate(202610110071,r['instance_id']);fixture=next(t for t in teachers if t['instance_id']==r['instance_id'] and t['mask_ordinal']==0)
        ck(tag+'independent_instance_regeneration',data==fixture['data'] and initial==fixture['incumbent_routes'])
        close(tag+'registered_deadline',r['deadline'],r['started']+10.)
        close(tag+'parent_clock_decomposition',r['wall_s'],r['submitted']-r['started'])
        ck(tag+'actual_total10s',0<=r['wall_s']<=10 and r['submitted']<=r['deadline'] and r['timeout'] is False and r['own_incumbent_fallback'] is False)
        ck(tag+'actual_worker_and_contract',r['worker_pid']==r['ready']['pid'] and r['ready']['contract_hash']==c.identity())
        a=r['actual_result'];ck(tag+'child_clock_inside_parent',r['started']<=a['worker_start']<=a['worker_end']<=r['submitted'] and a['request_deadline']==r['deadline'])
        ck(tag+'actual_child_no_timeout',a['timeout'] is False)
        replay=evaluate(data,a['routes'],c);ck(tag+'final_complete_hard',replay['hard'] is True and replay['complete'] is True)
        for key in ('cost','distance_km','energy_kwh','quality_loss_value'):close(tag+'final_'+key,replay[key],a['report'][key])
        ck(tag+'final_manifest_and_trace',replay['delivered']==a['report']['delivered'] and replay['traces']==a['report']['traces'])
        costs[r['arm']].append({'instance_id':r['instance_id'],'cost':replay['cost'],'parent_wall_s':r['wall_s']})
        if r['arm']=='gls':continue
        schedule=make_schedule(51,8,202610110072,r['instance_id'],fraction=.5)
        ck(tag+'actual_fixed8',a['planned_attempts']==a['actual_attempts']==a['complete_candidates']==a['hard_candidates']==8 and len(a['attempts'])==8)
        ck(tag+'actual_decode_count',a['decode_calls']==(0 if r['arm']=='nonlearning' else 8))
        current=initial
        for i,(row,mask) in enumerate(zip(a['attempts'],schedule),1):
            p=tag+'step'+str(i)+':'
            ck(p+'registered_mask',row['step']==i and row['removed']==mask and row['mask_sha256']==digest(mask))
            ck(p+'incumbent_recurrence',row['incumbent_before']==current)
            adj=adjacency(current,set(mask));ck(p+'independent_adjacency_bytes',row['adjacency_sha256']==hashlib.sha256(adj.tobytes()).hexdigest())
            ck(p+'retained_edge_list',row['retained_edges']==np.argwhere(np.triu(adj,1)>0).tolist())
            after=adjacency(row['candidate_routes'],set());ck(p+'all_unmasked_edges_preserved',np.all(after[adj>0]>0))
            before=evaluate(data,current,c);candidate=evaluate(data,row['candidate_routes'],c);candidate_count+=1
            ck(p+'full_candidate_hard',candidate['hard'] is True and candidate['complete'] is True and row['remaining']==[])
            for key in ('cost','distance_km','energy_kwh','quality_loss_value'):close(p+key,candidate[key],row['candidate_report'][key])
            ck(p+'candidate_manifest_and_trace',candidate['delivered']==row['candidate_report']['delivered'] and candidate['traces']==row['candidate_report']['traces'])
            close(p+'cost_before',before['cost'],row['cost_before'])
            accepted=candidate['cost']<before['cost'];ck(p+'strict_cost_acceptance',accepted==row['accepted'])
            current=row['candidate_routes'] if accepted else current
            ck(p+'actual_install',row['incumbent_after']==current)
            close(p+'cost_after',evaluate(data,current,c)['cost'],row['cost_after'])
        ck(tag+'final_recurrence',a['routes']==current)
    ck('no_online_or_research_upgrade',result['engineering_only'] is True and result['snapshot_only'] is True and result['formal_training_eligible'] is False and result['stable_online_gain'] is False)
    audit={'gate_name':'deployment_10s_budget','contract_hash':c.identity(),'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'checks':checks,
        'independent_acceptance':all(x['pass'] for x in checks),'all_fixed_attempts_complete_within_total10s':all(x['pass'] for x in checks),
        'candidate_routes_replayed':candidate_count,'final_routes_replayed':len(records),'costs_observed_engineering_only':costs,
        'common_ceiling_s':10,'equal_consumed_wall_time':False,'fixed8_is_anytime':False,
        'limits':['Two already observed snapshots; no fresh scientific cohort','Shared production C0 replay; not independent physical implementation','Whole weight hashes independently read; no independent per-call saved tensor reload','No timeout-injection regression in this exact seven-arm wrapper','Fixed8 opportunity control is not an equal-consumed-time or anytime comparison','GLS here is a snapshot baseline; no strong dynamic baseline qualification'],
        'formal_training_authorized_by_this_audit':False,'dynamic_online_qualified':False,'stable_core_online_gain':False}
    with (B/'INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':audit['failed'],'candidates':candidate_count,'final_routes':len(records),'costs':costs}));return int(audit['failed']>0)
if __name__=='__main__':sys.exit(main())
