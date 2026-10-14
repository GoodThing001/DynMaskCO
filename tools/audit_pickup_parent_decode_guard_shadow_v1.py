"""Read-only independent capsule arithmetic, stage evidence and loaded file identities."""
import json,hashlib
from pathlib import Path
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=old.ROOT;P=old.P;OUT=P/'parent_decode_guard_shadow_root_v1';SOURCE=P/'observed_day_extension_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ends(p):
    first=last=None
    with p.open(encoding='utf-8') as f:
        for line in f:
            if first is None:first=line
            last=line
    return [json.loads(first),json.loads(last)]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');_,decode=old.codecs();checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    packets=[decode(x) for x in ends(SOURCE/'REQUESTS_STARTED.jsonl')];responses=[decode(x) for x in ends(SOURCE/'PREPARED_RESPONSES.jsonl')]
    expected={}
    for packet,response in zip(packets,responses):
        q=response['response']['compiled_state'];canonical=dict(version='c0-reserve-v1',budget=q.budget,mode=q.mode,accepted=sorted(q.accepted),rejected=sorted(q.rejected),plan={str(k):list(v) for k,v in q.plan.items()},status_inst=q.status_inst)
        expected[packet['callback']]=(digest(canonical),digest(packet['packet']['snapshot']['replanner_state']))
        ck(str(packet['callback'])+':original_sealed_compile_digest',expected[packet['callback']][0]==response['response']['compiled_state_sha256'])
    ck('fixed_original_endpoint_selection',sorted(expected)==[15,80] and len(raw['records'])==8)
    ck('registered_components_only_not_science',not reg['new_C0_or_SAA_execution'] and not reg['new_scientific_data'] and not reg['full_live_request_qualified'])
    for p,h in reg['sources'].items():ck('registered:'+p,sha(p)==h)
    ck('exact_registered_cases',[(r['case'],r['callback'],r['fault']) for r in raw['records']]==[(0,15,0),(1,80,0)]+[(1+n,15,n) for n in range(1,7)])
    for record in raw['records']:
        n=str(record['case']);exp,prior=expected[record['callback']]
        ck(n+':independent_capsule_identity',record['expected_capsule']==exp and record['prior_capsule']==prior)
        ck(n+':actual_time_arithmetic',record['finished']-record['started']==record['wall_s'] and record['wall_s']<record['cap_s'])
        ck(n+':registered_reserved_budget',record['cap_s']==(.4 if record['fault'] else 10.) and record['reserve_s']==(.15 if record['fault'] else 2.))
        ck(n+':owned_worker_reaped',record['reaped'] and record['pid']==record['ready']['pid'])
        ck(n+':no_real_owner_promotion',not record['actual_simulator_owner_promoted'] and not record['timed_parent_pickle_decode'])
        begin=read(OUT/('SOURCE_START_'+n+'.json'))
        ck(n+':loaded_start_process_identity',begin['pid']==record['pid'] and len(begin['files'])==record['ready']['loaded_files'])
        for p,h in begin['files'].items():ck(n+':start_hash:'+p,sha(p)==h)
        ck(n+':loaded_modules_all_backed',set(begin['module_files'].values())<=set(begin['files']))
        if record['fault']:
            ck(n+':stage_reached_and_cancelled',record['stage_reached']==record['fault'] and record['error']=='cancelled')
            ck(n+':caller_capsule_still_prior',record['prepared'] is None and record['host_capsule']==prior)
            ck(n+':no_returned_result_after_cancel',not (OUT/('REPLICA_RESULT_'+n+'.json')).exists())
        else:
            end=read(OUT/('SOURCE_END_'+n+'.json'));result=read(OUT/('REPLICA_RESULT_'+n+'.json'))
            ck(n+':normal_exact_capsule',record['prepared']==record['host_capsule']==exp and record['error'] is None and record['stage_reached']==6)
            ck(n+':actual_shadow_guard_pointer',result['installed'] and result['reason']=='published' and result['compiled_sha256']==exp and not result['actual_simulator_owner_promoted'])
            ck(n+':loaded_source_native_closure',begin['files']==end['files'] and not result['source_changed'] and not result['new_loaded_files'] and result['loaded_closure_qualified'])
            ck(n+':matching_loaded_module_sets',begin['module_files']==end['module_files'])
    ck('full_live_or_full_pipeline_closure_not_upgraded',not raw['live_owner_handoff_qualified'] and not raw['parent_public_projection_encode_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['full_live_request_qualified'] and not raw['stable_online_gain'])
    failed=[c['name'] for c in checks if not c['pass_']]
    result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,registered_sha256=sha(OUT/'REGISTRATION.json'),producer_sha256=sha(OUT/'RESULT.json'),independent_without_shadow_guard_or_decoder_reexecution=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_PARENT_SHADOW.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed)));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
