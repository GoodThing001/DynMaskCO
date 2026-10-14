"""Additional stored-fault closure and late-selection evidence; no solver rerun."""
import copy,json,hashlib,psutil
from pathlib import Path
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=old.ROOT;OUT=old.P/'endogenous_owner_transaction_root_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    normal=decode(read(OUT/'CANDIDATE_RESULT_0.json'));fault=decode(read(OUT/'CANDIDATE_RESULT_1.json'));raw=read(OUT/'RESULT.json');r=raw['records'][1]
    for k in ('public_packet','public_digest','compiled_sha256','after_commit'):
        ck('same_registered_actual_candidate:'+k,old.pf.digest(normal[k])==old.pf.digest(fault[k]))
    for k in ('base_response','scoring_packet','proposal','common_scores','selected_certificate','selected_raw','compiled_state','compiled_state_sha256','accepted_route'):
        ck('stored_fault_actual_endogenous_response:'+k,old.pf.digest(normal['response'][k])==old.pf.digest(fault['response'][k]))
    ck('fault_selected_then_discarded_private_branch',r['prepared']==fault['compiled_sha256'] and r['promotion'] is not None and r['stage_reached']==3 and r['error']=='advance_cancelled' and not r['actual_process_owner_selected'] and r['primary_advance'] is None and r['reaped'])
    ck('actual_old_owner_retained',r['baseline']==r['old_owner_after']==fault['baseline'])
    ck('no_late_physical_record',not (OUT/'OWNER_RESULT_1.json').exists())
    for case in (0,1):
        a=read(OUT/('SOURCE_START_'+str(case)+'.json'));b=read(OUT/('SOURCE_COMPUTE_END_'+str(case)+'.json'))
        ck(str(case)+':entire_endogenous_loaded_files_and_module_map',a['files']==b['files'] and a['module_files']==b['module_files'])
        for p,h in b['files'].items():ck(str(case)+':actual_endogenous_loaded:'+p,sha(p)==h)
        response=decode(read(OUT/('RESPONSE_ATTEMPT_'+str(case)+'.json')))
        ck(str(case)+':source_gate_before_actual_private_install',response['source_qualified'] and response['source_start']==a['files'] and response['source_end']==b['files'])
        ck(str(case)+':computed_and_installed_before_selection_deadline',response['certified'] and response['worker_finished']<normal['response']['deadline'] if case==0 else response['certified'] and response['worker_finished']<response['deadline'])
    names={'qualify_pickup_endogenous_owner_transaction_v2.py','qualify_pickup_endogenous_owner_transaction_v1.py'}
    ck('actual_owned_processes_reaped',not [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])])
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),failure_names=[c['name'] for c in checks if not c['pass_']],producer_sha256=sha(OUT/'RESULT.json'),normal_audit_sha256=sha(OUT/'INDEPENDENT_PRIVATE_OWNER.json'),script_sha256=sha(__file__),no_solver_or_policy_reexecution=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_STORED_FAULT_CLOSURE.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=result['checks_count'],failed=result['failed'],failures=result['failure_names'])))
    return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
