"""Identity-only final recording audit. No experiment/model/physical replay."""
import hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/timed_actual_request_execution_root_v1';A=O/'downloaded'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def main():
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    reg=read(O/'EXECUTION_REGISTRATION.json');scope=read(O/'ACTUAL_TIMED_REQUEST_SCOPE.json')
    ck('registration_exact',sha(O/'EXECUTION_REGISTRATION.json')=='737325090b342bf714cabee8fffefd4b6262261249850ad19356d4bc1ce3acf8')
    for x in read(O/'DOWNLOAD_MANIFEST.json'):
        p=A/x['name'];ck('actual_original:'+x['name'],p.is_file() and p.stat().st_size==x['size'] and sha(p)==x['sha256'])
    for n,h in reg['package_files'].items():ck('registered_source:'+n,sha(O/n)==h==sha(A/n))
    ck('downloaded_registration_exact',sha(A/'EXECUTION_REGISTRATION.json')==sha(O/'EXECUTION_REGISTRATION.json'))
    recovery=read(O/'INDEPENDENT_STAGE_PATH_MEMBERSHIP_RECOVERY_R2.json');rr=read(O/'STAGE_PATH_MEMBERSHIP_RECOVERY_R2_REGISTRATION.json')
    ck('recovery_source_bound',sha(ROOT/'tools/recover_timed_stage_path_membership_r2.py')==rr['checker_sha256'])
    for n,h in rr['inputs'].items():ck('narrow_recovery_input:'+n,sha(O/n)==h)
    ck('recovery220_zero_without_replay',recovery['checks_count']==len(recovery['checks'])==220 and not any(not x['passed'] for x in recovery['checks']) and not recovery['model_or_C0_executed'])
    ck('original_FAILURE_counts_preserved',read(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json')['failed']==24 and read(A/'AUDIT_RUN_EXIT.json')['exit_code']==read(A/'CHAIN_EXIT.json')['exit_code']==1)
    ck('no_scientific_or_complete_budget_upgrade',all(scope[k] is False for k in ['full_request_budget_qualified','full_scientific_loaded_closure_qualified','formal_controls_qualified','learned_output_qualified','scientific_family3_started','stable_online_gain','complete_chain_within_budget','all_registered_faults_exercised','complete_all_request_dependency_closure']))
    backup=R/'current_entries_before_timed_actual_request_execution_20261012';ident=read(backup/'UPDATE_IDENTITY.json');prior=read(R/'current_entries_before_timed_request_independent_preparation_20261012/UPDATE_IDENTITY.json')
    before={x['path']:x['after_sha256'] for x in prior['files']}
    ck('16current_entries_same_identity_set',len(ident['files'])==16 and {x['path'] for x in ident['files']}==set(before))
    for x in ident['files']:
        ck('saved_prior_entry:'+x['path'],sha(backup/x['path'])==x['before_sha256']==before[x['path']])
        ck('current_entry:'+x['path'],sha(E/x['path'])==x['after_sha256'])
    receipt=read(O/'LOCAL_RECORDING_RECEIPT.json');detail=read(O/'REPORT_STAGE_CAPTURE_SECONDS_DETAIL.json')
    report=E/ident['report']
    ck('original_report_bytes_retained',sha(O/'REPORT_BEFORE_STAGE_CAPTURE_SECONDS_DETAIL.md')==detail['previous_report_sha256']==receipt['report_sha256'])
    ck('current_report_exact_saved_time_detail',sha(report)==detail['current_report_sha256'] and detail['actual_recorded_marker_to_delay_seconds']==[x['stage_marker_to_delay_seconds'] for x in scope['requests'] if x['actual_fault_delay_entry']])
    boundary=R/'reproduction/TIMED_ACTUAL_REQUEST_EVIDENCE_20261012.json'
    ck('scope_and_negative_append_exact',sha(O/'ACTUAL_TIMED_REQUEST_SCOPE.json')==receipt['scope_sha256']==read(boundary)['scope_sha256'] and sha(boundary)==receipt['boundary_sha256'])
    ck('entry_manifest_exact',sha(backup/'UPDATE_IDENTITY.json')==receipt['entry_manifest_sha256'])
    w=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
    ck('state_latest_timed_run_scope',w['delivery']['timed_actual_request_scope']==scope and w['delivery']['timed_actual_request_execution_ended'] and w['delivery']['independent_timed_checker_executed'])
    ck('next_no_rerun_or_scientific_claim','Do NOT repeat10 production' in w['delivery']['next'] and 'Family' not in w['delivery']['next'] or 'Pickup terminal64/pendingempty/no81' in w['delivery']['next'])
    b=tomllib.loads((O/'AUTOMATION_BEFORE_TIMED_EXECUTION.toml').read_text(encoding='utf-8-sig'))
    current=tomllib.loads(Path('C:/Users/mmc/.codex/automations/maskco/automation.toml').read_text(encoding='utf-8-sig'))
    addition=read(O/'AUTOMATION_ADDED_EVIDENCE.json')['addition']
    ck('automation_exact_append',current['prompt']==b['prompt']+addition)
    for k in ('version','id','kind','name','status','rrule','target_thread_id','created_at'):
        ck('automation_preserved:'+k,current[k]==b[k])
    ck('automation_notification_preference_preserved',current.get('notification_policy')==b.get('notification_policy'))
    ck('heartbeat_ACTIVE30min',current['id']=='maskco' and current['status']=='ACTIVE' and current['rrule']=='FREQ=MINUTELY;INTERVAL=30')
    failed=sum(not x['passed'] for x in checks)
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=failed,
        automation_prompt_sha256=hashlib.sha256(current['prompt'].encode('utf-8')).hexdigest(),
        original_independent_failed=24,narrow_recovery_failed=0,original_chain_exit=1,
        model_or_C0_or_experimental_QA_reexecuted=False,scope='Identity/recording only, permits actual zero-byte stdout/stderr. Original runtime/audit failures and scientific budget gaps unchanged.')
    with (O/'FINAL_ARCHIVE_IDENTITY_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(checks=len(checks),failed=failed,automation_exact_append=current['prompt']==b['prompt']+addition)))
    return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
