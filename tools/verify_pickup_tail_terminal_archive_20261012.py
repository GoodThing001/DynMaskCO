"""One final archive identity check; never reexecute production, science or audits."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,tomllib,psutil,pickle
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup';O=P/'resident_day_tail_root_v1';A=P/'resident_tail_stream_audit_root_v3';T=P/'terminal_restore_root_v2';B=R/'current_entries_before_tail_terminal_recovery_20261012'
def main():
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    inputs={str(p):sha(p) for p in [Path(__file__),ROOT/'tools/record_pickup_tail_terminal_recovery_20261012.py',T/'ARCHIVE_IDENTITY.json',T/'SEALED_COMPONENT_FACTS.json',O/'RUN_EXIT.json',O/'FAILURE.json',O/'STREAM_AUDIT_FOOTER_FAILURE_R2.json',A/'INDEPENDENT_RESIDENT_TAIL_STREAM_R3.json',T/'INDEPENDENT_TERMINAL_RESTORE.json',B/'UPDATE_IDENTITY.json']}
    create(T/'FINAL_ARCHIVE_REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,scope='Current bytes and entries/process/automation identity only, no production or auditor/solver/model rerun. Empty error logs allowed as actual natural-zero-run artifacts. Original failed producer/unknown worker returncode/R2 footer/raw event ledger gaps remain.'))
    for directory,n,field in [(O,'REGISTRATION.json','sources'),(A,'STREAM_AUDIT_REGISTRATION_R3.json','inputs'),(T,'REGISTRATION.json','sources'),(T,'AUDIT_REGISTRATION.json','inputs')]:
        for p,h in read(directory/n)[field].items():ck('registered_identity:'+str(directory.name)+':'+p,sha(p)==h)
    archive=read(T/'ARCHIVE_IDENTITY.json')
    for p,h in archive['frozen'].items():ck('frozen:'+p,sha(p)==h)
    ck('report_identity',sha(archive['report_path'])==archive['report_sha256']);ck('boundary_identity',sha(archive['boundary_path'])==archive['boundary_sha256']);ck('entry_manifest_identity',sha(B/'UPDATE_IDENTITY.json')==archive['entry_update_sha256'])
    entries=read(B/'UPDATE_IDENTITY.json')['files'];ck('sixteen_entries',len(entries)==16)
    for row in entries:
        ck('entry_after:'+row['path'],sha(E/row['path'])==row['after_sha256']);ck('entry_old_saved:'+row['path'],sha(B/row['path'])==row['before_sha256'])
    w=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json');p=w['pickup'];f=read(T/'SEALED_COMPONENT_FACTS.json')
    ck('caller_terminal_pointer',p['caller_selected_epoch']==p['current_authoritative_epoch']==64 and p['caller_selected_callback']==80 and p['caller_selected_order']==75 and p['caller_selected_slot']==1 and p['current_authoritative_payload_sha256']==f['terminal_payload_sha256'] and sha(ROOT/p['current_authoritative_source'])==f['terminal_payload_sha256'])
    ck('no_pending_callback81',p['discarded_pending'] is None and p['discarded_pending_order'] is None and p['discarded_pending_callback'] is None and p['next_resume_callback'] is None and p['current_observed_controller_at_terminal'])
    ck('scientific_gates_not_upgraded',not w['stable_online_gain_established'] and not w['MaskCO_attribution_established'] and not p['full_live_request_qualified'] and not p['full_loaded_dependency_closure_qualified'] and not p['physical_ack_loss_qualified'] and not p['resident_day_tail_whole_run_PASS'] and not p['raw_intermediate_physical_event_ledger_qualified'])
    ck('actual_result_counts',read(A/'INDEPENDENT_RESIDENT_TAIL_STREAM_R3.json')['checks_count']==31107 and read(A/'INDEPENDENT_RESIDENT_TAIL_STREAM_R3.json')['failed']==0 and read(T/'RESULT.json')['checks_count']==9 and read(T/'INDEPENDENT_TERMINAL_RESTORE.json')['checks_count']==1164 and read(T/'INDEPENDENT_TERMINAL_RESTORE.json')['failed']==0)
    ck('original_failures_preserved',read(O/'RUN_EXIT.json')['exit_code']==1 and read(P/'resident_day_tail_launch_root_v1/EXIT.json')['actual_producer_returncode']==1 and not (O/'RESULT.json').exists() and not (O/'ACTUAL_SHARED_PAYLOAD_64.bin').exists() and not (O/'OWNER_STATE_64.json').exists() and read(O/'STREAM_AUDIT_FOOTER_FAILURE_R2.json')['checks_count_unknown'])
    ck('natural_terminal_exit',read(T/'RESULT.json')['actual_worker_exit']==0 and read(T/'RUN_EXIT.json')['exit_code']==0 and read(T/'ACTUAL_EXIT_RECEIPTS.json')['original_worker_returncode_not_archived'])
    payload=(T/'RETAINED_PAYLOAD.bin').read_bytes();cp=pickle.loads(payload)
    ck('actual_terminal_raw_identity',len(payload)==123700 and sha(T/'RETAINED_PAYLOAD.bin')=='6730d6883ec6bc4ec60e4fdf30abae0202130e17a248c2673e97c46b3a047d92' and payload==(T/'ACTUAL_SHARED_PAYLOAD_64.bin').read_bytes()==(T/'ACTUAL_WRITER_PAYLOAD_64.bin').read_bytes() and cp['epoch']==64 and cp['state']['event_id']==158 and len(cp['state']['state']['accepted'])==74 and all(v.status=='closed' for v in cp['state']['fleet']))
    cfg=tomllib.loads(Path(r'C:\Users\mmc\.codex\automations\maskco\automation.toml').read_text(encoding='utf-8-sig'))
    ck('automation_active30_correct_scope',cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=MINUTELY;INTERVAL=30' and 'resident_tail_stream_audit_root_v3' in cfg['prompt'] and '原整轮仍FAIL' in cfg['prompt'] and 'SHA6730d6883ec6bc4ec60e4fdf30abae0202130e17a248c2673e97c46b3a047d92' in cfg['prompt'] and '无callback81' in cfg['prompt'])
    create(R/'AUTOMATION_TAIL_TERMINAL_SCOPE_UPDATE_20261012.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),tool_update_success=True,actual_status=cfg['status'],actual_rrule=cfg['rrule'],actual_prompt_chars=len(cfg['prompt']),scope_terminal_epoch=64,original_whole_run_PASS=False,shared_MD_uploaded=False))
    names={'qualify_pickup_resident_day_tail_v1.py','run_pickup_resident_day_tail_bounded_v1.py','audit_pickup_resident_tail_stream_r2.py','audit_pickup_resident_tail_stream_r3.py','qualify_pickup_terminal_restore_v2.py','audit_pickup_terminal_restore_v2.py'}
    live=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(x).name in names for x in p.info['cmdline'] or [])];ck('owned_actual_processes_gone',not live)
    for p,h in inputs.items():ck('archive_inputs_unchanged:'+p,sha(p)==h)
    failures=[c['name'] for c in checks if not c['passed']];create(T/'FINAL_ARCHIVE_VERIFICATION.json',dict(checks=checks,checks_count=len(checks),failed=len(failures),failure_names=failures,owned_processes_active=live,original_whole_run_PASS=False,scope='Archive identity only; no scientific or physical implementation rerun.'))
    print(json.dumps(dict(checks=len(checks),failed=len(failures),failure_names=failures,active=live)));return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
