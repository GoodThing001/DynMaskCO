"""One new final identity check, permitting honestly empty error logs."""
from pathlib import Path
import json,os,tomllib,struct,hashlib
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/lineage_header_cancel_root_v1';B=R/'current_entries_before_lineage_header_cancel_20261012'
def main():
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    reg=read(O/'REGISTRATION.json');aud=read(O/'AUDIT_REGISTRATION.json');arc=read(O/'ARCHIVE_IDENTITY_R2.json');idx=read(B/'UPDATE_IDENTITY_R2.json')
    for title,mapping in [('registration',reg['sources']),('audit_inputs',aud['inputs']),('frozen',arc['frozen'])]:
        for p,h in mapping.items():ck(title+':'+p,Path(p).is_file() and sha(p)==h)
    for begin_name,end_names in [('SOURCE_START.json',['SOURCE_COMPUTE_13.json','SOURCE_END_13.json','SOURCE_COMPUTE_14.json','SOURCE_END_14.json']),('AUDIT_SOURCE_START.json',['AUDIT_SOURCE_END.json'])]:
        begin=read(O/begin_name)
        for end_name in end_names:
            end=read(O/end_name);ck('closed:'+end_name,begin['files']==end['files'] and begin['module_files']==end['module_files'])
        for p,h in begin['files'].items():ck('actual_file:'+begin_name+':'+p,sha(p)==h)
    ck('archive_report',sha(arc['report_path'])==arc['report_sha256']);ck('archive_boundary',sha(arc['boundary_path'])==arc['boundary_sha256']);ck('archive_R2_entries',sha(B/'UPDATE_IDENTITY_R2.json')==arc['entry_update_sha256'])
    for x in idx['files']:ck('entry:'+x['path'],sha(E/x['path'])==x['after_sha256'] and sha(B/x['path'])==x['before_sha256'])
    ck('sixteen_entries',len(idx['files'])==16)
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_LINEAGE_HEADER_CANCEL.json');ck('production_independent_actual_results',(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(20,0,1818,0));ck('supervisor_audit_actual_exit',read(O/'RUN_EXIT.json')['exit_code']==read(O/'AUDIT_RUN_EXIT.json')['exit_code']==0)
    ck('honest_worker_kill_return1',r['actual_worker_terminated'] and r['actual_worker_exit']==1)
    head=struct.Struct('<QQ32s32s32s');ret=(O/'RETAINED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(ret)
    ck('retained_real13_original_bytes',epoch==13 and size==42569 and ret[head.size:]==(O/'RETAINED_PAYLOAD.bin').read_bytes()==(O/'ACTUAL_WRITER_PAYLOAD_13.bin').read_bytes()==(O/'ACTUAL_SHARED_PAYLOAD_13.bin').read_bytes() and ret==(O/'ACTUAL_SHARED_ENVELOPE_13.bin').read_bytes() and ph==hashlib.sha256(ret[head.size:]).digest() and ph.hex()=='5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7')
    bad=(O/'ACTUAL_FAULT_SHARED_ENVELOPE_14.bin').read_bytes();ep,n,b,p,d=head.unpack_from(bad);wrong=bytearray(sh);wrong[0]^=1
    ck('shared_bad14_does_not_replace13',ep==14 and n==51049 and b==bytes(wrong) and bad==(O/'FAULT_HEADER_ENVELOPE_14.bin').read_bytes() and bad[head.size:]==(O/'ACTUAL_WRITER_PAYLOAD_14.bin').read_bytes() and p==hashlib.sha256(bad[head.size:]).digest() and r['host'][1]['ticket'] is None and r['host'][1]['authority_digest']==sh.hex())
    v=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json');pk=v['pickup']
    ck('all_current_and_legacy_pointers',(pk['current_authoritative_epoch'],pk['caller_selected_epoch'],pk['current_authoritative_slot'],pk['caller_selected_slot'],pk['caller_selected_callback'],pk['caller_selected_order'],pk['next_resume_callback'],pk['discarded_pending_callback'],pk['discarded_pending_order'])==(13,13,1,1,30,48,31,31,39))
    ck('current_source_payload',sha(ROOT/pk['current_authoritative_source'])==ph.hex() and pk['current_authoritative_payload_bytes']==42569 and pk['current_authoritative_payload_sha256']==ph.hex());ck('cross_line_pointer', 'retainedslot1epoch13' in v['delivery']['next'] and 'pending31/order39' in v['delivery']['next'])
    correction=read(O/'LEGACY_POINTER_FIELD_CORRECTION_R2.json');ck('prior_pointer_bytes_preserved',sha(O/'WORK_STATUS_BEFORE_LEGACY_POINTER_CORRECTION_R2.json')==correction['previous_WORK_STATUS_sha256'] and sha(B/'UPDATE_IDENTITY.json')==idx['supersedes_pointer_metadata_only'] and sha(O/'ARCHIVE_IDENTITY.json')==arc['initial_archive_identity_sha256'])
    config=tomllib.loads(Path(r'C:\Users\mmc\.codex\automations\maskco\automation.toml').read_text(encoding='utf-8-sig'));ck('automation_ACTIVE_30_scope',config['status']=='ACTIVE' and config['rrule']=='FREQ=MINUTELY;INTERVAL=30' and 'lineage_header_cancel_root_v1' in config['prompt'] and '20运行/1818' in config['prompt'] and 'RETAINED_PAYLOAD epoch13' in config['prompt']);ck('automation_tool_success',not read(R/'AUTOMATION_LINEAGE_HEADER_CANCEL_UPDATE_20261012.json')['app_tool_is_error'])
    names={'qualify_pickup_lineage_header_cancel_v1.py','audit_pickup_lineage_header_cancel_v1.py'};live=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in names for t in p.info['cmdline'] or [])];ck('no_owned_live_workers',not live)
    ck('scientific_gates_unchanged_false',not any(r[k] for k in ('physical_ack_loss_qualified','full_live_request_qualified','full_loaded_dependency_closure_qualified','stable_online_gain')))
    failed=[x['name'] for x in checks if not x['passed']];create(O/'FINAL_ARCHIVE_VERIFICATION.json',dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,no_production_or_scientific_rerun=True,owned_processes_active=live,empty_stderr_valid_if_hashed=True));print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed)));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
