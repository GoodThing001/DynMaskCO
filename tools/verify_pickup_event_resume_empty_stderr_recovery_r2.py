"""Narrow audit assumption correction: zero-byte stderr is valid for natural0 exit."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'research/review_3080_20261010/pickup/event_loss_resume_root_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[O/n for n in ('FINAL_ARCHIVE_VERIFICATION.json','worker.stderr.txt','RESULT.json','RUN_EXIT.json','AUDIT_RUN_EXIT.json','ACTUAL_EXIT_RECEIPTS.json','INDEPENDENT_EVENT_LOSS_RESUME.json')]+[ROOT/'tools/verify_pickup_event_loss_resume_archive_20261012.py',Path(__file__)]
identities={str(p):sha(p) for p in inputs}
with (O/'FINAL_ARCHIVE_RECOVERY_R2_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs=identities,scope='Only reject checker assumption that every artifact including stderr must be nonempty. Confirm actual empty stderr with natural0 exit. No rerun of producer, independent audit, final1183 checks or old qualified QA; preserve original finalFAIL.',original_registered_barriers_unchanged=True),f,indent=2)
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
v=read(O/'FINAL_ARCHIVE_VERIFICATION.json');r=read(O/'RESULT.json');receipt=read(O/'ACTUAL_EXIT_RECEIPTS.json');a=read(O/'INDEPENDENT_EVENT_LOSS_RESUME.json')
failed=[x['name'] for x in v['checks'] if not x['passed']]
ck('original1183_only_empty_stderr_failure_preserved',v['checks_count']==1183 and v['failed']==1 and failed==['new_artifact_readable:worker.stderr.txt'])
ck('stderr_actual_empty_valid_no_error_output',(O/'worker.stderr.txt').is_file() and (O/'worker.stderr.txt').read_bytes()==b'')
ck('actual_worker_natural0',r['actual_worker_exit']==receipt['worker_actual_return_code']==0 and not r['actual_worker_terminated'] and receipt['worker_natural_exit_0'] and not receipt['worker_was_terminated'])
ck('actual_supervisor_and_audit0',read(O/'RUN_EXIT.json')['exit_code']==read(O/'AUDIT_RUN_EXIT.json')['exit_code']==receipt['supervisor_actual_exit']==receipt['independent_actual_exit']==0)
ck('production19_independent1807_unchanged',(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(19,0,1807,0))
ck('original_other1182_checks_retained',sum(c['passed'] for c in v['checks'])==1182)
ck('no_scientific_or_full_request_upgrade',not r['full_live_request_qualified'] and not r['full_loaded_dependency_closure_qualified'] and not r['physical_ack_loss_qualified'] and not r['stable_online_gain'])
ck('narrow_inputs_unchanged',all(sha(Path(p))==h for p,h in identities.items()))
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),original_final_archive_checks=1183,original_final_archive_failed=1,original_successful_items_retained=1182,qualification_composed_with_narrow_semantic_correction=True,original_failure_sha256=identities[str(O/'FINAL_ARCHIVE_VERIFICATION.json')],no_production_or_audit_rerun=True,no_registered_barrier_or_tolerance_changed=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
with (O/'FINAL_ARCHIVE_VERIFICATION_RECOVERY_R2.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(dict(checks=len(checks),failed=out['failed'],original_final_failed_preserved=1,production_and_independent_rerun=False)));assert not out['failed']
