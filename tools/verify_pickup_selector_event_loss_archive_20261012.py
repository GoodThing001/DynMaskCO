"""Read-only final identities; no experiment or old QA rerun."""
import json,hashlib,os,tomllib
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/selector_event_loss_root_v1';B=R/'current_entries_before_selector_event_loss_20261012'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1048576),b''):h.update(v)
    return h.hexdigest()
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for field,name in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):
    for p,h in read(O/name)[field].items():ck(name+':'+p,sha(p)==h)
start=read(O/'SOURCE_START.json')
for name in ('SOURCE_COMPUTE_9.json','SOURCE_END_9.json','SOURCE_COMPUTE_10.json','SOURCE_END_10.json','SOURCE_AT_SELECTOR_PUBLICATION.json'):
    end=read(O/name);ck(name+':closed',start['files']==end['files'] and start['module_files']==end['module_files'])
for p,h in start['files'].items():ck('actual_producer:'+p,sha(p)==h)
start=read(O/'AUDIT_SOURCE_START.json');end=read(O/'AUDIT_SOURCE_END.json');ck('audit_closure',start['files']==end['files'] and start['module_files']==end['module_files'])
for p,h in start['files'].items():ck('actual_audit:'+p,sha(p)==h)
for x in read(B/'UPDATE_IDENTITY.json')['files']:ck('entry:'+x['path'],sha(E/x['path'])==x['after_sha256']);ck('backup:'+x['path'],sha(B/x['path'])==x['before_sha256'])
arc=read(O/'ARCHIVE_IDENTITY.json')
for p,h in arc['frozen'].items():ck('frozen:'+p,sha(p)==h)
ck('report',sha(arc['report_path'])==arc['report_sha256']);ck('boundary',sha(arc['boundary_path'])==arc['boundary_sha256']);ck('entry_index',sha(B/'UPDATE_IDENTITY.json')==arc['entry_update_sha256'])
ck('family2_original_registration',sha(R/'delivery/multistep_insertable_family2_root_v1/RESEARCH_REGISTRATION.json')=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176')
state=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json');ck('current_scoped_state',state['pickup']['selector_event_loss_independent_checks']==2314 and state['pickup']['caller_selected_epoch']==10 and state['pickup']['discarded_pending_order'] is None and state['pickup']['caller_selected_order']==5 and state['pickup']['next_resume_callback']==28 and not state['pickup']['full_live_request_qualified'] and not state['pickup']['full_loaded_dependency_closure_qualified']);ck('crossline_next_pointer','actual selected slot0 epoch10/callback27' in state['delivery']['next'])
a=tomllib.loads(Path(r'C:/Users/mmc/.codex/automations/maskco/automation.toml').read_text(encoding='utf-8-sig'));ck('automation_active30',a['status']=='ACTIVE' and a['rrule']=='FREQ=MINUTELY;INTERVAL=30');ck('automation_current_scope','不要重复本22/2314' in a['prompt'] and '8.0261676秒' in a['prompt'] and 'callback27/order5' in a['prompt'])
ck('automation_receipt',read(R/'AUTOMATION_SELECTOR_EVENT_LOSS_UPDATE_20261012.json')['automation_id']=='maskco')
names={'qualify_pickup_selector_event_loss_v1.py','audit_pickup_selector_event_loss_v1.py'};active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(t).name in names for t in p.info['cmdline'] or [])];ck('owned_stage_processes_exited',not active)
for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json'):ck(n,read(O/n)['exit_code']==0)
receipt=read(O/'ACTUAL_EXIT_RECEIPTS.json');ck('killed_worker_not_natural0',receipt['worker_was_terminated'] and receipt['worker_actual_return_code']==read(O/'RESULT.json')['actual_worker_exit'] and receipt['worker_natural_exit_0'] is False)
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not x['passed'] for x in checks),owned_stage_processes=active,verification_only_no_experiment_rerun=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,shared_MD_uploaded=False,stable_online_gain=False)
with (O/'FINAL_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(checks=len(checks),failed=out['failed'],owned_processes=active,automation=a['status'])));assert not out['failed']
