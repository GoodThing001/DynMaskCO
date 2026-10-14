"""Final read-only new-stage inputs, loaded bytes, backups and automation identities."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,tomllib
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'action_tensor_binding_root_v2';OLD=D/'action_tensor_binding_root_v1';B=R/'current_entries_before_action_tensor_binding_20261012'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for n in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):
    for p,h in read(O/n)['inputs'].items():ck(n+':'+p,sha(p)==h)
for prefix in ('','AUDIT_'):
    start=read(O/(prefix+'SOURCE_START.json'));end=read(O/(prefix+'SOURCE_END.json'));ck(prefix+'closed',start['files']==end['files'] and start['module_files']==end['module_files'])
    for p,h in start['files'].items():ck(prefix+'loaded:'+p,sha(p)==h)
for x in read(B/'UPDATE_IDENTITY.json')['files']:ck('entry:'+x['path'],sha(E/x['path'])==x['after_sha256']);ck('saved_before:'+x['path'],sha(B/x['path'])==x['before_sha256'])
arc=read(O/'ARCHIVE_IDENTITY.json')
for p,h in arc['frozen'].items():ck('frozen:'+p,sha(p)==h)
ck('report',sha(arc['report_path'])==arc['report_sha256']);ck('boundary',sha(arc['boundary_path'])==arc['boundary_sha256']);ck('entryindex',sha(B/'UPDATE_IDENTITY.json')==arc['entry_update_sha256'])
r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_ACTION_TENSOR_BINDING.json');ck('2254_15160_actual0',(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(2254,0,15160,0))
ck('tensor_same_original_bytes',sha(OLD/'TENSORS.npz')==sha(O/'TENSORS.npz')==r['tensor_sha256']=='31e8da097121f173e0f97adb1f3d6758d2caeab1ed12cdb54ee258262243c58d')
failure=read(OLD/'RESULT.json');ck('original2254_oneFAIL_retained',failure['checks_count']==2254 and failure['failed']==1 and [x['name'] for x in failure['checks'] if not x['passed']]==['loaded_identity'] and read(OLD/'RUN_EXIT.json')['exit_code']==1)
recovery=read(OLD/'CLOSURE_FAILURE_RECOVERY_REGISTRATION_R2.json')
for field,name in [('original_result_sha256','RESULT.json'),('original_registration_sha256','REGISTRATION.json'),('original_start_sha256','SOURCE_START.json'),('original_end_sha256','SOURCE_END.json')]:ck('recovery_original:'+field,sha(OLD/name)==recovery[field])
ck('binding_module_unmodified',sha(ROOT/'tools/delivery_action_tensor_binding_v1.py')==recovery['binding_module_sha256'])
ck('family2_original_registration',sha(D/'multistep_insertable_family2_root_v1/RESEARCH_REGISTRATION.json')=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176')
state=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json');ck('delivery_current_scoped',state['delivery']['action_tensor_binding_runtime_checks']==2254 and state['delivery']['action_tensor_binding_independent_checks']==15160 and not state['delivery']['action_tensor_binding_learned_output_qualified']);ck('pickup_authority_unaltered',state['pickup']['caller_selected_epoch']==12 and state['pickup']['caller_selected_callback']==29 and state['pickup']['next_resume_callback']==30 and not state['pickup']['full_live_request_qualified'])
auto=tomllib.loads(Path('C:/Users/mmc/.codex/automations/maskco/automation.toml').read_text(encoding='utf-8-sig'));ck('automation_active30',auto['status']=='ACTIVE' and auto['rrule']=='FREQ=MINUTELY;INTERVAL=30');ck('automation_current_scope','不要重复本2254/15160' in auto['prompt'] and '第3族未启动' in auto['prompt']);ck('automation_receipt',read(R/'AUTOMATION_ACTION_TENSOR_BINDING_UPDATE_20261012.json')['automation_id']=='maskco')
names={'qualify_delivery_action_tensor_binding_v1.py','qualify_delivery_action_tensor_binding_v2.py','audit_delivery_action_tensor_binding_v2.py'};active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in names for t in p.info['cmdline'] or [])];ck('actual_owned_processes_exited',not active)
for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json'):ck(n,read(O/n)['exit_code']==0)
ck('no_science_model_physics_budget_upgrade',not any(r[k] for k in ('learned_output_qualified','full_C0_qualified','hard_request_budget_qualified','new_scientific_family','stable_online_gain')))
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),owned_processes_active=active,verification_only_no_experiment_rerun=True,old_FAIL_preserved=True,shared_MD_uploaded=False,stable_online_gain=False)
with (O/'FINAL_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(checks=len(checks),failed=out['failed'],active=active,automation=auto['status'])));assert not out['failed']
