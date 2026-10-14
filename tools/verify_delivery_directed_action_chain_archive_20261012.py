"""Read-only final identity verification of this newly completed stage."""
import json,hashlib,os,tomllib
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'delivery/directed_action_chain_root_v1';B=R/'current_entries_before_directed_action_chain_20261012'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
checks=[]
def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
for n in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):
    for p,h in read(O/n)['inputs'].items():ck(n+':'+p,sha(p)==h)
for prefix in ('','AUDIT_'):
    start=read(O/(prefix+'SOURCE_START.json'));end=read(O/(prefix+'SOURCE_END.json'));ck(prefix+'same_files_maps',start['files']==end['files'] and start['module_files']==end['module_files'])
    for p,h in start['files'].items():ck(prefix+'actual:'+p,sha(p)==h)
for x in read(B/'UPDATE_IDENTITY.json')['files']:ck('entry:'+x['path'],sha(E/x['path'])==x['after_sha256']);ck('backup:'+x['path'],sha(B/x['path'])==x['before_sha256'])
arc=read(O/'ARCHIVE_IDENTITY.json')
for p,h in arc['frozen'].items():ck('frozen:'+p,sha(p)==h)
ck('report',sha(arc['report_path'])==arc['report_sha256']);ck('boundary',sha(arc['boundary_path'])==arc['boundary_sha256']);ck('entry_index',sha(B/'UPDATE_IDENTITY.json')==arc['entry_update_sha256'])
ck('family2_registration_unchanged',sha(R/'delivery/multistep_insertable_family2_root_v1/RESEARCH_REGISTRATION.json')=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176')
state=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json');ck('current_new_scope',state['delivery']['directed_action_chain_independent_checks']==1886864 and state['delivery']['directed_action_learned_scorer_qualified'] is False and state['delivery']['modified_production_decoder_qualified'] is False)
a=tomllib.loads(Path(r'C:/Users/mmc/.codex/automations/maskco/automation.toml').read_text(encoding='utf-8-sig'));ck('automation_active30',a['status']=='ACTIVE' and a['rrule']=='FREQ=MINUTELY;INTERVAL=30');ck('automation_current_summary','不要重复本163/1886864' in a['prompt'] and '实际最大8.881784197001252e-16' in a['prompt'])
names={'qualify_delivery_directed_action_chain_v1.py','audit_delivery_directed_action_chain_v1.py'};active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(t).name in names for t in p.info['cmdline'] or [])];ck('owned_stage_processes_exited',not active)
for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json'):ck(n,read(O/n)['exit_code']==0)
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not x['passed'] for x in checks),owned_stage_processes=active,production_decoder_modified=False,scientific_training_started=False,new_scientific_family=False,shared_MD_uploaded=False,stable_online_gain=False,verification_only_no_experiment_rerun=True)
with (O/'FINAL_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(checks=len(checks),failed=out['failed'],owned_processes=active,automation=a['status'])));assert not out['failed']
