"""Verify only new archive/entry/automation identities, without scientific replay."""
import hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/actual_score_chain_execution_root_v1';A=O/'downloaded'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    checks=[]
    def check(name,v):checks.append(dict(name=name,passed=bool(v)))
    manifest=read(O/'DOWNLOAD_MANIFEST.json')
    for x in manifest:
        p=A/x['name'];check('download:'+x['name'],p.stat().st_size==x['size'] and sha(p)==x['sha256'])
    check('registration_exact',sha(O/'EXECUTION_REGISTRATION.json')==sha(A/'EXECUTION_REGISTRATION.json')=='86959e8c62018023cc812337b2c801cf66e794491ec2138c642e602d35b5b948')
    receipt=read(O/'LOCAL_RECORDING_RECEIPT.json')
    check('new_scope_identity',sha(O/'ACTUAL_CHAIN_COMPONENT_SCOPE.json')==receipt['scope_sha256'])
    check('new_report_identity',sha(Path(receipt['report_path']))==receipt['report_sha256'])
    boundary=R/'reproduction/ACTUAL_SCORE_CHAIN_COMPONENT_EVIDENCE_20261012.json'
    check('new_boundary_identity',sha(boundary)==receipt['boundary_sha256'])
    backup=R/'current_entries_before_actual_score_chain_execution_20261012'
    check('new_entry_manifest_identity',sha(backup/'UPDATE_IDENTITY.json')==receipt['entry_manifest_sha256'])
    entries=read(backup/'UPDATE_IDENTITY.json')['files']
    check('sixteen_current_entries',len(entries)==16)
    for x in entries:
        check('current:'+x['path'],sha(E/x['path'])==x['after_sha256'])
        check('preserved_before:'+x['path'],sha(backup/x['path'])==x['before_sha256'])
    scope=read(O/'ACTUAL_CHAIN_COMPONENT_SCOPE.json')
    check('exact_new_download_scope',scope['downloaded_files']==182 and scope['downloaded_bytes']==72979943)
    check('ended_actual_tree',scope['actual_owned_tree_ended'])
    check('limited_component_and_science_boundaries',scope['raw_component_qualified'] and scope['complete_chain_component_qualified'] and not any(scope[k] for k in ['learned_output_qualified','formal_controls_qualified','full_request_budget_qualified','full_scientific_loaded_closure_qualified','scientific_family3_started','stable_online_gain','shared_MD_uploaded','old_initial_component_QA_repeated']))
    check('both_actual_complete_fallbacks',len(scope['chains'])==2 and all(x['steps']==25 and x['status']=='complete' and not x['remaining'] and x['incumbent_hard'] and x['candidate_hard'] and not x['candidate_selected'] and x['parameters_unchanged'] for x in scope['chains']))
    check('explicit_not_tensecond_qualification',scope['chains'][0]['wall_seconds']>10 and not scope['full_request_budget_qualified'])
    check('scoped_actual_closures',all(x['file_identities_equal'] and x['module_map_equal'] and not x['changed'] and not x['new_files'] for x in scope['actual_workload_closures'].values()))
    cfgpath=Path(r'C:/Users/mmc/.codex/automations/maskco/automation.toml')
    beforepath=O/'AUTOMATION_BEFORE_ACTUAL_SCORE_CHAIN_EXECUTION.toml'
    c=tomllib.loads(cfgpath.read_text(encoding='utf-8-sig'));b=tomllib.loads(beforepath.read_text(encoding='utf-8-sig'))
    add=read(O/'AUTOMATION_ADDED_EVIDENCE.json')['addendum']
    check('automation_exact_evidence_append',c['prompt']==b['prompt']+add)
    for k in ['id','kind','name','status','rrule','target_thread_id']:
        check('automation_preserved:'+k,c[k]==b[k])
    check('automation_active30min',c['status']=='ACTIVE' and c['rrule']=='FREQ=MINUTELY;INTERVAL=30')
    for k in ['notification_policy','notificationPolicy']:
        check('notification_preserved:'+k,c.get(k)==b.get(k))
    w=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
    check('latest_delivery_next_closes_preparation',w['delivery']['actual_score_chain_execution_ended'] and w['delivery']['actual_score_chain_complete_component_qualified'] and 'do NOT repeat' in w['delivery']['next'])
    # Preserve routine inspection mistakes as scope notes, never rewrite experiments.
    notes=['A readonly console summary assumed steps were filenames; actual steps are file/SHA dictionaries. It raised TypeError after printing the first-arm results. No artifact/model/solver was changed or rerun.', 'First automation read used ensure_ascii and exceeded the tool output budget; parsing the truncated output raised SyntaxError. Exact before-TOML was already preserved; bounded UTF-8 chunks read that original backup. No experimental source/result changed.']
    failed=sum(not x['passed'] for x in checks)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=failed,
        scope='New archive/16 current entries/backup/automation identity only; no production, independent scientific replay or closed QA rerun.',
        readonly_console_notes=notes,automation_before_sha256=sha(beforepath),automation_current_sha256=sha(cfgpath),
        automation_addendum_sha256=sha(O/'AUTOMATION_ADDED_EVIDENCE.json'),model_solver_calls=0,scientific_family3_started=False)
    create(O/'FINAL_RECORDING_VERIFICATION.json',result)
    create(O/'AUTOMATION_RECORDING_RECEIPT.json',dict(before_sha256=result['automation_before_sha256'],current_sha256=result['automation_current_sha256'],
        exact_addendum_applied=c['prompt']==b['prompt']+add,active=c['status']=='ACTIVE',recording_verification_sha256=sha(O/'FINAL_RECORDING_VERIFICATION.json')))
    print(json.dumps(dict(checks_count=len(checks),failed=failed,archive_files=len(manifest),active30min=c['status']=='ACTIVE',scope=result['scope']),ensure_ascii=True))
    raise SystemExit(bool(failed))
if __name__=='__main__':main()
