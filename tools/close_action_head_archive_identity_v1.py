"""Close archive identity and precisely describe two later mapped sealed files."""
import json,shutil,tomllib
from pathlib import Path
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';O=R/'delivery/actual_action_head_execution_root_v1';A=O/'downloaded'

def main():
    s=read(A/'SOURCE_START.json');e=read(A/'SOURCE_END.json');a=read(A/'AUDIT_SOURCE_START.json');z=read(A/'AUDIT_SOURCE_END.json')
    added={n:p for n,p in e['module_files'].items() if n not in s['module_files']}
    changed={n:p for n,p in e['module_files'].items() if n in s['module_files'] and p!=s['module_files'][n]}
    removed=sorted(set(s['module_files'])-set(e['module_files']))
    assert set(added)=={'coldchain_state','delivery_evaluator'} and not changed and not removed
    assert s['files']==e['files'] and not e['new_files'] and not e['changed']
    assert all(p in s['files'] and s['files'][p]==e['files'][p] for p in added.values())
    assert a['files']==z['files'] and a['module_files']==z['module_files'] and not z['new_files'] and not z['changed']
    details=dict(production_files_begin_end=8890,production_module_map_begin=1063,production_module_map_end=1065,production_module_map_equal=False,added_module_mappings={n:dict(path=p,file_sha256=s['files'][p],file_sealed_at_start=True,bytes_unchanged=True) for n,p in added.items()},changed_or_removed_mappings=False,audit_files_begin_end=8954,audit_module_map_begin_end=1015,audit_module_map_equal=True,new_unsealed_files=False,scope='Two modules become mapped during production, both exact original source files and existing cached bytecode already sealed at start. Mapping sets are not identical. No new unsealed dependency file; no C0 evaluation or oldQA replay.',scientific_full_loaded_dependency_closure_qualified=False)
    assert len(s['files'])==len(e['files'])==8890 and len(a['files'])==len(z['files'])==8954
    create(O/'MODULE_MAP_LOADING_NOTE.json',details)
    rec=read(O/'LOCAL_STAGE_RECORDING_RECEIPT.json');report=Path(rec['report_path'])
    shutil.copy2(report,O/'REPORT_BEFORE_MODULE_MAP_DETAIL.md')
    with report.open('a',encoding='utf-8') as f:f.write('\n实际加载范围补充：生产实际8890文件起止逐SHA一致，无新增未封存文件；module映射从1063至1065，新增coldchain_state和delivery_evaluator两映射，二者原源/既存pyc已包含开始文件清单且字节不变。因此不称生产module映射集合起止相同。核查8954文件/1015映射起止一致。此证明只属本初始评分负载，正式完整科学依赖资格仍False；未实际运行C0评价。原RESULT/审计/闭包/计数/退出及全部生产字节未改、未重跑。\n')
    prior=R/'current_entries_before_actual_initial_action_head_execution_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files']
    assert all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_action_head_module_map_detail_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    work=E/'results/research_20261010_continuation_v2/WORK_STATUS.json';w=read(work)
    w['delivery']['actual_action_head_module_map_loading_note']=str((O/'MODULE_MAP_LOADING_NOTE.json').relative_to(ROOT))
    w['delivery']['actual_action_head_production_module_map_equal']=False
    w['delivery']['actual_action_head_no_new_unsealed_files']=True
    work.write_text(json.dumps(w,ensure_ascii=False,indent=2),encoding='utf-8')
    updates=[dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(E/x['path'])) for x in entries]
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),module_map_note_sha256=sha(O/'MODULE_MAP_LOADING_NOTE.json'),shared_MD_uploaded=False))
    manifest=read(O/'DOWNLOAD_MANIFEST.json');checks={}
    for x in manifest:checks['original:'+x['name']]=(A/x['name']).stat().st_size==x['size'] and sha(A/x['name'])==x['sha256']
    for x in updates:
        checks['entry:'+x['path']]=sha(E/x['path'])==x['after_sha256']
        checks['previous:'+x['path']]=sha(backup/x['path'])==x['before_sha256']
    reg=read(O/'EXECUTION_REGISTRATION.json')
    for n,h in reg['package_files'].items():checks['local_frozen:'+n]=sha(O/n)==h
    checks['before_execution_registration_identity']=sha(O/'EXECUTION_REGISTRATION.json')==read(O/'LOCAL_EXECUTION_BINDING.json')['registration_sha256']==sha(A/'EXECUTION_REGISTRATION.json')
    checks['producer_and_audit_original_natural_zero']=read(A/'RUN_EXIT.json')['natural_exit'] and read(A/'AUDIT_RUN_EXIT.json')['natural_exit'] and read(A/'CHAIN_EXIT.json')['exit_code']==0
    checks['whole_scientific_gates_closed']=not w['delivery']['actual_action_head_execution_scope']['full_C0_qualified'] and not w['delivery']['actual_action_head_execution_scope']['full_request_budget_qualified'] and not w['delivery']['actual_action_head_execution_scope']['stable_online_gain']
    assert all(checks.values())
    create(O/'FINAL_ARCHIVE_IDENTITY_VERIFICATION.json',dict(checks_count=len(checks),failed=0,checks=checks,identity_only=True,module_map_note_sha256=sha(O/'MODULE_MAP_LOADING_NOTE.json'),report_sha256=sha(report),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),no_experiment_or_audit_reexecuted=True))
    print(json.dumps(dict(archive_identity_checks=len(checks),failed=0,production_files=8890,production_module_maps=[1063,1065],scientific_full_closure=False)))

if __name__=='__main__':main()
