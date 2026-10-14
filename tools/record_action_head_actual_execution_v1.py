"""Record only observed original outputs, retaining all prior bytes and gates."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';O=D/'actual_action_head_execution_root_v1';A=O/'downloaded'

def main():
    manifest=read(O/'DOWNLOAD_MANIFEST.json');reg=read(O/'EXECUTION_REGISTRATION.json')
    assert all(sha(A/x['name'])==x['sha256'] and (A/x['name']).stat().st_size==x['size'] for x in manifest)
    assert sha(A/'EXECUTION_REGISTRATION.json')==sha(O/'EXECUTION_REGISTRATION.json')
    chain=read(A/'CHAIN_EXIT.json');run=read(A/'RUN_EXIT.json') if (A/'RUN_EXIT.json').exists() else None
    audit_exit=read(A/'AUDIT_RUN_EXIT.json') if (A/'AUDIT_RUN_EXIT.json').exists() else None
    result=read(A/'RESULT.json') if (A/'RESULT.json').exists() else None
    audit=read(A/'INDEPENDENT_ACTION_HEAD_AUDIT.json') if (A/'INDEPENDENT_ACTION_HEAD_AUDIT.json').exists() else None
    failures={n:read(A/n) for n in ['FAILURE.json','AUDIT_FAILURE.json','SUPERVISOR_FAILURE.json'] if (A/n).exists()}
    latest=read(sorted(O.glob('DOWNLOAD_PREFLIGHT_*.json'))[-1]);actual=read_status=json.loads(latest['stdout'])
    assert actual['complete'] and not actual['owned_processes']
    qualified=chain['exit_code']==0 and run and run['exit_code']==0 and audit_exit and audit_exit['exit_code']==0 and result and result['failed']==0 and audit and audit['failed']==0
    production_records=len(result['records']) if result else len(list(A.glob('*initial_*.json')))
    if qualified:
        assert production_records==60 and len(audit['replays'])==60 and audit['independent_backbone_forward'] and audit['independent_coordinate_normalization']
    observation=read(sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))[-1]);resource=json.loads(observation['stdout'])
    scope=dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),qualified=bool(qualified),model_executed=bool(production_records),production_records=production_records,production_checks=result['checks_count'] if result else None,production_failed=result['failed'] if result else None,independent_checks=audit['checks_count'] if audit else None,independent_failed=audit['failed'] if audit else None,run_exit=run,audit_exit=audit_exit,chain_exit=chain,failures=failures,actual_owned_processes_ended=True,initial_available_ram_gib=resource['available_ram_gib'],initial_cpu15_busy_percent=resource['cpu15_busy_percent'],downloaded_files=len(manifest),optimizer_steps=0,formal_controls_qualified=False,full_C0_qualified=False,full_request_budget_qualified=False,scientific_family3_started=False,stable_online_gain=False,shared_MD_uploaded=False)
    create(O/'ACTUAL_EXECUTION_SCOPE.json',scope)
    if qualified:
        description='原预训initial与fresh随机888各30首状态共60实际动作评分前向完成；同seed20261012四叶196929头、0更新。%d运行/%d新进程独立项0失败，60完整backbone/全10D/坐标归一及全部234叶参数独立重载精确；NumPy头最大差%.9g（事前2e-5仅头算术界限）。'%(result['checks_count'],audit['checks_count'],audit['max_abs_head_error'])
        next_action='Actual initial action-head scoring component and60 independent backbone/feature/seed/head replays qualified on fixed old30 first states; do NOT repeat. Next prospective integrate actual score provenance into full temporary-action chain/controls/completeC0/abortable whole request budget and matched equal-time iteration, then formal same-init trainedCE/task-trainedrandom and independently strong dynamic admission. No family3/data until all engineering/scientific barriers. Pickup terminal64/pendingempty, no callback81. Stable attributable online gain not established.'
    else:
        description='原实际链退出%d，保留全部源/登记/输出/失败/原退出；已存%d前向记录，生产%s失败、独立%s失败，缺项不补造；未完成整体工程资格。'%(chain['exit_code'],production_records,str(scope['production_failed']),str(scope['independent_failed']))
        next_action='Original initial action-head execution failed/unfinished qualification; inspect ACTUAL_EXECUTION_SCOPE/actual FAILURE/AUDIT_FAILURE and retained outputs before any recovery. Actual owned tree exited. Do NOT repeat closedQA or edit old executed source. New exclusive prospective engineering recovery only same scientific parameters after exact diagnosis. Family2FAILclosed, family3notstarted. Pickup terminal64/pendingempty/no81; scientific models/controls/fullC0/full10s/equal-time/strongdynamic remain.'
    note='\n> 2026-10-12 配送实际动作头组件：'+description+' 原登记SHA'+scope['registration_sha256']+'，真实3080 RAM%.6fGiB/CPU15负载%.1f%%核查后独占PY/JSON派发，900秒/阶段监督；全部原件只读下载并逐SHA核验，本人实际树已退出。固定旧30/22500动作，无教师标签成本输入；共享原JAX/Transformer，不是另一模型实现。未训练、不是同任务已训随机或同初始化已训CE，无solver/C0/新科学数据/第3族/整请求10秒资格或在线收益；第2族FAIL封闭。取货权威终局64/pending空无81，所有旧FAIL/缺ledger/13800保持，共享MD未上传。\n'%(resource['available_ram_gib'],resource['cpu15_busy_percent'])
    report=E/'02_配送线/文档/结果与进度/配送_实际初始动作头执行与完整独立重载核查_2026-10-12.md'
    text='# 实际初始动作头执行与完整独立重载核查\n\n'+note+'\n'+json.dumps({k:scope[k] for k in ['production_checks','production_failed','independent_checks','independent_failed','run_exit','audit_exit','chain_exit']},ensure_ascii=False,indent=2)+'\n\n资格仅此固定旧状态初始动作评分工程：实际60前向和独立60精确重放均非收益或科学样本。原checkpoint与同头参数未更新，完整76M backbone10D只读扩展；fresh888同3D初始化后7零行。当前ordinal车槽置换等变性未证明；新头还没有科学训练或实际完整解码/C0/全部正式控制及预算接线。固定首状态全保留，不选成本、目标或权重。生产/核查共享原JAX，核查不导入新head/forward/runner/binding，不证明另一Transformer实现；NumPy只独立复算头算术。900秒工程监督与有界下载不是在线10秒全请求资格。\n'
    create(report,text)
    boundary=R/'reproduction/ACTUAL_INITIAL_ACTION_HEAD_EVIDENCE_20261012.json';create(boundary,dict(**scope,scope_sha256=sha(O/'ACTUAL_EXECUTION_SCOPE.json')))
    prior=R/'current_entries_before_action_head_full_replay_preparation_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_actual_initial_action_head_execution_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    changes=[]
    for x in entries:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(actual_action_head_execution_directory=str(O.relative_to(ROOT)),actual_action_head_execution_closed=True,actual_action_head_model_executed=scope['model_executed'],actual_action_head_engineering_qualified=bool(qualified),actual_action_head_initial_component_qualified=bool(qualified),actual_action_head_execution_scope=scope,next=next_action)
            w['reproduction']['actual_initial_action_head_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [实际初始动作头执行与完整独立重载核查](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    assert all(sha(A/x['name'])==x['sha256'] for x in manifest)
    create(O/'LOCAL_STAGE_RECORDING_RECEIPT.json',dict(scope_sha256=sha(O/'ACTUAL_EXECUTION_SCOPE.json'),report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),qualified=bool(qualified),old_FAIL_preserved=True))
    print(json.dumps({k:scope[k] for k in ['qualified','production_records','production_checks','production_failed','independent_checks','independent_failed','downloaded_files']}))

if __name__=='__main__':main()
