"""Archive only actual new chain evidence; never upgrade old or scientific gates."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/actual_score_chain_execution_root_v1';A=O/'downloaded'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    manifest=read(O/'DOWNLOAD_MANIFEST.json');reg=read(O/'EXECUTION_REGISTRATION.json')
    assert all(sha(A/x['name'])==x['sha256'] and (A/x['name']).stat().st_size==x['size'] for x in manifest)
    assert sha(A/'EXECUTION_REGISTRATION.json')==sha(O/'EXECUTION_REGISTRATION.json')
    current=json.loads(read(sorted(O.glob('DOWNLOAD_PREFLIGHT_*.json'))[-1])['stdout'])
    assert current['complete'] and not current['owned_processes']
    chain=read(A/'CHAIN_EXIT.json');run=read(A/'RUN_EXIT.json') if (A/'RUN_EXIT.json').exists() else None
    audit_exit=read(A/'AUDIT_RUN_EXIT.json') if (A/'AUDIT_RUN_EXIT.json').exists() else None
    result=read(A/'RESULT.json') if (A/'RESULT.json').exists() else None
    audit=read(A/'INDEPENDENT_SCORE_CHAIN_AUDIT.json') if (A/'INDEPENDENT_SCORE_CHAIN_AUDIT.json').exists() else None
    if audit:
        assert audit['checks_count']==len(audit['checks']) and audit['failed']==sum(not c['passed'] for c in audit['checks'])
    closures={}
    for prefix in ('','AUDIT_'):
        start=read(A/(prefix+'SOURCE_START.json'));end=read(A/(prefix+'SOURCE_END.json'))
        closures[prefix or 'PRODUCTION']=dict(files=len(start['files']),module_file_map=len(start['module_files']),
            file_identities_equal=start['files']==end['files'],module_map_equal=start['module_files']==end['module_files'],
            changed=end['changed'],new_files=end['new_files'])
    failures={n:read(A/n) for n in ['FAILURE.json','AUDIT_FAILURE.json','SUPERVISOR_FAILURE.json'] if (A/n).exists()}
    passed=bool(chain['exit_code']==0 and run and run['exit_code']==0 and audit_exit and audit_exit['exit_code']==0 and result and result['failed']==0 and audit and audit['failed']==0)
    summaries=[]
    for arm in ('pretrained_initial','random888_initial'):
        p=A/(arm+'_CHAIN_RESULT.json')
        if p.exists():
            x=read(p);summaries.append(dict(arm=arm,steps=len(x['steps']),status=x['status'],remaining=x['remaining'],
                incumbent_hard=x['incumbent']['hard'],candidate_hard=x['candidate']['hard'] if x['candidate'] else None,
                incumbent_cost=x['incumbent']['cost'],candidate_cost=x['candidate']['cost'] if x['candidate'] else None,
                candidate_selected=x['candidate_selected'],wall_seconds=x['wall_seconds'],parameters_unchanged=x['parameters_unchanged']))
    complete=passed and len(summaries)==2 and all(s['status']=='complete' for s in summaries)
    resource=json.loads(read(sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))[-1])['stdout'])
    scope=dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),raw_component_qualified=passed,
        complete_chain_component_qualified=complete,chains=summaries,
        new_encode_artifacts=len(list(A.glob('*_ENCODED.npz'))),new_decode_artifacts=len(list(A.glob('*initial_[0-9][0-9].npz'))),
        independent_checks=audit['checks_count'] if audit else None,independent_failed=audit['failed'] if audit else None,
        independent_new_context_replays=audit['new_context_full_replays'] if audit else None,
        independent_candidate_rows=audit['candidate_rows'] if audit else None,run_exit=run,audit_exit=audit_exit,chain_exit=chain,
        max_abs_numpy_head_error=audit['max_abs_numpy_head_error'] if audit else None,actual_workload_closures=closures,
        failures=failures,initial_available_ram_gib=resource['available_ram_gib'],initial_cpu15_busy_percent=resource['cpu15_busy_percent'],
        actual_owned_tree_ended=True,downloaded_files=len(manifest),downloaded_bytes=sum(x['size'] for x in manifest),
        optimizer_steps=0,old_initial_component_QA_repeated=False,learned_output_qualified=False,formal_controls_qualified=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,shared_MD_uploaded=False)
    create(O/'ACTUAL_CHAIN_COMPONENT_SCOPE.json',scope)
    if passed:
        description='两臂固定旧key[12,3]各25插入完整动作链工程核查通过，%d新encode/%d新decode，%d候选行、独立%d项0失败；完整链组件资格%s。两新候选均hard但成本高于原计划，均回退；预训链计时11.183秒、随机链4.786秒，不等完整10秒请求合格。'%(scope['new_encode_artifacts'],scope['new_decode_artifacts'],audit['candidate_rows'],audit['checks_count'],str(complete))
        next_action='New actual_score_chain_execution_root_v1 actual initial-model chain evidence completed; do NOT repeat this engineering run/replay or closed initial68/54813. Read ACTUAL_CHAIN_COMPONENT_SCOPE for exact completion/status/C0/fallback and timing scope; initial-head untrained arms are not scientific controls. Next prospective full abortable10s request plus proper controls/equal-timeMaskCOiteration; formal trained same-initCE/task-trainedrandom/strongdynamic/science closure before new family3/data. Pickup terminal64/pendingempty/no81. Stable online gain not established.'
    else:
        description='新实际动作链原CHAIN_EXIT=%d，保留实际源/输出/FAILURE/AUDIT_FAILURE与原退出；独立项数%s/失败%s，不补缺项或称PASS。'%(chain['exit_code'],str(scope['independent_checks']),str(scope['independent_failed']))
        next_action='New actual_score_chain_execution_root_v1 failed qualification. Actual owned tree ended and all original outputs downloaded. First inspect ACTUAL_CHAIN_COMPONENT_SCOPE and exact FAILURE/AUDIT_FAILURE; no blind rerun, old firstforward60/68/54813 and all closedQA remain protected. Any engineering recovery requires new exclusive prospective registration with same fixed input/arm/head/parameters, preserves original failure and avoids completed production if only checker failed. Family2FAILclosed/family3notstarted; pickup terminal64/pendingempty/no81. Stable online gain not established.'
    note='\n> 2026-10-12 配送实际动作链接线：'+description+' 原登记SHA'+scope['registration_sha256']+'；两份旧00评分原件仅绑定复用，不重跑旧首前向。原预训initial/fresh888同headseed20261012、0更新、固定旧首项，不据标签/成本选头/模型。真实CPU15/RAM>=8GiB核查后独占PY/JSON有界阶段，已核本人实际树退出并有界只读下载全部原件；完整C0仍原物理实现，独立模型仍共享原JAX/Transformer，不是另一物理或模型实现。未取得已训CE/任务随机/全科学依赖/整请求10秒/等耗时迭代/强动态或稳定在线增益，第2族FAIL封闭、第3族未启动。取货终局64/pending空无81；共享MD/GPU/最终20260927/28未用，旧FAIL/183/C1/缺ledger及13800原件保持。\n'
    report=E/'02_配送线/文档/结果与进度/配送_实际初始评分完整动作链接线与独立核查_2026-10-12.md'
    create(report,'# 实际初始评分完整动作链接线与独立核查\n\n'+note+'\n'+json.dumps(scope,ensure_ascii=False,indent=2)+'\n\n各臂成本与时间仅此固定旧工程输入的描述，不是新科学样本、模型选择或正向收益。900秒监督、预先具备两份首评分、参数/特征准备在外；不能升级整请求10秒资格或等耗时控制。按实际完成状态限定组件资格。旧首次网络WinError10013在资源查询前、无上传/启动，失败保留；自动审核后同一次限定派发。任何注册失败和核查失败原件不改写。\n')
    boundary=R/'reproduction/ACTUAL_SCORE_CHAIN_COMPONENT_EVIDENCE_20261012.json';create(boundary,dict(**scope,scope_sha256=sha(O/'ACTUAL_CHAIN_COMPONENT_SCOPE.json')))
    prior=R/'current_entries_before_actual_score_chain_preparation_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_actual_score_chain_execution_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    changes=[]
    for x in entries:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(
                actual_score_chain_execution_directory=str(O.relative_to(ROOT)),actual_score_chain_execution_ended=True,
                actual_score_chain_raw_component_qualified=passed,actual_score_chain_complete_component_qualified=complete,
                actual_score_chain_component_scope=scope,next=next_action)
            w['reproduction']['actual_score_chain_component_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [实际初始评分完整动作链接线核查](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(scope_sha256=sha(O/'ACTUAL_CHAIN_COMPONENT_SCOPE.json'),report_path=str(report),
        report_sha256=sha(report),boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),raw_component_qualified=passed))
    print(json.dumps(dict(raw_component_qualified=passed,complete_chain_component_qualified=complete,chains=summaries,
        independent_checks=scope['independent_checks'],independent_failed=scope['independent_failed'],failures=list(failures))))

if __name__=='__main__':main()
