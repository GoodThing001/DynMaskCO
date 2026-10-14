"""Record original bounded request failure and narrow audit recovery; no runtime."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/timed_actual_request_execution_root_v1';A=O/'downloaded'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    reg=read(O/'EXECUTION_REGISTRATION.json');manifest=read(O/'DOWNLOAD_MANIFEST.json')
    assert sha(O/'EXECUTION_REGISTRATION.json')=='737325090b342bf714cabee8fffefd4b6262261249850ad19356d4bc1ce3acf8'
    assert all(sha(A/x['name'])==x['sha256'] and (A/x['name']).stat().st_size==x['size'] for x in manifest)
    assert all(sha(A/n)==sha(O/n)==h for n,h in reg['package_files'].items())
    assert sha(A/'EXECUTION_REGISTRATION.json')==sha(O/'EXECUTION_REGISTRATION.json')
    pre=json.loads(read(sorted(O.glob('DOWNLOAD_PREFLIGHT_*.json'))[-1])['stdout']);assert pre['complete'] and not pre['owned_processes']
    result=read(A/'RESULT.json');audit=read(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json');recovery=read(O/'INDEPENDENT_STAGE_PATH_MEMBERSHIP_RECOVERY_R2.json')
    assert audit['checks_count']==18647 and audit['failed']==24 and recovery['checks_count']==220 and recovery['failed']==0
    assert recovery['original_audit_sha256']==sha(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json')
    run=read(A/'RUN_EXIT.json');ae=read(A/'AUDIT_RUN_EXIT.json');chain=read(A/'CHAIN_EXIT.json')
    assert run['exit_code']==0 and ae['exit_code']==chain['exit_code']==1
    rows=[]
    for x in result['receipts']:
        tag=x['arm']+'_fault'+str(x['fault_stage']);root=A/(tag+'_worker');q=root/'request';s=read(root/'SOURCE_START.json')
        delay=root/'FAULT_DELAY_ENTERED.json';st=read(root/('STAGE_%d.json'%x['fault_stage'])) if x['fault_stage'] and (root/('STAGE_%d.json'%x['fault_stage'])).exists() else None
        entry=read(delay) if delay.exists() else None;end=root/'FAULT_SOURCE_AT_ENTRY.json';pe=read(end) if end.exists() else None
        rows.append(dict(tag=tag,elapsed=x['elapsed'],actual_stage=x['actual_stage'],selected=x['selected'],
            original_plan_retained=x['selected_routes']==read(O/'CHAIN_INPUT.json')['initial'],worker_reaped=x['worker_reaped'],
            actual_worker_returncode=x['actual_worker_returncode'],natural_exit=x['natural_exit'],kill_clock=x['kill_clock'],reap_clock=x['reap_clock'],
            completed_scores=len(list(q.glob('[0-9][0-9].npz'))),completed_transitions=len(list(q.glob('[0-9][0-9].json'))),
            complete_chain_stop=(q/'CHAIN_STOP.json').exists(),candidate_C0=(q/'C0_SELECTION.json').exists(),
            actual_fault_delay_entry=entry,stage_marker_to_delay_seconds=entry['clock']-st['clock'] if entry and st else None,
            partial_files=[str(p.relative_to(root)) for p in root.rglob('*.part')],
            start_files=len(s['files']),start_module_map=len(s['module_files']),
            fault_entry_files=len(pe['files']) if pe else None,fault_files_equal=s['files']==pe['files'] if pe else None,
            fault_map_equal=s['module_files']==pe['module_files'] if pe else None))
    ast=read(A/'TIMED_AUDIT_SOURCE_START.json');aend=read(A/'TIMED_AUDIT_SOURCE_END.json')
    resource=json.loads(read(sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))[-1])['stdout'])
    scope=dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),run_exit=run,audit_exit=ae,chain_exit=chain,
        original_independent_checks=audit['checks_count'],original_independent_failures=audit['failed'],original_independent_successful_items=18623,
        narrow_path_recovery_checks=220,narrow_path_recovery_failed=0,original_FAIL_rewritten=False,
        combined_persisted_raw_replay_qualified=True,source_identity_of_original_audit=sha(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json'),
        requests=rows,completed_scores=sum(x['completed_scores'] for x in rows),completed_transitions=sum(x['completed_transitions'] for x in rows),
        independent_actual_saved_forward_replays=audit['new_timed_evidence_forward_replays'],max_abs_numpy_head_error=audit['max_abs_numpy_head_error'],
        unexercised_faults=audit['unexercised_faults'],missing_completed_source_end=audit['missing_completed_source_end'],
        actual_bounded_cancellation_and_fallback_observed=True,all_reaped=result['all_reaped'],all_within_total_window=result['all_within_total_window'],
        selected_count=result['selected_count'],all_registered_faults_exercised=False,complete_chain_within_budget=False,
        complete_all_request_dependency_closure=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,learned_output_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        audit_closure=dict(files=len(ast['files']),module_map=len(ast['module_files']),files_equal=ast['files']==aend['files'],module_map_equal=ast['module_files']==aend['module_files']),
        downloaded_files=len(manifest),downloaded_bytes=sum(x['size'] for x in manifest),all_downloaded_original_hashes_match=True,actual_owned_tree_ended=True,
        available_ram_gib=resource['available_ram_gib'],cpu15_busy_percent=resource['cpu15_busy_percent'],cpu_affinity=[15],optimizer_steps=0,
        old_QA_repeated=False,production_repeated=False,full_independent_audit_repeated=False,shared_MD_uploaded=False,GPU_used=False,
        limitation='Public prepared snapshot only, caller synchronous copy/SHA/JSON/archive/cleanup and receipt-file creation outside elapsed. No live vehicle installation, arbitrary native/parent/load/slow-disk preemption, full scientific controls, equal-time iteration or stable online benefit.')
    assert len(rows)==10 and scope['completed_scores']==100 and scope['completed_transitions']==98 and scope['selected_count']==0
    create(O/'ACTUAL_TIMED_REQUEST_SCOPE.json',scope)
    next_action='New timed_actual_request_execution_root_v1 completed10 unique fixed8+2s initial-model public snapshot requests. All cancelled/reaped8.041..8.100s and retained original plan; both normals only16/25 insertions. Original18647/24FAIL and audit/chain exit1 preserved; new220/0 narrow Windows relative-path membership evidence combines with18623 original successful items, not rewrittenPASS or model/C0 rerun. Only faults1/2 actual delay entered;3..8 unexercised,8 requests missing completed source end; complete chain/fullbudget/science remainFalse. Do NOT repeat10 production/100 actual saved-forward replay/narrow recovery or oldQA; no time extension/sample warmup/model or sample selection. Next prospectively diagnose evidence-IO/loaded-file check and actual model/chain scheduling cost boundaries from saved clocks, then design bounded architecture/control and complete formal trainedsameinitCE/tasktrainedrandom/equal-timeiteration/independentstrongdynamic/science closure before any family3/freshdata. Pickup terminal64/pendingempty/no81. Stable attributable online gain not established.'
    note='\n> 2026-10-12 配送实际模型计时请求：新timed_actual_request_execution_root_v1事前登记SHA'+scope['registration_sha256']+'，10个固定请求各执行8秒/回收预留2秒，实际8.040783–8.099843秒均回收/保留原计划、0选择。两个正常臂仅各16/25插入；全部100保存评分/98已完成转移独立复算。原独立18647项24FAIL/审计退出1/CHAIN_EXIT1保留，24仅Windows相对路径反斜杠与正斜杠字符串比较；新事前窄核验220项0失败与原18623成功项组合，不改原FAIL，不重跑模型/C0或完整核查。仅故障1/2实际进入延迟；故障3/4虽有完成标记但依赖快照采集未到延迟、5–8未到目标标记，六项未验；8请求缺结束闭包。完整链/整请求/科学/已训控制资格仍False。369原件93715960字节有界只读下载逐SHA一致，实际本人树已退出；随机16.npz.part残写与所有实际kill返回码1/原生产FAILURE保留，不称自然0。第2族FAIL封闭、第3族未启动；取货终局64/pending空无81，旧QA/183/C1/缺ledger/13800字节不改，无GPU/共享MD/最终种子使用，稳定可归因核心在线增益仍未建立。\n'
    report=E/'02_配送线/文档/结果与进度/配送_实际模型计时请求取消与阶段核验路径恢复_2026-10-12.md'
    prose='''# 实际模型计时请求取消与阶段核验路径恢复

本次只取得固定旧公共快照上实际初始模型请求取消、回收和原计划回退的组件证据。完整链与完整请求资格未通过，科学第3族未启动。

原预训initial/fresh888同headseed20261012、0更新、固定原首项key[12,3]/25masked，不复用旧00评分、不按标签或成本换样本。真实3080 CPU15/RAM11.6138GiB/负载0.8%核查后只上传独占PY/JSON。模型日初加载及公共输入准备在外，没有样本前向或C0预热。十请求每次10秒窗包含输入复制、实际10D/原incumbent C0/encode/当前context模型评分与候选过滤regret、部分链保存以及父取消回收清理；receipt文件创建仍在elapsed外，不包括现场私有事件/车辆安装。

两正常请求都只完成16次插入，仍有9个masked客户待插。八故障请求只实际进入输入解码入口和特征完成入口的12秒延迟；这两次完成标记至延迟分别约7.78/7.49秒，包含全部8896加载文件快照与保存。该证据采集显著消耗固定8秒窗口，但不是链超时唯一原因定位；不能扣除此耗时或延长预算升级资格。encoder/首评分故障虽到标记但没有延迟入口原件，其余四故障没有到目标标记。全部无返回输出选择、保留完整原计划；100已存前向和98转移只是新工程工作负载，不是新科学数据/已训控制/收益。随机臂第17评分NPZ半写文件原样留存，不补完成内容。

原独立核查原checkpoint234叶/76263424参数、fresh888、同seed四叶196929头、可见10D与当前临时锚点、100实际已保存完整前向/cache/JAX评分和部分链均按原注册复算。2e-5仅头跨库算术界限，实际最大1.0132789611816406e-6；参数/特征/latent/JAX评分仍精确。原18647项中18623成功；24失败来自检查器把Windows记录的request\\文件名与request/文件名直接比较。新独占本地事前R2只重现这24字符串失败并按实际Windows相对路径、已有文件SHA和原入口时钟核验，220项0失败，不执行模型、C0、求解器或原完整审计。组合证据不把原结果改成PASS，原审计及CHAIN退出1保持。原共享Transformer/JAX/C0仍不是另一模型或物理实现。

仅故障1/2开始至实际入口8896文件/1027映射闭合，另8请求没有完成SOURCE_END或故障入口快照；被kill不补自然结束闭包。独立9242文件/1017映射起止一致仅本新部分工作负载，不能替代完整科学控制链。10个worker均实际被父终止并回收，返回码1保存；监督driver自然0、独立自然1、CHAIN1，三者不得混写。

下一步只能从已存时钟、实际保存链与输入定位请求架构和证据I/O的耗时边界，再另事前设计有限工程。禁止重跑本十请求、100保存前向核查、R2或旧QA，禁止扩大8秒/预热同样本追速度、依据成本或标签挑模型头。正式已训同初始化CE、同任务训练随机、各控制、等耗时MaskCO迭代、独立强动态接拒、完整科学依赖与全请求资格仍待；全部屏障后才另登记有限新鲜科学族。取货已权威终局，无callback81或旧未决单。
'''
    create(report,prose+note+'\n'+json.dumps(scope,ensure_ascii=False,indent=2)+'\n')
    boundary=R/'reproduction/TIMED_ACTUAL_REQUEST_EVIDENCE_20261012.json'
    create(boundary,dict(**scope,scope_sha256=sha(O/'ACTUAL_TIMED_REQUEST_SCOPE.json')))
    prior=R/'current_entries_before_timed_request_independent_preparation_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_timed_actual_request_execution_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    changes=[]
    for x in entries:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(
                timed_actual_request_execution_directory=str(O.relative_to(ROOT)),timed_actual_request_execution_ended=True,
                timed_actual_request_scope=scope,independent_timed_checker_executed=True,
                timed_request_combined_persisted_raw_replay_qualified=True,timed_request_complete_chain_within_budget=False,next=next_action)
            w['reproduction']['timed_actual_request_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [实际模型计时请求取消与窄核验](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(scope_sha256=sha(O/'ACTUAL_TIMED_REQUEST_SCOPE.json'),report_sha256=sha(report),boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),original_FAILED_results_preserved=True,science_qualified=False))
    print(json.dumps(dict(entries=len(changes),files=len(manifest),original_independent_failed=24,narrow_recovery_failed=0,selected=0,full_request_budget_qualified=False)))
if __name__=='__main__':main()
