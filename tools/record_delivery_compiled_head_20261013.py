"""Record original candidate FAIL and diagnose saved closure metadata only."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'research/review_3080_20261010'
STAGE=TREE/'delivery/compiled_head_identity_root_v1'
RAW=STAGE/'downloaded'
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def create(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def seal():
    files=[STAGE/'EXECUTION_REGISTRATION.json',STAGE/'DOWNLOAD_MANIFEST.json']
    files+=list(STAGE.glob('DOWNLOAD_PREFLIGHT_*.json'))+list(STAGE.glob('RESOURCE_PREFLIGHT_*.json'))
    files+=sorted(p for p in RAW.rglob('*') if p.is_file())
    create(STAGE/'SAVED_RECORDING_REGISTRATION.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tool_sha256=sha(Path(__file__)),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        scope='Saved-only hashes/metadata/arithmetic and original strict/source failures; no candidate/head/backbone/model/C0/solver/audit rerun. No original missing starting identities reconstructed.'))


def record():
    reg=read(STAGE/'SAVED_RECORDING_REGISTRATION.json')
    assert sha(Path(__file__))==reg['tool_sha256'] and all(sha(ROOT/p)==h for p,h in reg['inputs'].items())
    manifest=read(STAGE/'DOWNLOAD_MANIFEST.json')
    for item in manifest:
        p=RAW/item['name'];assert p.stat().st_size==item['size'] and sha(p)==item['sha256']
    pre=[json.loads(read(p)['stdout']) for p in STAGE.glob('DOWNLOAD_PREFLIGHT_*.json')]
    assert pre and all(x['complete'] and not x['owned_processes'] for x in pre)
    r=read(RAW/'RESULT.json');a=read(RAW/'INDEPENDENT_TIMED_REQUEST_AUDIT.json')
    assert r['count']==34 and r['exact_equal_count']==0 and not r['strict_output_identity_qualified']
    assert a['checks_count']==9099 and a['failed']==1 and a['failure_names']==['independent_actual_loaded_closure']
    assert len(a['registered_barrier_failure_names'])==34
    start,end=(read(RAW/p) for p in ('TIMED_AUDIT_SOURCE_START.json','TIMED_AUDIT_SOURCE_END.json'))
    ps,pe=(read(RAW/p) for p in ('SOURCE_START.json','SOURCE_END.json'))
    assert ps['files']==pe['files'] and ps['module_files']==pe['module_files']
    assert not end['changed'] and end['new_files'] and start['module_files']==end['module_files']
    # Starting module map can contain imports performed later in closure() even
    # though the function captured its starting files before those imports.
    mapped_missing={name:path for name,path in start['module_files'].items() if path not in start['files'] and path in end['files']}
    helper=(RAW/'action_head_runtime_v1.py').read_text(encoding='utf-8')
    assert helper.index('for m in list(sys.modules.values())')<helper.index("for name in ('numpy','jax'")<helper.index('maps={}')
    diagnosis=dict(independent_new_file_count=len(end['new_files']),independent_changed_files=[],
        starting_module_map_paths_missing_starting_file_identity=mapped_missing,
        initial_files_collected_before_runtime_import_loop=True,module_map_collected_after_loop=True,
        audit_imports_only_numpy_psutil_before_first_closure=True,
        actual_start_missing_identity_not_repaired=True,production_and_audit_rerun=False,
        source_qualification=False,
        conclusion='Starting closure imports extra runtime libraries after initial sys.modules file capture. Saved start-map/file mismatch and frozen source order substantiate a snapshot-order defect; existing bytes unchanged does not restore their missing initial qualification.')
    create(STAGE/'SAVED_CLOSURE_ORDER_DIAGNOSIS.json',diagnosis)
    scope=dict(registration_sha256=sha(STAGE/'EXECUTION_REGISTRATION.json'),candidate_attempts=1,
        saved_inputs=34,exact_equal_outputs=0,strict_candidate_qualified=False,
        registered_score_identity_failures=34,max_abs_original_score_difference=max(x['max_abs_score_difference'] for x in r['records']),
        independent_checks=9099,independent_failures=1,independent_failure_names=a['failure_names'],
        independent_success_items=9098,independent_max_abs_numpy_head_error=a['max_abs_numpy_head_error'],
        numpy_bound_diagnostic_only=2e-5,production_source_start_end_equal=True,
        production_loaded_files=len(ps['files']),production_module_files=len(ps['module_files']),
        independent_loaded_files_start=len(start['files']),independent_loaded_files_end=len(end['files']),
        closure_diagnosis=diagnosis,raw_overall_qualified=False,
        candidate_operation_seconds=a['completed_operation_seconds'],
        candidate_operation_exact_seconds=a['completed_operation_exact_seconds'],
        compilation_sync_input_transfer_host_conversion_in_operation=True,
        original_second_arm_compiler_cache_reused=True,whole_request_time_comparison_qualified=False,
        run_exit=read(RAW/'RUN_EXIT.json'),audit_exit=read(RAW/'AUDIT_RUN_EXIT.json'),chain_exit=read(RAW/'CHAIN_EXIT.json'),
        download_files=len(manifest),download_bytes=sum(x['size'] for x in manifest),download_all_hashes_match=True,
        actual_owned_tree_exited_before_download=True,
        candidate_not_installed=True,original_backbone_executed=False,optimizer_steps=0,old_QA_repeated=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,learned_output_qualified=False,scientific_family3_started=False,
        stable_online_gain=False,GPU_used=False,shared_MD_uploaded=False,final_pickup_seeds_used=False)
    create(STAGE/'STAGE_ADJUDICATION.json',scope)
    report=EXT/'02_配送线/文档/结果与进度/配送_动作头编译调度候选与逐位一致门失败_2026-10-13.md'
    text='''# 配送动作头编译调度候选与逐位一致门失败

本轮完成一个实际调度实现与有限工程执行：新compiled_action_head_v1将原动作头表达式包成单个jax.jit函数，并对gather/三段均值矩阵乘法/concat/两层线性/tanh中间值加optimization_barrier。它仅是保留表达式意图，不能假定保留浮点位模式。原upstream/生产模型/失败/科学门均不修改。

执行前注册SHA：`%s`。使用全部34个已保存输入，两initial臂各step0..16原顺序；输入SHA、原四叶196929参数身份固定。只运行一个候选，每行一次调用，编译/参数输入转换/同步/host转换计入本head操作钟，不预热样本，不尝试第二变体，不按成本标签挑头。无原backbone、encode/decode、路线solver/C0、训练或新科学数据，正式第3族未启动。

严格评分逐位一致为0/34，34项原始放行失败全部保留，最大绝对差%.12g。独立NumPy头最大差%.12g仍在事前诊断2e-5以内，但该诊断不能替代逐位门；不接入原部署/动作链，不改旧门或按精度阈值重写FAIL。单头操作合计pretrained %.9f秒、random %.9f秒，第二臂复用同进程编译缓存；保存与依赖归档在head操作钟外，这些时间不能冒充完整请求、等耗时在线加速或正向效用。

原生产自然退出0、%.9f秒，独立自然退出1、%.9f秒，监督退出1保持。独立9099项1FAIL，9098其余原件/参数/输入/评分差值/NumPy/时钟/生产来源等项成功，整体审计不是PASS；34评分不一致门另列registered_barrier_failure_names。唯一审计FAIL为independent_actual_loaded_closure，原结果/源码/退出1不改。

另事前saved-only记录在不执行任何模型/原审计情况下核对源与起止清单：原helper closure先捕获sys.modules文件，随后导入JAX/Flax等额外库，最后才生成module map；此次checker开始只主动导入NumPy/psutil，首次closure内部导入额外依赖。实际起始map已有%d映射指向起始files未覆盖而末端已覆盖的文件；末端新增%d实际文件，既有字节无变、module map起止相同。这是具体封存顺序缺陷证据，不能补造原缺失起始哈希或升级本审计来源资格。原生产已在开始前导入head/JAX，%d文件/%dmodule起止逐字一致，仅本head候选负载。

确认实际本人树结束后，%d原件%d字节有界只读下载逐SHA一致。共享MD/88历史未上传、GPU及最终取货20260927/28未使用。取货仍权威终局64/callback80/pending空，不跑81或旧无模型QA；第2族FAIL封闭，正式训练/DEV/确认不重跑。

下一先在新事前工程中隔离编译候选与原表达式的中间值/归约差异，以及完善起始闭包预初始化；不得直接以2e-5接入、改评分门或重跑本34候选/9099审计追PASS。只有独立一致性和实际源资格通过，才接下一完整预算阶段。正式已训同初始化CE/同任务随机、独立强动态接拒、等耗时MaskCO迭代和科学完整依赖仍待。稳定、可归因MaskCO核心在线正向增益未建立。
'''%(scope['registration_sha256'],scope['max_abs_original_score_difference'],scope['independent_max_abs_numpy_head_error'],a['completed_operation_seconds']['pretrained_initial'],a['completed_operation_seconds']['random888_initial'],scope['run_exit']['wall_seconds'],scope['audit_exit']['wall_seconds'],len(mapped_missing),len(end['new_files']),len(ps['files']),len(ps['module_files']),len(manifest),scope['download_bytes'])
    create(report,text)
    next_action='compiled_head_identity_root_v1 implemented one optimization-barrier jit head over all34 saved latent/binding inputs,0 backbone/solver/C0/training. Strict exact output0/34 FAIL,max2.086162567e-7; NumPy2e-5 diagnostic NOT substitute, do not install candidate or relax gate. Producer0; independent9099/1 closure-order FAIL and chain1 preserved; saved-only start-map/files/source order diagnoses runtime imports after initial file capture, no old missing SHA repaired. Do NOT repeat34 candidate calls or9099 audit/old QA. Next preregister bounded intermediate arithmetic/compiled-vs-original reduction diagnosis (new observables, no backbone/utility/model selection) and closure preinitialization; then semantics-qualified scheduling and full8+2 end gates. Formal trained CE/task random/strongdynamic/equal-time iteration/scienceclosure false,family2closed/family3notstarted,pickupterminal64/pendingempty/no81.'
    note='\n> 2026-10-13 动作头编译调度候选实际收口：compiled_head_identity_root_v1一个jit+中间barrier候选，全部34保存输入无backbone/C0/训练；逐位0/34，最大差2.086162567e-7，严格放行FAIL不接部署。原独立9099/1起始依赖清单FAIL/退出1、chain1保持，9098其余证据通过不是整体PASS；saved-only实际map/files/源顺序定位closure内部导入在初始文件捕获之后，原缺始身份不补造。单头1.672821/0.177397秒含编译等但第二臂复用缓存，非完整请求或在线收益。下一新事前中间算术/归约诊断与起始闭包修正，不重跑34/9099/旧QA、不放宽评分门；正式科学/稳定增益False，第2族封闭/第3族未启、取货终局64不重跑。\n'
    entries=[EXT/'项目当前状态.md',EXT/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',EXT/'02_配送线/README.md',EXT/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup=TREE/'current_entries_before_compiled_head_identity_20261013';backup.mkdir(exist_ok=False);changed=[]
    for p in entries:
        raw=p.read_bytes();q=backup/p.relative_to(EXT);q.parent.mkdir(parents=True,exist_ok=True);q.open('xb').write(raw)
        if p.suffix=='.json':
            w=json.loads(raw.decode('utf-8-sig'));w['updated']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
            w['delivery'].update(compiled_head_candidate=scope,next=next_action)
            w['delivery']['active_compiled_head_stage']['status']='completed_strict_score_and_independent_source_FAIL_preserved_no_install'
            w['latest_delivery_stage_receipts_root']=STAGE.relative_to(ROOT).as_posix()
            new=json.dumps(w,ensure_ascii=False,indent=2)+'\n'
        else:
            new=note+raw.decode('utf-8-sig')
            if p==EXT/'02_配送线/README.md':new+='\n- [动作头编译调度候选与逐位门失败](文档/结果与进度/'+report.name+')\n'
        assert p.read_bytes()==raw
        p.write_text(new,encoding='utf-8',newline='\n')
        changed.append(dict(path=p.relative_to(EXT).as_posix(),before_sha256=sha(q),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changed,report=report.relative_to(EXT).as_posix(),shared_MD_uploaded=False))
    create(TREE/'reproduction/COMPILED_HEAD_STRICT_IDENTITY_FAILURE_EVIDENCE_20261013.json',scope)
    create(STAGE/'LOCAL_RECORDING_RECEIPT.json',dict(entries=len(changed),report_sha256=sha(report),next=next_action,raw_input_hashes_unchanged=all(sha(ROOT/p)==h for p,h in reg['inputs'].items())))
    print(json.dumps(dict(recorded=True,strict_candidate_qualified=False,independent_failures=1,new_file_count=len(end['new_files']),start_map_paths_without_start_file_hash=len(mapped_missing),stable_online_gain=False)))


if __name__=='__main__':{'seal':seal,'record':record}[sys.argv[1]]()
