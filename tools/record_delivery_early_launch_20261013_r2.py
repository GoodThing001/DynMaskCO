"""Seal saved-only evidence first, then record this bounded lifecycle stage."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
V1=R/'delivery/early_launch_lifecycle_root_v1'
V2=R/'delivery/early_launch_lifecycle_root_v2'
V3=R/'delivery/early_launch_lifecycle_root_v3'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def create(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def seal():
    inputs=[V1/'ENGINEERING_REGISTRATION.json',V1/'RESULT.json',V1/'producer_SUPERVISOR_EXIT.json',
        V2/'ENGINEERING_REGISTRATION.json',V2/'RESULT.json',V2/'INDEPENDENT_LIFECYCLE_AUDIT.json',
        V2/'producer_SUPERVISOR_EXIT.json',V2/'audit_SUPERVISOR_EXIT.json',
        V3/'ENGINEERING_REGISTRATION.json',V3/'RESULT.json',V3/'INDEPENDENT_LIFECYCLE_AUDIT.json',
        V3/'producer_SUPERVISOR_EXIT.json',V3/'audit_SUPERVISOR_EXIT.json']
    for name in ('mapping_failure','log_collision','popen_failure'):
        inputs += [V1/(name+'_ACTUAL_HANDLE_OBSERVATION.json'),
            V1/'cases'/name/(name+'_fault0_CALLER_FAILURE_AFTER_CLEANUP.json')]
    create(V3/'RECORDING_INPUT_REGISTRATION_R2.json',dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tool_sha256=sha(Path(__file__)),files={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        saved_only_scope='Three first-run pre-child startup boundaries; no producer/model/trace/solver execution',
        full_request_budget_qualified=False))
    print('Saved-only recording inputs sealed; no workload replay.')


def record():
    reg=read(V3/'RECORDING_INPUT_REGISTRATION_R2.json')
    assert sha(Path(__file__))==reg['tool_sha256']
    assert all(sha(ROOT/p)==h for p,h in reg['files'].items())
    a2=read(V2/'INDEPENDENT_LIFECYCLE_AUDIT.json');a3=read(V3/'INDEPENDENT_LIFECYCLE_AUDIT.json')
    assert a2['checks']==379 and a2['failures']==['Actual producer full component closure']
    assert a3['checks']==295 and a3['failures']==[]
    checks=[]
    def check(n,v):checks.append(dict(name=n,passed=bool(v)))
    expected={'mapping_failure':('OSError',2),'log_collision':('FileExistsError',4),'popen_failure':('FileNotFoundError',4)}
    for name,(typ,count) in expected.items():
        raw=read(V1/(name+'_ACTUAL_HANDLE_OBSERVATION.json'))
        receipt=read(V1/'cases'/name/(name+'_fault0_CALLER_FAILURE_AFTER_CLEANUP.json'))
        check(name+': original actual exception',raw['original_exception']['type']==typ)
        check(name+': exact acquired real maps',raw['acquired_mapping_count']==count)
        check(name+': actual closed maps',raw['mapping_closed']==[True]*count)
        check(name+': no actual child',raw['launched_child_count']==0 and not raw['child_leaked_after_parent'])
        check(name+': all acquired named handles closed',all(v is None or v is True for v in receipt['handles_closed'].values()))
        check(name+': no fabricated worker',receipt['child_launched'] is False and receipt['worker_trace_sha256'] is None)
        check(name+': original exception retained',raw['original_exception']['traceback'].endswith(receipt['traceback'].split('\n',1)[1]))
        check(name+': no science upgrade',receipt['qualification'] is False and receipt['full_request_budget_qualified'] is False)
    assert all(c['passed'] for c in checks)
    create(V3/'SAVED_V1_PRECHILD_BOUNDARY_AUDIT.json',dict(checks=len(checks),failures=[],items=checks,
        production_rerun=False,original_v1_source_fail_preserved=True,model_executed=False))
    v3=read(V3/'RESULT.json');case=v3['cases']['identity_interrupt']
    start=read(V3/'SOURCE_START.json');end=read(V3/'SOURCE_END.json')
    assert not any(v3['source_delta'].values()) and start==end
    scope=dict(registered_new_boundaries=8,first_three_saved_only_checks=24,
        v1_original_producer_exit=1,v1_original_source_fail=True,
        v1_four_child_entries_not_reached=True,v1_marker_slice_error_retained=True,
        v2_original_producer_exit=1,v2_original_audit_exit=1,
        v2_independent_checks=379,v2_independent_failures=1,
        v2_original_failure='bounded_phase_trace_v1 imported after starting module map; no new or changed file bytes',
        v2_four_actual_child_boundaries_verified=True,
        v3_new_boundary='SystemExit during child identity after actual startup, before input request',
        v3_independent_checks=295,v3_independent_failures=0,
        v3_producer_exit=0,v3_audit_exit=0,v3_actual_child_returncode=case['child_returncodes'][0],
        v3_actual_child_reaped=True,v3_source_files=len(start['files']),v3_source_modules=len(start['module_files']),
        v3_actual_start_end_equal=True,v3_startup_window_seconds=case['elapsed'],
        parent_startup_allocations_now_inside_try=True,
        before_child_cleanup_supported=True,original_exception_survives_receipt_failure=True,
        explicit_SHCore_and_ntmarta_preload=True,old_DLL_loader_cause_identified=False,
        model_engine_bytes_unchanged=sha(V3/'delivery_phase_actual_engine_v1.py')==sha(V1/'delivery_phase_actual_engine_v1.py'),
        actual_model_parent_executed=False,model_executed=False,old_QA_repeated=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        server_upload=False,shared_MD_uploaded=False,final_seeds_used=False,
        caveat='Three old saved pre-child cases and four child recovery observations have original source FAIL; v3 only one new interrupt component has full starting/ending source identity. No combined old FAIL converted to PASS; no full model timing or online utility.')
    create(V3/'STAGE_ADJUDICATION.json',scope)
    report=E/'02_配送线/文档/结果与进度/配送_启动资源生命周期修正与新故障验收_2026-10-13.md'
    text='''# 配送启动资源生命周期修正与新故障验收

本轮实际推进配送调用端的早期启动资源保护。新parent v3将共享内存、日志、Popen及进程身份查询全部放入try/finally；新cleanup v2支持child/句柄尚未创建，只有exact Popen wait返回后才读取worker原字节，逐项关闭已获取资源。caller trace归档失败时仍释放memoryview；清理回执写入失败不会替代原始启动异常。已选结果不撤销。新模型worker在engine开始快照之前显式预加载SHCore.dll/ntmarta.dll并保留句柄；engine/head/解码/C0原字节未改，未执行模型请求。

三个新目录均先登记再执行。v1真实验证第三共享映射失败、stdout原文件碰撞及不存在解释器；24项另进程只读保存结果核查通过，不重新运行这三项。四个子进程路径在v1因测试夹具标记长度写错而未到达，实际child退出1原件保留。v1还出现ntmarta.dll、ast及action_head_runtime起始身份缺失，生产整体退出1保留；资源回收观察不能冒充全源资格。

v2修正夹具长度、包导入顺序并显式预加载已观察native文件，仅运行此前未到达的四项：child身份查询PermissionError、真实初始化自然退出7、ready后在输入请求前终止、清理回执写入故障。四项均回收实际child、关闭实际映射并保留原异常；最后一项保留PermissionError与archive故障note。独立379项1FAIL，原生产与审计自然退出1保持；唯一失败为bounded_phase_trace_v1在起始module map后导入，所有文件字节不变。不得改写原FAIL或重跑四项追求通过。

v3把遥测模块导入移到起始快照之前（不调用其kernel），另登记一个未运行过的SystemExit启动中断边界。实际child已到达ready/原marker保存、请求序号仍0；parent原SystemExit保留、child经本人Popen句柄kill/wait回收、共享内存与日志关闭。生产和独立监督均自然退出0；独立295项零失败，生产/独立本组件实际起止files及module maps一致。被kill的fixture仍实际返回1，不写自然0；child没有自然结束闭包。v3只支持这一新中断组件的源封存与生命周期，不合并升级旧v1/v2失败。

没有运行JAX/NumPy模型、完整模型parent、遥测kernel/parser旧QA、C0/solver、训练或新科学数据；未使用服务器/GPU、最终取货20260927/28或共享MD同步。原本机进程列表查询AccessDenied，不据此声称无其他进程；只终止自己直接启动的fixture句柄。原配送第2族FAIL封闭、第3族未启，取货原终局64/callback80/pending空不重跑。

下一步按v3源码接入全新真实模型计时请求的独立核查器和准确限定执行登记/监督/传输，实际3080 RAM>=8GiB/CPU15/本人树核验后才执行；8秒计算+2秒回收、完整source/参数放行检查保持计时内，不预热同样本或按成本选头。正式已训同初始化CE/同任务随机、等耗时MaskCO迭代、独立强动态与科学完整依赖仍未合格。稳定可归因MaskCO核心在线正向增益尚未建立。
'''
    create(report,text)
    note='\n> 2026-10-13 配送早期启动资源修正：新parent v3/cleanup v2覆盖尚未创建child/句柄、真实启动身份异常及回执写入异常。v1三个pre-child保存结果24只读项通过，四子路径因夹具错误未到达；v2只补四未到达路径，独立379/1模块起始身份FAIL及原退出1保持。v3另登记新SystemExit中断，生产/独立自然0、295/0、实际起止依赖一致，仅启动组件；本人child实际kill1/reap、全部资源关闭。0模型/C0/训练/旧QA重跑，第2族封闭/第3族未启、取货终局64不重跑，稳定在线增益未建立。下一真实模型完整计时请求准确核查/登记与科学控制。\n'
    entries=[E/'项目当前状态.md',E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
        E/'02_配送线/README.md',E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup=R/'current_entries_before_early_launch_lifecycle_20261013';backup.mkdir(exist_ok=False)
    changes=[]
    next_action='New early_launch_lifecycle_root_v3 parent/cleanup now protect allocations and Popen/identity from startup. Three v1 saved pre-child boundaries have24 narrow read-only checks, original v1 source/fixture FAIL preserved; four previously unreached v2 boundaries have379/1 bounded module-start FAIL kept; one new v3 SystemExit actual startup has295/0 and exact source start/end, parent fixture only. Do NOT repeat8 startupcases or old trace/model/QA. Next seal new actual-model timed replay checker for v3 parent/worker and native initialization, exact limited new-model execution registration/supervisor/transport; check actual3080 RAM>=8GiB CPU15 owner tree before new dispatch. Preserve8+2/mandatory gates inside request/no sample warmup. Model parent not yet run; full request/scientific trained CE/task random/strongdynamic/equal-time iterative gates false; family2closed/family3notstarted; pickupterminal64/pendingempty/no81.'
    for p in entries:
        if not p.is_file():continue
        raw=p.read_bytes();q=backup/p.relative_to(E);q.parent.mkdir(parents=True,exist_ok=True);q.open('xb').write(raw)
        if p.suffix=='.json':
            w=json.loads(raw.decode('utf-8-sig'));w['updated']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
            w['delivery'].update(early_launch_lifecycle=scope,next=next_action)
            w['latest_delivery_stage_receipts_root']=str(V3.relative_to(ROOT)).replace('\\','/')
            new=json.dumps(w,ensure_ascii=False,indent=2)+'\n'
        else:
            new=note+raw.decode('utf-8-sig')
            if p==E/'02_配送线/README.md':new+='\n- [启动资源生命周期修正与新故障验收](文档/结果与进度/'+report.name+')\n'
        assert p.read_bytes()==raw,'Concurrent entry change'
        p.write_text(new,encoding='utf-8',newline='\n')
        changes.append(dict(path=str(p.relative_to(E)),before_sha256=sha(q),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    create(R/'reproduction/EARLY_LAUNCH_LIFECYCLE_BOUNDARY_20261013.json',scope)
    create(V3/'LOCAL_RECORDING_RECEIPT.json',dict(entries=len(changes),report_sha256=sha(report),
        original_failures_preserved=True,model_executed=False,next=next_action))
    assert all(sha(ROOT/p)==h for p,h in reg['files'].items())
    print('Lifecycle stage recorded; original FAIL retained; no online-gain upgrade.')


if __name__=='__main__':
    {'seal':seal,'record':record}[sys.argv[1]]()
