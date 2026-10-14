"""Archive narrow actual trace evidence without replay or changing old FAIL."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/phase_cleanup_boundary_engineering_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    audit=read(O/'INDEPENDENT_TRACE_AUDIT.json');assert audit['checks']==102 and audit['failures']==['Producer actual source start/end']
    result=read(O/'RESULT.json');receipt=read(O/'ACTUAL_CLEANUP_RECEIPT.json')
    assert not result['source_unchanged'] and receipt['worker_reaped'] and receipt['actual_worker_returncode']==1
    assert read(O/'producer_SUPERVISOR_EXIT.json')['actual_returncode']==1 and read(O/'audit_SUPERVISOR_EXIT.json')['actual_returncode']==1
    for p in ['producer_SUPERVISOR_EXIT.json','audit_SUPERVISOR_EXIT.json']:assert read(O/p)['natural_exit'] is True
    diagnosis=read(O/'PRODUCTION_SOURCE_FAILURE_DIAGNOSIS.json')
    assert len(diagnosis['new_files'])==1 and next(iter(diagnosis['new_files'])).endswith('SHCore.dll') and not diagnosis['changed_files']
    src=read(O/'SOURCE_START.json');end=read(O/'SOURCE_END.json');a=read(O/'AUDIT_SOURCE_START.json');b=read(O/'AUDIT_SOURCE_END.json')
    assert a==b
    scope=dict(registration_sha256=sha(O/'TOOLS_REGISTRATION_R2.json'),original_producer_exit=1,original_audit_exit=1,
        original_audit_checks=102,original_audit_failures=1,raw_replay_failure_names=[],
        registered_barrier_failures=['Actual producer and child SHCore.dll missing starting identity'],
        raw_trace_and_cleanup_checks_passed=101,trace_kernel_raw_component_qualified=True,
        production_source_qualified=False,overall_component_pass=False,
        actual_cleanup_seconds=receipt['elapsed'],actual_worker_returncode=1,actual_worker_reaped=True,
        raw_archive_bytes=sum((O/x['file']).stat().st_size for x in result['cases']),raw_archive_count=len(result['cases']),
        production_files_start=len(src['files']),production_files_end=len(end['files']),production_module_maps=len(src['modules']),
        independent_files=len(a['files']),independent_module_maps=len(a['modules']),
        model_engine_parent_executed=False,model_executed=False,full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,family3_started=False,
        source_early_parent_launch_identity_guard_still_pending=True,old_QA_repeated=False,stable_online_gain=False,
        caveat='Only trace calls and reusable cleanup helper ran; parent v2 and actual model request remain source-only. No natural child end closure after kill; no unique original DLL load cause proven.')
    create(O/'INDEPENDENT_BOUNDARY_ADJUDICATION.json',scope)
    prior=read(R/'current_entries_before_phase_timing_wiring_20261012/UPDATE_IDENTITY.json')['files']
    assert len(prior)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in prior)
    backup=R/'current_entries_before_phase_cleanup_boundary_20261012';backup.mkdir(exist_ok=False)
    for x in prior:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    report=E/'02_配送线/文档/结果与进度/配送_遥测异常清理原始记录验收与依赖屏障失败_2026-10-12.md'
    text='''# 遥测异常清理原始记录验收与依赖屏障失败

新增独占phase_cleanup_boundary_engineering_root_v1，先封存新cleanup helper、parent v2副本及固定trace-only工程登记，再追加准确工具/60秒监督登记。原四逻辑案例产生五原buffer归档（实际kill同时保留worker与caller），此前maximum_buffers计数描述在执行前R2澄清；所有原登记保留。旧engine/head/consumer/filter/输入原字节与8+2合同未改。仅Python3.12.7/psutil5.9.0/CPU7的小型遥测组件，无NumPy/JAX模型、C0、solver、真实模型parent或旧请求/QA执行。

实际验证嵌套success、操作异常传播/status2、容量溢出显式失败及真实子进程open操作取消。父核实际PID/create_time/cmdline/owner，独占Popen进程句柄终止并wait/reap后才复制原worker buffer。cleanup helper保留caller异常记录、worker open原字节、归档并关闭句柄，末时钟在实际回收/归档/清理后；本取消清理段0.015060900012031198秒，实际kill returncode1单独保存，不写自然0。它不是模型8秒计算+2秒回收或完整现场请求验收；初始化/等待入口与清理段范围不同。

五原buffer合计525696字节已保存。另进程独立parser/schema、原字节SHA、直接字段/精确有理数算术和另用端点分段复算嵌套区间并集、13固定篡改拒绝、reap/外层钟边界与实际退出核验：原INDEPENDENT_TRACE_AUDIT为102项1FAIL，101其余项通过。原open仍open，不算完成；closed只遥测标记，不等源/候选/科学资格。原生产与独立监督均自然退出1，原独立FAIL/退出不改，不重复组件。

唯一失败是生产实际起止新增SHCore.dll：父196→197文件/88module映射，child开始196→实际open入口197，既有文件逐字无变化、无新增module；该DLL缺起始身份，不补造或放宽门。未唯一定位首次加载调用。独立200文件/89module映射起止一致，只有本trace/parser无模型负载闭包；kill没有child自然结束闭包。追加裁决分开raw_replay_failure_names为空与registered_barrier_failures，raw trace/helper组件可用证据与整体/生产源不合格分开，禁止把102/1或生产1写整体PASS。

新parent v2源码已接异常finally的同helper：已选输出不撤销，异常时终止/回收后保留原trace并记完整失败时钟。该模型parent尚未实际执行；进程创建/初始身份仍在原try之前，需要进一步接线复核，不能凭helper局部合格宣称所有异常路径已闭合。全新实际请求重放核查器、准确限定模型执行/监督传输、3080真实RAM>=8GiB/CPU15/本人树检查仍待。禁止盲跑parent/model草案、放宽8秒、样本预热追速度或重跑旧10/100/220与任何已完成QA。此次source FAIL先保存诊断，未重复trace生产/核查，也未改全局依赖。

正式已训同初始化CE/同任务随机/各控制、等耗时MaskCO迭代、独立强动态接拒、科学完整依赖与整请求预算仍缺。第2族FAIL封闭、第3族未启；取货权威终局64/callback80/pending空无81。旧FAIL/ledger/13800字节缺口、31必要门/183/C1保持，无服务器/GPU/共享MD/最终20260927/28。稳定可归因MaskCO核心在线增益仍未建立。
'''
    create(report,text)
    boundary=R/'reproduction/PHASE_CLEANUP_BOUNDARY_EVIDENCE_20261012.json';create(boundary,scope)
    note='\n> 2026-10-12 遥测异常清理实际组件：新phase_cleanup_boundary_engineering_root_v1先登记，实际nested/exception/overflow/open-writer取消，五原buffer525696字节，helper取消回收归档0.015060900012031198秒/实际kill1。独立102项1FAIL原件与监督自然1保持，101其余原字节/算术/13篡改/退出项通过；唯一失败实际父/child新增SHCore.dll缺起始身份，不补造/重跑。独立200文件89map闭合仅trace负载，整体源/科学/预算仍未合格。parent v2源码异常清理已接helper但真实模型parent未执行/早期launch保护待；下一完成未闭合源接线与完整新请求核查准确注册，不重跑旧QA，不放宽8秒/样本预热。第2族封闭、第3族未启，取货终局64/pending空无81，稳定可归因在线增益未建立。\n'
    next_action='Trace-only phase_cleanup_boundary_engineering_root_v1 completed with original producer/audit natural exit1: independent102 checks1 source-gate failure, raw101 passed; parent and child gained SHCore.dll without starting seal, old FAIL retained and no kernel rerun. Five original buffers and actual open writer kill1/reap retained, helper cleanup.015060900012031198s only narrow scope. New parent v2 source uses helper but model/parent not run, early process creation/identity still outside try needs fix. Next preserve this source failure; seal all actually loaded native dependencies before any new workload, finish parent early-launch cleanup and complete new actual request replay checker/exact limited model registration/supervision/transport and actual3080 RAM>=8GiB/CPU15 owner tree. Do not repeat completed trace/kernel/parser or old10/100/220, relax8s, sample warmup or select by cost. Fullscientific controls/equal-time/strong dynamic/fullrequest remain missing, family2closed/family3notstarted, pickupterminal64/pendingempty/no81.'
    changes=[]
    for x in prior:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(phase_cleanup_boundary=scope,next=next_action)
            w['reproduction']['phase_cleanup_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [遥测异常清理原件验收与源屏障失败](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    shutil.copy2(Path.home()/'.codex/automations/maskco/automation.toml',O/'AUTOMATION_BEFORE_BOUNDARY.toml')
    addition='\n2026-10-12配送遥测异常清理实际组件及源屏障失败：delivery/phase_cleanup_boundary_engineering_root_v1事前固定trace-only登记和准确工具/60秒监督，新cleanup helper及parent v2源码保持原输入/模型/方法/8+2。仅本地Python3.12.7/psutil5.9.0/CPU7实际nested成功/异常status2/容量overflow/真实open writer取消，0模型/parent workload/NumPy/JAX/C0/solver/旧QA。核实际PID/create_time/cmdline/owner，Popen自身handle kill/wait后保存五原buffer525696字节、caller异常与worker open状态，helper实际回收归档清理.015060900012031198秒/实际kill returncode1；不是全模型10秒。原生产监督自然1、独立监督自然1，原INDEPENDENT_TRACE_AUDIT102项1FAIL保持，101其它raw字段SHA/独立有理数区间并集/13篡改拒绝/时钟及实际退出通过。唯一FAIL生产父196→197文件88map与child起始196→open入口197新增SHCore.dll缺起始身份，无既有字节变化/新增module，未定位唯一加载调用；源不合格不补造/放宽/重跑。独立200文件89map起止闭合仅本无模型负载，kill无自然child结束闭包。追加裁决raw_replay_failure_names为空/registered source failure分开，不能把原102/1或exit1写整体PASS。四逻辑案例五buffer原计数在数据前R2澄清，原登记保留。parent v2源码异常finally已接同helper但未运行实际parent/model；早期进程创建/身份在try之前仍须接线复核，helper局部资格不升级全部异常/请求。下一保存本源FAIL，完善实际native起始封存/早期launch清理及完整新实际请求核查准确限定登记监督传输，使用前3080 RAM>=8GiB/CPU15/本人树；不重复本trace/kernel/parser或旧10/100/220及完成QA，不放宽8秒/旧样本预热/按成本选头。正式已训同初始化CE/同任务随机/各控制/等耗时迭代/独立强动态/科学全依赖和全请求仍缺，第2族FAIL封闭、第3族未启；取货终局64/callback80/pending空无81。新报告配送_遥测异常清理原始记录验收与依赖屏障失败_2026-10-12.md、reproduction新PHASE_CLEANUP_BOUNDARY_EVIDENCE_20261012.json和16入口WORK_STATUS先存旧字节更新；全部旧FAIL/ledger/13800/31/183/C1保持，无服务器/GPU/共享MD/最终种子。稳定可归因核心在线增益未建立，maskco ACTIVE30分钟，阶段失败可通知，之后常规未变静默。'
    create(O/'AUTOMATION_ADDITION.json',dict(addition=addition))
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(entries=16,source_failure_preserved=True,old_QA_repeated=False,report_sha256=sha(report),boundary_sha256=sha(boundary),tool_sha256=sha(__file__)))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    print('Original102/1 and source failure retained; raw101 separated;16 entry backups updated.')
if __name__=='__main__':main()
