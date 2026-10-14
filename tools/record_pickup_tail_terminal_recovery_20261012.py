"""Seal qualified components while keeping the original whole-run and audit failures."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup';O=P/'resident_day_tail_root_v1';A=P/'resident_tail_stream_audit_root_v3';T=P/'terminal_restore_root_v2';L=P/'resident_day_tail_launch_root_v1'
def main():
    audit=read(A/'INDEPENDENT_RESIDENT_TAIL_STREAM_R3.json');tr=read(T/'RESULT.json');ta=read(T/'INDEPENDENT_TERMINAL_RESTORE.json')
    assert audit['failed']==tr['failed']==ta['failed']==0
    assert read(O/'RUN_EXIT.json')['exit_code']==read(L/'EXIT.json')['exit_code']==1 and read(T/'RUN_EXIT.json')['exit_code']==0
    assert not audit['original_whole_run_PASS'] and not tr['whole_previous_run_PASS'] and not (O/'RESULT.json').exists() and not (O/'OWNER_STATE_64.json').exists() and not (O/'ACTUAL_SHARED_PAYLOAD_64.bin').exists()
    for directory,filename,field in [(O,'REGISTRATION.json','sources'),(A,'STREAM_AUDIT_REGISTRATION_R3.json','inputs'),(T,'REGISTRATION.json','sources'),(T,'AUDIT_REGISTRATION.json','inputs')]:
        assert all(sha(p)==h for p,h in read(directory/filename)[field].items())
    names={'qualify_pickup_resident_day_tail_v1.py','run_pickup_resident_day_tail_bounded_v1.py','audit_pickup_resident_tail_stream_r2.py','audit_pickup_resident_tail_stream_r3.py','qualify_pickup_terminal_restore_v2.py','audit_pickup_terminal_restore_v2.py'}
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for directory,items in [(T,['build_pickup_terminal_restore_v2.py','qualify_pickup_terminal_restore_v2.py','audit_pickup_terminal_restore_v2.py']),(A,['build_pickup_resident_tail_stream_audit_r3.py','audit_pickup_resident_tail_stream_r3.py'])]:
        for n in items:
            dest=directory/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(A/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True,previous_R2_actual_exit=1,previous_R2_footer_failed=True,no_production_replay=True))
    create(T/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True))
    create(T/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,independent_actual_exit=0,worker_actual_exit=tr['actual_worker_exit'],worker_natural_exit=True,original_producer_actual_exit=1,original_supervisor_actual_exit=1,original_worker_returncode_not_archived=True,owned_processes_active=active))
    _,decode=old.codecs();term=decode(read(T/'TERMINAL.json'));final=term['state'];restore=decode(read(T/'RESTORE_IN_REQUEST.json'));firstrestore=decode(read(O/'RESTORE_IN_REQUEST.json'));last=decode(read(O/'REQUEST_63.json'));hosts=[read(O/('HOST_'+str(i)+'.json')) for i in range(14,64)]
    assert len({h['pid'] for h in hosts})==1 and last['callback']==80 and last['order']==75 and tr['host']['ticket']['epoch']==64
    b=read(O/'SOURCE_START.json');ab=read(A/'STREAM_AUDIT_SOURCE_START_R3.json');tb=read(T/'SOURCE_START.json');tab=read(T/'AUDIT_SOURCE_START.json')
    facts=dict(new_submitted_requests=50,original_producer_exit=1,original_runtime_total_not_written=True,original_run_s=read(L/'EXIT.json')['wall_s'],independent_request_checks=audit['checks_count'],request_replay_failures=audit['failed'],registered_barrier_failures=audit['registered_barrier_failures'],reference_physical_events=audit['reference_physical_events'],original_intermediate_raw_event_ledger_present=False,normal_min=min(h['wall_s'] for h in hosts),normal_max=max(h['wall_s'] for h in hosts),first_restore_s=firstrestore['finished']-firstrestore['started'],terminal_runtime_checks=tr['checks_count'],terminal_independent_checks=ta['checks_count'],terminal_failed=0,terminal_wall_s=tr['host']['wall_s'],terminal_restore_s=restore['finished']-restore['started'],terminal_physical_events=tr['physics'],terminal_payload_bytes=tr['terminal_payload_bytes'],terminal_payload_sha256=sha(T/'RETAINED_PAYLOAD.bin'),source63_bytes=(O/'ACTUAL_SHARED_PAYLOAD_63.bin').stat().st_size,source63_sha256=sha(O/'ACTUAL_SHARED_PAYLOAD_63.bin'),clock=float(final['clock']),event=int(final['event_id']),replan=int(final['replan_count']),accepted=len(final['state']['accepted']),rejected=len(final['state']['rejected']),served_customers=int(final['served'].sum())-1,vehicle_statuses=[v.status for v in final['fleet']],return_arrivals=[float(v.return_arrival) for v in final['traces']],utility=float(term['evaluation']['utility']),production_files=len(b['files']),production_modules=len(b['module_files']),request_audit_files=len(ab['files']),request_audit_modules=len(ab['module_files']),terminal_files=len(tb['files']),terminal_maps=len(tb['module_files']),terminal_audit_files=len(tab['files']),terminal_audit_maps=len(tab['module_files']),original_whole_run_PASS=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False)
    create(T/'SEALED_COMPONENT_FACTS.json',facts)
    report=E/'01_取货线/文档/结果与进度/取货_五十揭示连续提交与终局超时后独立恢复_2026-10-12.md'
    create(report,f'''# 五十揭示连续提交与终局超时后独立恢复（2026-10-12）

## 裁决与资格

resident_day_tail_root_v1原整轮退出1，原FAILURE、RUN_EXIT、源码、已提交50次及未发布终局原件全部保留，不能写整体PASS。新流式独立R3对原callback31..80的{facts['independent_request_checks']}项复算0失败；新terminal_restore_root_v2仅恢复实际选定callback80/epoch63并执行终局，9运行/1164独立项0失败。组合是「同resident提交50次 + 另恢复终局」，不是51窗口同resident通过、日初单进程全80资格或完整科学运行。原中间逐物理推进原始累计事件档案未写出，该缺口仍False；完整逐请求私有提交状态、原检查点和全执行轨迹保留并复算。

## 原连续50次与未选终局

生产执行前登记，从lineage_header_cancel_root_v1调用端actual retained epoch13/callback30原42569字节SHA5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7恢复，未使用错误header14或取消order39副本。仅原库/200历史/80数据、父shared预置和旧epoch12预热在窗外，首31恢复{facts['first_restore_s']}秒在首10秒窗内、0接拒/评分/物理。实际callback31/order39至callback80/order75连续50次均由同resident的紧前已提交结果开始，序号14..63，各10秒窗（执行8/预留2）、{facts['normal_min']}..{facts['normal_max']}秒；首及中途不回收，不能声称每请求均实测回收。

实际当前-only own_uncond_dynamic/own_incumbent同计划、现场公共提取编码解码、原200历史共同K10每次20完整分支、原全C0认证编译/guard/actual plan_and_commit、检查点selector发布、父有界SHA/base/稳定selector及截止前唯一选择、事件ACK/原字节归档。50次均未换路线，无学习模型、新科学日或效用选样。实际选定epoch63/callback80/source raw{facts['source63_bytes']}字节SHA{facts['source63_sha256']}是新恢复唯一来源；clock15.865883827209473/event148/replan132。全部50次均有实际HOST、REQUEST、ATTEMPT、writer/shared/envelope及开始/计算结束/提交后加载身份。

原terminal64私有物理完成至event158、原完整80评价hard，但8秒截止前没有共享发布或调用端选择，不能算权威终局。监督实际{facts['original_run_s']}秒退出1，600秒上限未触发；父实际超时并回收本人resident，未保存的worker原返回码不补造。原私有writer122491字节SHAa9c479ba37f0b693dff413714e2bd5d21f5545c39cbdfd65429dbea92db9359a及TERMINAL/SOURCE_END64保留，不能作为新恢复输入。原程序在发布前同步编码包含50份响应和累计事件的OWNER_STATE_64；该档案未写出，而逐请求末段报文已超过18MB。累计JSON编码是具体候选瓶颈，未独立定位唯一耗时原因，也不据此改变原预算或FAIL。

## 独立检查器恢复

冻结原整轮审计器未运行。新R2改为50报文逐份流式核查并完成计算/写SOURCE_END，结果页引用不存在的生产RESULT.json触发FileNotFoundError，实际退出1；内存检查未归档、计数未知，不宣称全部通过。原源码/起止闭包/失败receipt保留。本人进程已退出后另事前注册R3新独占目录，同固定全部原输入，仅页末绑定实际RUN_EXIT/FAILURE、逐项落盘防止结果丢失；这是未合格审计器恢复，没有重跑50生产请求或旧已合格QA。

R3直接原策略和环境从同epoch13参考复算全部1000K10分支float.hex/NumPy分组mean、原完整C0、编译、50公共剩余后缀与全部私有提交字段/实际raw SHA谱系/父选择时钟/真实源身份；参考95原物理推进与保存event53→148/各提交状态一致，但原逐推进累计ledger缺失，不能补造其raw档案资格。{facts['independent_request_checks']}项原始重放0失败与2个原注册屏障（原terminal未选、原累计物理事件档案缺失）分开；不改整轮FAIL。

## 终局独占恢复

terminal_restore_root_v2在执行前另登记同原策略/物理/80数据/预算，唯一输入actual shared epoch63，原失败私有终局不使用。父shared预置/库/200历史/80数据及旧epoch13环境预热在外；一个新10秒窗口内实际SHA/epoch/base/pickle及全部私有字段恢复{facts['terminal_restore_s']}秒，随后原_run_loop仅剩10物理推进与返仓、0新接拒/评分/模型。新程序只归档该终局的10事件，不再在发布前编码50累计响应；旧源不改。检查点epoch64/slot1在8秒前验证选择，事件ACK与原字节归档/自然0退出回收合计{facts['terminal_wall_s']}秒。新原writer/shared/envelope/retained{facts['terminal_payload_bytes']}字节SHA{facts['terminal_payload_sha256']}逐字一致；它是新实际序列化，不要求等同原未选122491字节或补造旧缺13800字节。

终局clock{facts['clock']}/event{facts['event']}/replan{facts['replan']}，74接受且74客户全部服务/6拒绝，三车closed、返仓{facts['return_arrivals']}，所有80订单已决定、pending为空。原coldchain_evaluator_a1完整80数据/真实全轨迹hard/0失败，效用{facts['utility']}只属已观察日该控制工程描述，不与旧cond_hist947.40554或旧工程1363.838015846908当策略增益。9运行/1164另进程独立项0失败，直接原终局物理、原完整评价、所有10事件/私有字段、实际原pickle字节/父SHA谱系/时间及自然退出核验；独立不调用新worker/engine/projector作为参考，物理/canonical仍共享原实现。

## 边界与下一步

原生产{facts['production_files']}文件/{facts['production_modules']}map、50核查{facts['request_audit_files']}/{facts['request_audit_modules']}，新终局{facts['terminal_files']}/{facts['terminal_maps']}及独立{facts['terminal_audit_files']}/{facts['terminal_audit_maps']}实际加载起止闭合，仅本地Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型负载。无服务器/GPU/共享MD上传，最终20260927/28未用。原生成器首次预执行定位导致IndentationError字节保留，当时未启动生产。

本已观察日当前控制链已到权威终局，无待恢复callback81或旧pending39；不得再次执行50请求、原终局失败或本恢复/审计QA。父同步归档/任意负载慢磁盘/native内部抢占/selector撕裂内存屏障/父崩溃持久化/外部不可逆操作仍未验，full_live_request/full_loaded_dependency_closure/广义physical_ack_loss均False；没有完整正式控制臂、已训同初始化CE、同任务训练随机控制、独立强动态接拒或等耗时MaskCO迭代。下一工程优先接齐这些学习与控制链以及科学完整实际依赖/预算，屏障全满足后另数据前登记有限新鲜科学日。配送2254/15160动作张量及理想163/1886864沿用不重跑，第2族FAIL封闭，第3族未启动；负结果31必要门/183历史/C1及全部旧FAIL保持。16入口先保存旧字节本地更新。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/RESIDENT_TAIL_TERMINAL_RECOVERY_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),facts=facts,original_directory=str(O.relative_to(ROOT)),audit_directory=str(A.relative_to(ROOT)),terminal_directory=str(T.relative_to(ROOT)),scoped_fifty_requests_qualified=True,scoped_separate_terminal_restore_qualified=True,whole_original_run_PASS=False,whole80_initial_single_resident_qualified=False,raw_intermediate_event_ledger_qualified=False,new_scientific_family=False))
    backup=R/'current_entries_before_tail_terminal_recovery_20261012';entries=read(R/'current_entries_before_lineage_header_cancel_20261012/UPDATE_IDENTITY_R2.json')['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries);backup.mkdir(exist_ok=False)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note=f"\n> 2026-10-12取货后50请求与单独终局恢复：原resident_day_tail整轮173.071秒退出1（terminal未选/累计事件档案缺失）保留。50实际callback31..80连续提交、{facts['independent_request_checks']}流式独立项0失败；未合格R2页末失败另目录R3恢复，未重跑生产50。新actual selected63恢复终局9运行/1164独立项0失败、1.968556秒含自然0回收、10终局物理；74接受全服务/6拒绝、全车返仓closed、原完整80评价hard。父最终slot1 epoch64/callback80/order75 raw123700 SHA6730d6883ec6bc4ec60e4fdf30abae0202130e17a248c2673e97c46b3a047d92；pending为空，无callback81。资格为同resident50+另恢复终局，原整轮/原raw中间ledger仍FAIL，不等完整科学门。下一正式学习/已训CE/强动态接拒/等耗时迭代/全科学依赖预算；配送第2族FAIL封闭、第3族未启动，稳定在线增益未建立。\n"
    updates=[]
    for x in entries:
        path=E/x['path'];text=path.read_text(encoding='utf-8')
        if path.suffix=='.json':
            obj=json.loads(text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(resident_day_tail_directory=str(O.relative_to(ROOT)),resident_day_tail_original_exit=1,resident_day_tail_whole_run_PASS=False,resident_tail_stream_audit_directory=str(A.relative_to(ROOT)),resident_tail_stream_audit_checks=facts['independent_request_checks'],resident_tail_stream_raw_replay_failures=0,resident_tail_registered_barrier_failures=facts['registered_barrier_failures'],raw_intermediate_physical_event_ledger_qualified=False,terminal_restore_directory=str(T.relative_to(ROOT)),terminal_restore_runtime_checks=9,terminal_restore_independent_checks=1164,scoped_fifty_submitted_requests_qualified=True,scoped_separate_terminal_restore_qualified=True,whole80_initial_single_resident_qualified=False,current_authoritative_epoch=64,current_authoritative_callback=80,current_authoritative_order=75,current_authoritative_slot=1,current_authoritative_payload_sha256=facts['terminal_payload_sha256'],current_authoritative_payload_bytes=facts['terminal_payload_bytes'],current_authoritative_source=str((T/'RETAINED_PAYLOAD.bin').relative_to(ROOT)),caller_selected_epoch=64,caller_selected_callback=80,caller_selected_order=75,caller_selected_slot=1,discarded_pending=None,discarded_pending_callback=None,discarded_pending_order=None,next_resume_callback=None,current_observed_controller_at_terminal=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Old observed controller now at actual selected terminal64/callback80/order75; all80 decided,74 accepted served/allreturned,pendingempty. Do NOT replay resident50, originalfailedterminal, separate9/1164terminal, R3audit or earlierqualifiedQA. Original wholeFAIL/missingrawintermediateledger/R2footerFAIL preserved; scope same-resident50 plus separateterminalrestore, not initialwhole80/science. Next prospectively integrate formal learned proposals/same-init trainedCE/task-trainedrandom/independentstrongdynamic admission/equal-timeMaskCOiteration, actual fullscientific closure and wholebudget barriers before separately preregistering freshscience. No finalseeds or extra oldday nomodelQA.')
            obj['delivery']['next']='Actual model scoring head/action provenance/allcontrol outputs/same-init trainedCE/task-trainedrandom/fullC0/interruptiblebudget remain for prospective engineering before finitefamily3/newdata. Do not repeat2254/15160 tensors, ideal163/1886864 or closedfamily2. Pickup actualterminal64/callback80 pendingempty; do not resume old13 or assume callback81. Original residentwholeFAIL/missingledger preserved.'
            obj['reproduction']['resident_tail_terminal_recovery_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');obj['latest_stage_receipts_root']=str(T.relative_to(ROOT));new=json.dumps(obj,ensure_ascii=False,indent=2)
        else:
            new=note+text if path.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if path.parent==E/'01_取货线':new+='\n- [五十揭示连续提交与终局超时后独立恢复](文档/结果与进度/'+report.name+')\n'
        path.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(path)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(T/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active))
    print(json.dumps(dict(entries=16,facts=facts,active=active),ensure_ascii=True))
if __name__=='__main__':main()
