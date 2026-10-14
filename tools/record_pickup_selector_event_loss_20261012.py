"""Archive this new selector-only publication; preserve all historical bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/selector_event_loss_root_v1'

def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_SELECTOR_EVENT_LOSS.json')
    assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(22,0,2314,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_terminated'] and r['actual_worker_exit']!=0
    for field,n in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):assert all(sha(p)==h for p,h in read(O/n)[field].items())
    names=('qualify_pickup_selector_event_loss_v1.py','audit_pickup_selector_event_loss_v1.py','build_pickup_selector_event_loss_v1.py')
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);assert not dest.exists();shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True))
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(supervisor_actual_exit=0,independent_actual_exit=0,worker_was_terminated=True,worker_actual_return_code=r['actual_worker_exit'],worker_natural_exit_0=False,owned_processes_active=active))
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_10.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));normal,published=actual['records'];ep=normal['private_after'];dp=published['private_after'];h0,h1=r['host'];s=read(O/'SOURCE_START.json');i=read(O/'AUDIT_SOURCE_START.json');w=read(O/'SELECTOR_PUBLICATION_WITNESS.json')
    report=E/'01_取货线/文档/结果与进度/取货_选择器已发布而通知回执缺失的现场事务组件_2026-10-12.md'
    create(report,f'''# 选择器已发布而通知回执缺失的现场事务组件（2026-10-12）

pickup/selector_event_loss_root_v1执行前登记，固定actual epoch8/callback25原shared35310字节、SHA25ac18f030b209f0bc8de031d746b311174a48ade174661d6c9cbecf9f47dfd4起点。父shared预置、运行库/原200历史/原80订单日及仅旧epoch7环境预热在外；首10秒窗口验证实际原字节SHA/epoch/base、pickle解码、全部封存字段恢复，{restore['finished']-restore['started']:.9f}秒，0接拒/评分/物理。旧order36在未选epoch9中的私有接受/安装不属调用端执行，不用该分支；新正常callback26/order36才从actual epoch8继续首次正式处理。

本新独占父程序将共享selector作为发现提交的依据，在循环中读取对齐32位槽号，无需先收到Win32事件。只对当前sequence选择：核slot/epoch/长度上限/base、复制完整有界payload并核实际SHA、再次确认selector未变，要求验证和选择均在执行8秒截止前。未选择旧sequence、未从未选槽扫描“更优”结果，也没有pickle父解码。本登记的发布协议与旧依赖事件方案分开，旧生产/登记/FAIL与已合格组件原字节保持；没有升级旧方案资格。没有证明任意内存屏障/撕裂selector/对抗输入或父崩溃持久化。

正常callback26/order36实际接受，own_incumbent未换路线，原现场私有公共投影编码解码/current-only own_uncond_dynamic、原200历史共同K10全部20分支、原完整C0认证编译、原guard安装和actual plan_and_commit、原checkpoint生成发布及父SHA选择/ACK归档，{h0['wall_s']:.9f}秒，保留同resident未回收。一原run_loop权威物理推进至clock{float(ep['clock'])}/event{int(ep['event_id'])}/replan{int(ep['replan_count'])}/{int(ep['served'].sum())}服务；父选择槽1/epoch9，actual writer/shared/envelope{normal['payload_size']}字节、SHA{normal['payload_sha256']}逐字相符。这不是旧取消QA重跑，是原取消分支未计权威执行后实际选定起点的新正式接续。

同resident/env下一callback27/order{published['order']}以实际epoch9提交结果继续。一新权威物理推进至clock{float(dp['clock'])}/event{int(dp['event_id'])}/replan{int(dp['replan_count'])}/{int(dp['served'].sum())}服务，当前-only接受{published['order']}、共同K10全20分支/完整C0/编译/原guard安装及current plan_and_commit均完成，未换路线。完整payload/header先写未选槽0，经原deadline/cancel门后切共享selector至0/epoch10；新故障随后保存实际完整envelope与见证，在Win32事件和ACK之前12秒延迟，stage6实际到达。父在截止前实际发现selector变更并核SHA/谱系、选择唯一epoch10票据，事件与ACK现场均未收到。检查点{published['payload_size']}字节、SHA{published['payload_sha256']}，writer/worker所存已发布envelope/actual shared/父选择时复制与末端RETAINED envelope逐字一致。

{h1['wall_s']:.9f}秒含完整请求、已选字节归档及执行8秒后kill/reap，父与回收预留2秒，总10秒。原worker真实returncode{r['actual_worker_exit']}因终止单独保存，不写自然退出0；监督与独立实际退出0。epoch10已在截止前被调用端选择，order{published['order']}及该1次推进计权威执行，不能因通知缺失或随后终止撤销回到epoch9；也不能把它写成丢弃私有分支。下一需要从actual selected slot0 epoch10/callback27恢复接续新callback28，当前没有再恢复epoch10/重复票据或执行第3揭示，不假定下一订单身份。对外不可逆效果、父崩溃/撕裂发布等未测试。

22运行/2314另进程独立项0失败。独立用原接拒/全部40共同K10分支float.hex和NumPymean/原完整C0/编译/直接公共剩余后缀/原2物理事件与全封存私有字段/实际原pickle字节SHA谱系/父验证与选择时钟/通知与ACK缺失/本人进程退出，不调用新engine/projector/checkpoint worker作参考。两次物理均对应已选正式提交；物理和canonical摘要共享原实现，不是第二物理实现，2314检查也不等独立科学样本。

正常开始/两次计算提交结束/实际selector发布延迟入口{len(s['files'])}文件/{len(s['module_files'])}module映射一致，审计{len(i['files'])}文件/{len(i['module_files'])}映射起止一致，仅Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型负载。被kill无生产自然故障末端闭包。首正常未测回收，第二故障实际含回收；父有界小载荷SHA/复制/同步归档计时不证明任意大小/慢磁盘/任意父阶段抢占，native内部任意取消或科学全依赖仍待。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False。正式模型/同初始化已训CE、独立强动态接拒、全日10秒等耗时迭代和科学完整依赖未闭合。下一另事前工程从实际已选epoch10接续真正后续揭示，核恢复后继续/重复一致性与剩余发布预算；所有工程屏障满足后才另登记有限新鲜科学日，不按临时效用挑样。配送理想163/1886864资格沿用不重复，第2族科学FAIL封闭，三臂训练/采集/审计/DEV/确认禁用，第3族未启动。负结果31/183/C1、原10890/4FAIL与缺失13800字节原件保持。

16入口先保存旧字节再本地更新，本人进程退出，无服务器/GPU/共享MD上传，最终取货20260927/28未用，没有新科学数据/训练/收益裁决。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/SELECTOR_EVENT_LOSS_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_SELECTOR_EVENT_LOSS.json'),runtime_checks=22,independent_checks=2314,failed=0,selected_epoch=10,caller_authoritative_callback=27,caller_authoritative_order=published['order'],first_authoritative_callback=26,first_authoritative_order=36,selected_slot=0,checkpoint_event_received=False,ACK_received=False,discarded_branch_in_this_stage=False,payload_bytes=published['payload_size'],payload_sha256=published['payload_sha256'],scoped_selector_event_loss_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_selector_event_loss_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_prepared_slot_cancel_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12取货selector发布后无通知/回执：事前登记actual epoch8计时内恢复，callback26/order36正式接受选slot1/epoch9；同resident callback27/order5完整接拒/K10/C0/安装plan后切selector到slot0/epoch10、通知与ACK前实际延迟12秒。父独立于事件轮询selector，在截止前核实际SHA/谱系并选择epoch10，故障回收后完整原字节一致。22运行/2314独立项0失败，正常4.253428秒/故障8.026168秒；order5已属调用端权威接受，下一从actual epoch10恢复，不回退epoch9。科学/正式CE/强动态/全日10秒和完整依赖仍待，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(selector_event_loss_directory=str(O.relative_to(ROOT)),selector_event_loss_runtime_checks=22,selector_event_loss_independent_checks=2314,selector_event_loss_independent_failed=0,scoped_selector_event_loss_component_qualified=True,caller_selected_epoch=10,caller_selected_callback=27,caller_selected_order=5,caller_selected_slot=0,discarded_pending_callback=None,discarded_pending_order=None,next_resume_callback=28,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat new22/2314 selector/event-loss or prior QA. Actual epoch8 was restored inside request; callback26/order36 is authoritative epoch9, callback27/order5 selector-published before event/ACK and parent SHA/lineage validated and chose actual slot0 epoch10 before cutoff. Both accepted authoritative; no discarded branch in this stage. Next prospectively restore actual epoch10/callback27 raw35318 bytes for subsequent callback28 and remaining recovery/publication/reclaim boundaries. Formal learned/same-init trained CE/independent strong dynamic admission, full-day10s/equal-time iteration/scientific full closure required before fresh science. Preserve prior FAIL/missing13800 bytes.')
            obj['delivery']['next']=obj['delivery']['next'].split('Continue pickup ')[0]+'Continue pickup actual selected slot0 epoch10/callback27 after authoritative order36 and order5 commits; notification/ACK suppressed but parent choice occurred before cutoff. Next restore actual epoch10 to subsequent callback28; formal CE/strong dynamic/full-day10s/dependency gaps continue.'
            obj['reproduction']['selector_event_loss_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [选择器已发布而通知回执缺失的现场事务](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=16,runtime_checks=22,independent_checks=2314,failed=0,owned_processes=active,actual_selected_epoch=10,actual_selected_order=5)))
if __name__=='__main__':main()
