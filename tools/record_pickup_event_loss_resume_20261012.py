"""Archive only new selected10 resume, no prior experiments or old FAIL rewrite."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/event_loss_resume_root_v1'
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_EVENT_LOSS_RESUME.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(19,0,1807,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_exit']==0 and not r['actual_worker_terminated']
    for field,n in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):assert all(sha(p)==h for p,h in read(O/n)[field].items())
    prior=R/'pickup/selector_event_loss_root_v1';assert sha(prior/'ACTUAL_SHARED_PAYLOAD_10.bin')==read(prior/'HOST_10.json')['ticket']['payload_sha256']=='a4b0c063e7b6959654429bc969f7072fa1de466898ec60be4c194c8283449dbe'
    names=('qualify_pickup_event_loss_resume_v1.py','audit_pickup_event_loss_resume_v1.py','build_pickup_event_loss_resume_v1.py');active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(t).name in names for t in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        f=O/('FROZEN_'+n);assert not f.exists();shutil.copy2(ROOT/'tools'/n,f);frozen[str(f)]=sha(f)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True));create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(supervisor_actual_exit=0,independent_actual_exit=0,worker_was_terminated=False,worker_actual_return_code=0,worker_natural_exit_0=True,owned_processes_active=active))
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_12.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));dup=decode(read(O/'DUPLICATE_SELECTED10.json'));first,last=actual['records'];ep,lp=first['private_after'],last['private_after'];rh=r['restore_host'];h0,h1=r['host'];s=read(O/'SOURCE_START.json');ai=read(O/'AUDIT_SOURCE_START.json')
    report=E/'01_取货线/文档/结果与进度/取货_丢通知后选定检查点恢复重复与连续新揭示组件_2026-10-12.md'
    create(report,f'''# 丢通知后选定检查点恢复、重复与连续新揭示组件（2026-10-12）

pickup/event_loss_resume_root_v1事前登记，固定来自selector_event_loss_root_v1已被父截止前实际SHA/谱系选择的slot0 epoch10/callback27原shared35318字节，SHAa4b0c063e7b6959654429bc969f7072fa1de466898ec60be4c194c8283449dbe。上阶段order5及物理推进已属权威执行；不因通知/ACK缺失回退epoch9，不读任何未选私有分支。

本轮父初始shared预置、运行库/原200历史/原80订单数据及仅旧epoch8环境预热在外。第一个独立10秒窗口内实际读选定原字节、核SHA/epoch/base、pickle解码并恢复全部私有封存字段，恢复段{restore['finished']-restore['started']:.9f}秒，整个恢复加重复票据请求{rh['wall_s']:.9f}秒。epoch10重复只返回已提交摘要，0新接拒/共同评分/物理推进、0新检查点发布；实际共享envelope和selector不变。此重复发生在恢复到已选检查点之后，限定本epoch10/谱系，不是任意网络重排、父崩溃或通用exactly-once资格。首窗口保留resident没有测回收。

同一实际恢复后的resident/env接续原run_loop下一新callback28/order{first['order']}，由原current-only own_uncond_dynamic实际拒绝；随后callback29/order{last['order']}实际接受。两订单身份只在实际揭示读出，事前固定连续两次，不据效用选样。两次均计时内现场私有公共提取编码解码、当前-only接拒、own_incumbent同计划提议、原200历史共同K10各20完整分支、原完整C0认证编译、原guard私有安装、actual plan_and_commit及原检查点生成发布；父轮询selector核当前sequence/长度/base/有界原payload SHA/selector复读和截止前唯一选择、实际通知/ACK及原字节归档。未换路线，没有模型。

第一个新请求{h0['wall_s']:.9f}秒，父选slot1/epoch11，保留resident未回收；端点clock{float(ep['clock'])}/event{int(ep['event_id'])}/replan{int(ep['replan_count'])}/{int(ep['served'].sum())}服务，原writer/shared/envelope{first['payload_size']}字节、SHA{first['payload_sha256']}逐字一致。第二从第一真实提交结果继续，{h1['wall_s']:.9f}秒含最终worker自然退出0及回收，父选slot0/epoch12；端点clock{float(lp['clock'])}/event{int(lp['event_id'])}/replan{int(lp['replan_count'])}/{int(lp['served'].sum())}服务，实际writer/shared/envelope及RETAINED{last['payload_size']}字节、SHA{last['payload_sha256']}逐字一致。两次各2个原物理推进，共4个均属调用端权威执行，没有故障/取消/丢弃分支。三个窗口分别执行8秒/预留2秒/总10秒，恢复/重复窗和后续两个窗口分开，不能合成一个10秒请求。

19运行/1807另进程独立项0失败，监督、独立与worker实际退出0。独立直接原接拒/全部40共同K10分支float.hex及NumPymean/原完整C0/编译/直接公共剩余后缀/原4物理事件/全部私有字段/原pickle字节SHA与谱系/恢复重复字段/选定和请求时钟/本人进程退出，不以新engine/projector/checkpoint worker作参考；物理及canonical摘要仍共享原实现。没有重新执行callback27、上阶段22/2314或任何旧取消/序号/已合格QA。

生产恢复/重复/两计算与自然提交结束{len(s['files'])}实际文件/{len(s['module_files'])}module映射一致；独立{len(ai['files'])}文件/{len(ai['module_files'])}映射起止闭合，仅本地Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型负载。最后请求测自然回收，首恢复/首现场未测每请求回收；日初预热与shared预置在外。没有新延迟故障，不能新增任意selector内存屏障/撕裂、native内部指令点/任意大小/慢磁盘/父同步归档抢占、父崩溃持久化或真实外部不可逆效果资格。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False。正式模型/同初始化已训CE、独立强动态接拒、全日10秒等耗时迭代和科学完整依赖仍待；本组件只关闭实际丢通知后选定epoch10恢复、窄重复无再执行及两个新真实揭示接续。下一另事前工程从actual selected slot0 epoch12/callback29继续真正callback30，不假定下一订单，不按已观察临时效用挑样；同时接齐正式控制与全日预算，全部屏障满足后另登记有限新鲜科学日。

配送理想163/1886864沿用不重复，第2族科学FAIL封闭，三臂训练/采集/审计/DEV/确认禁用，第3族未启动；评分张量/锚点身份/实际各模型控制/完整C0与可中止预算待。负结果31/183/C1、原10890/4FAIL及缺失13800载荷字节保持。16入口先保存原字节后本地更新，本人进程退出，无服务器/GPU/共享MD上传，最终取货20260927/28未用，无新科学数据/训练/收益裁决。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/EVENT_LOSS_RESUME_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_EVENT_LOSS_RESUME.json'),runtime_checks=19,independent_checks=1807,failed=0,source_selected_epoch=10,source_parent_choice_sha256=sha(prior/'HOST_10.json'),restored_duplicate_new_admission_calls=0,restored_duplicate_new_physical_calls=0,source_raw_sha256=sha(prior/'ACTUAL_SHARED_PAYLOAD_10.bin'),selected_epoch=12,caller_authoritative_callback=29,caller_authoritative_order=76,selected_slot=0,first_new_order=71,first_new_order_accepted=False,second_new_order_accepted=True,actual_physical_events=4,discarded_branch_in_this_stage=False,payload_bytes=last['payload_size'],payload_sha256=last['payload_sha256'],scoped_event_loss_resume_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_event_loss_resume_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_selector_event_loss_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12取货丢通知后选定恢复接续：事前登记actual epoch10/callback27原shared35318字节在独立计时窗恢复，重复epoch10返回已提交摘要且0新接拒/评分/物理/发布，共享字节不变；同resident实际callback28/order71拒绝选epoch11、callback29/order76接受选slot0/epoch12，4个原物理推进权威。19运行/1807独立项0失败，恢复重复0.620684秒、两新请求3.552830/3.981350秒，末次含自然退出回收；三窗分别计时。下一actual epoch12/callback29→callback30，未假定订单。正式CE/强动态/全日10秒/完整依赖仍待，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior_text=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior_text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(event_loss_resume_directory=str(O.relative_to(ROOT)),event_loss_resume_runtime_checks=19,event_loss_resume_independent_checks=1807,event_loss_resume_independent_failed=0,scoped_event_loss_resume_component_qualified=True,caller_selected_epoch=12,caller_selected_callback=29,caller_selected_order=76,caller_selected_slot=0,discarded_pending_callback=None,discarded_pending_order=None,next_resume_callback=30,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat19/1807 selected10 event-loss resume or any completed QA. Actual selected10/callback27 raw35318 restored in separate10s window, duplicate10 returned committed digest with0 admission/scoring/physics/publication. Same resident next28/order71 rejected authoritative11, next29/order76 accepted authoritative12;4 authoritative physics, no discarded branch. Selected slot0 epoch12/callback29 raw38929 for genuine next callback30, no assumed order. Last normal request includes natural0 exit/reap. Continue remaining engineering and formal learned/same-init trained CE/independent strong dynamic admission/full-day10s/equal-time/scientific loaded closure before fresh science; preserve FAIL/missing13800.')
            obj['delivery']['next']=obj['delivery']['next'].split('Continue pickup ')[0]+'Continue pickup actual selected slot0 epoch12/callback29 after selected10 restoration/duplicate and new authoritative28/order71 rejection,29/order76 acceptance; genuine next callback30, no assumed identity. Formal CE/strong dynamic/full-day10s/scientific closure still pending.'
            obj['reproduction']['event_loss_resume_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior_text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior_text+note
            if f.parent==E/'01_取货线':new+='\n- [丢通知后选定恢复与连续新揭示](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=16,runtime_checks=19,independent_checks=1807,failed=0,owned_processes=active,actual_selected_epoch=12)))
if __name__=='__main__':main()
