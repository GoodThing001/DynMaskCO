"""Append scoped evidence and preserve all 16 prior entry bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,sys
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/prepared_slot_cancel_root_v1'

def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_PREPARED_SLOT_CANCEL.json')
    assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(19,0,2309,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_terminated'] and r['actual_worker_exit']!=0
    for field,n in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):assert all(sha(p)==h for p,h in read(O/n)[field].items())
    names=('qualify_pickup_prepared_slot_cancel_v1.py','audit_pickup_prepared_slot_cancel_v1.py','build_pickup_prepared_slot_cancel_v1.py')
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);assert not dest.exists();shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True))
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(supervisor_actual_exit=0,independent_actual_exit=0,worker_was_terminated=True,worker_actual_return_code=r['actual_worker_exit'],worker_natural_exit_0=False,owned_processes_active=active))
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_9.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));normal,discarded=actual['records'];ep=normal['private_after'];dp=discarded['private_after'];h0,h1=r['host'];s=read(O/'SOURCE_START.json');i=read(O/'AUDIT_SOURCE_START.json')
    source=R/'pickup/inactive_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_7.bin'
    report=E/'01_取货线/文档/结果与进度/取货_真实恢复接续与完整未选检查点取消组件_2026-10-12.md'
    create(report,f'''# 真实恢复接续与完整未选检查点取消组件（2026-10-12）

pickup/prepared_slot_cancel_root_v1执行前登记，从inactive_slot_cancel_root_v1调用端实际选定epoch7/callback24/event44的原shared {source.stat().st_size}字节、SHA{sha(source)}继续。父初始shared预置、运行库/历史/原80订单数据及仅旧epoch6私有环境预热在计时外；首10秒窗口中验证原SHA/epoch/base、pickle解码并恢复全部封存私有字段，恢复段{restore['finished']-restore['started']:.9f}秒，0接拒/评分/物理。旧callback25/order27被取消私有分支接受/安装，不是调用端已执行，本轮不读取该分支。本轮仅新正常callback25与下一callback26故障，不重做19/2309半写或其他旧QA。

正常callback25/order{normal['order']}首次调用端权威接受，原current-only own_uncond_dynamic/own_incumbent同计划、原200历史共同K10全部20分支、原完整C0认证/编译、原JointPolicyOwner私有安装/实际plan_and_commit、原检查点pickle生成、共享发布和父选票据/ACK归档，未换路线。正常{h0['wall_s']:.9f}秒，保留同resident未回收；原run_loop一次权威物理推进，clock{float(ep['clock'])}/event{int(ep['event_id'])}/replan{int(ep['replan_count'])}/{int(ep['served'].sum())}服务。父选择槽1/epoch8；actual writer/shared/envelope {normal['payload_size']}字节、SHA{normal['payload_sha256']}逐字一致。该正常提交才使订单27成为调用端接受，不计旧取消分支的接拒/推进。

同resident/env下一callback26/order{discarded['order']}以该实际提交结果继续，一次私有物理推进到clock{float(dp['clock'])}/event{int(dp['event_id'])}/replan{int(dp['replan_count'])}/{int(dp['served'].sum())}服务。当前-only临时接受{discarded['order']}，原共同K10全部20分支、C0、编译、原guard私有安装及当前plan_and_commit均已完成。完整writer {discarded['payload_size']}字节、SHA{discarded['payload_sha256']}；新故障固定完整payload与新epoch9 HEADER均写到未选槽0后、共享selector切换前12秒延迟，stage5实际到达。worker保存的完整未选envelope、父从实际shared保存的完整envelope与原writer payload逐字相符，header长度/SHA/epoch/base/私有state摘要与全字段均合格。该槽此刻是完整有效新候选，不是调用端选择。共享槽号仍1，无新Win32事件/ACK、无epoch9票据或调用端选择。

实际{h1['wall_s']:.9f}秒取消kill/reap，执行8秒/父与回收预留2秒/总10秒。RETAINED原完整envelope逐字保留已选epoch8，完整epoch9候选被丢弃；order{discarded['order']}临时接拒/私有安装/plan和1次物理推进销毁，不计调用端权威执行。worker真实returncode{r['actual_worker_exit']}因终止，监督和独立实际退出0，不写worker自然退出0。下一真正未决订单{discarded['order']}必须从actual epoch8恢复，不用未选epoch9或旧取消分支。本轮没有再恢复/正式执行该订单，没有新重复序号/丢ACK测试。

19运行/2309另进程独立项0失败。独立直接原接拒、原StrictOnlineEnv、原完整C0、原SAA组装，全部40共同分支float.hex/NumPy分组mean、接拒、公共剩余后缀、编译、两次实际物理事件、全部封存私有字段、实际原pickle字节/SHA/谱系、正常选择与完整未选候选、精确耗时和真实进程退出逐项核验；不用新engine/projector/checkpoint worker作参考。2次物理事件中1次权威、1次取消私有，不能均计调用端推进；物理/canonical摘要共享原实现，不是另一物理实现，也不是2309独立科学样本。

生产开始至正常计算/提交后及第二私有提交后完整未选写入入口{len(s['files'])}文件/{len(s['module_files'])}module映射一致，独立{len(i['files'])}文件/{len(i['module_files'])}映射起止闭合，仅本Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型负载。kill不能写自然故障末端闭包；正常首请求未测回收，取消窗口含实际回收。初始shared预置和日初预热在外，父同步小载荷归档不证明任意父阶段/慢磁盘硬实时。只覆盖完整payload/header就绪后、selector之前这一入口；未验任意选择器撕裂/内存屏障、selector之后事件丢失、父崩溃/持久化/外部不可逆操作或任意native指令点。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False，正式学习、同初始化已训CE、独立强动态接拒、全日10秒等耗时迭代和科学完整依赖仍待。下一另事前工程从actual selected slot1 epoch8/callback25真正未决order{discarded['order']}继续其它发布边界/连续控制与回收；所有工程门满足后才另登记有限新鲜科学日，不按临时效用挑样。配送理想163/1886864不重复，第2族FAIL封闭，三臂训练/采集/审计/DEV/确认禁用，第3族未启动。负结果31/183/C1、旧10890/4FAIL与缺13800字节保持。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。

16入口保存旧字节后本地更新，本人进程退出，无服务器/GPU/共享MD上传，最终20260927/28未用，无新科学数据/模型/收益裁决。
''')
    boundary=R/'reproduction/PREPARED_SLOT_CANCEL_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_PREPARED_SLOT_CANCEL.json'),runtime_checks=19,independent_checks=2309,failed=0,selected_epoch=8,caller_authoritative_callback=25,caller_authoritative_order=normal['order'],discarded_callback=26,discarded_order=discarded['order'],selected_slot=1,complete_inactive_slot=0,candidate_epoch=9,normal_payload_bytes=normal['payload_size'],normal_payload_sha256=normal['payload_sha256'],scoped_prepared_slot_cancel_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_prepared_slot_cancel_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_inactive_slot_cancel_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12取货完整未选检查点取消：事前登记从actual epoch7计时内恢复，callback25/order27首次正式接受并选择slot1/epoch8；同resident下一callback26/order36私有接拒/完整评分/C0/安装和plan后完整payload与epoch9 HEADER写未选slot0，selector前取消。完整候选不等调用端选择，已选epoch8原envelope逐字保留。19运行/2309独立项0失败，正常3.194149秒/取消8.024898秒；36非调用端接受，完整科学/全日10秒/强动态与已训CE仍待，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(prepared_slot_cancel_directory=str(O.relative_to(ROOT)),prepared_slot_cancel_runtime_checks=19,prepared_slot_cancel_independent_checks=2309,prepared_slot_cancel_independent_failed=0,scoped_prepared_slot_cancel_component_qualified=True,caller_selected_epoch=8,caller_selected_callback=25,caller_selected_order=27,discarded_pending_callback=26,discarded_pending_order=36,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat new19/2309 complete inactive candidate or prior QA. Actual selected slot1 epoch8/callback25/order27 accepted; callback26/order36 accepted/installed only in discarded private branch whose complete epoch9 payload/header is in unselected slot0, selector/event/ACK absent. Next prospectively restore actual epoch8 for pending36 and subsequent publication boundaries/reclaim budget. Formal learned/same-init trained CE/independent strong dynamic admission, full-day10s/equal-time iteration and scientific full loaded closure required before fresh science. Preserve prior FAIL and missing13800 bytes.')
            obj['delivery']['next']=obj['delivery']['next'].split('Continue pickup ')[0]+'Continue pickup actual selected slot1 epoch8/callback25 after authoritative order27 commit; pending36 remains uncommitted after complete inactive-envelope cancellation. Formal CE/strong dynamic/full-day10s/dependency gaps continue.'
            obj['reproduction']['prepared_slot_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [真实恢复接续与完整未选检查点取消](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=16,runtime_checks=19,independent_checks=2309,failed=0,restore_s=restore['finished']-restore['started'],records=[dict(epoch=x['sequence'],order=x['order'],clock=float(x['clock']),event=int(x['private_after']['event_id']),replan=int(x['private_after']['replan_count']),served=int(x['private_after']['served'].sum()),size=x['payload_size'],sha=x['payload_sha256']) for x in actual['records']],owned_processes=active)))
if __name__=='__main__':main()
