"""Archive new inactive-buffer partial publication only, preserve all prior bytes."""
import hashlib,json,shutil,os,sys
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/inactive_slot_cancel_root_v1'
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
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_INACTIVE_SLOT_CANCEL.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(19,0,2309,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_terminated'] and r['actual_worker_exit']!=0
    for field,n in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):assert all(sha(p)==h for p,h in read(O/n)[field].items())
    names=('qualify_pickup_inactive_slot_cancel_v1.py','audit_pickup_inactive_slot_cancel_v1.py','build_pickup_inactive_slot_cancel_v1.py')
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);assert not dest.exists();shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True))
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(supervisor_actual_exit=0,independent_actual_exit=0,worker_was_terminated=True,worker_actual_return_code=r['actual_worker_exit'],worker_natural_exit_0=False,owned_processes_active=active))
    sys.path.insert(0,str(ROOT/'tools'));import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_8.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));normal,discarded=actual['records'];ep=normal['private_after'];dp=discarded['private_after'];h0,h1=r['host'];w=read(O/'PARTIAL_WRITE_WITNESS.json');s=read(O/'SOURCE_START.json');i=read(O/'AUDIT_SOURCE_START.json')
    report=E/'01_取货线/文档/结果与进度/取货_真实恢复接续与未选缓冲区半写取消组件_2026-10-12.md'
    create(report,f'''# 真实恢复接续与未选缓冲区半写取消组件（2026-10-12）

pickup/inactive_slot_cancel_root_v1执行前登记，新独占程序及审计均在运行前绑定字节。从live_resume_saa_cancel_root_v1调用端实际选定epoch6/callback23/event42原shared30927字节（SHA7615a4283f7c75852a6958f9f2eaea065433d3ba7071111cb294d47f10a56e91）继续。父初始shared预置、运行库/历史/原80订单日和仅旧epoch5环境预热在计时外，实际选定epoch6载荷读入/SHA/epoch/base/pickle/全封存字段恢复在首10秒窗口，恢复段{restore['finished']-restore['started']:.9f}秒，0接拒/评分/物理。旧order58只被取消私有分支接受，本轮从真实epoch6重新进入仍未正式提交的callback24，不读取该旧分支安装，不重跑15/1757等旧QA。

新双缓冲组件每槽2MiB+HEADER，初始选槽0封存epoch6；当前worker只写另一个未选槽，完整payload和header写完且deadline/cancel门通过后，将共享控制偏移48的对齐32位槽号切到新槽，再发Win32事件。父收到事件读所选槽，核谱系和界限后选择票据。它是本地内存协议组件，未改旧单缓冲原件，不对旧已完成结果作推翻或重新验收；不声称任意指令处内存屏障/选择器撕裂或父崩溃一致性已经证明。

正常callback24/order58实际current-only own_uncond_dynamic接受/own_incumbent同计划提议、原200历史共同K10全部20分支、原完整C0/编译/原JointPolicyOwner私有安装/实际plan_and_commit和检查点生成发布。两次原run_loop物理推进，clock{ep['clock']}/event{ep['event_id']}/replan{ep['replan_count']}/21服务，未换路线。正常{h0['wall_s']:.9f}秒含选定恢复、现场公共投影编码解码、全部控制计算、私有安装、原检查点共享发布/父选择ACK和归档；保留resident未回收。父实际选槽1/epoch7，writer/shared/envelope33907字节，SHA{normal['payload_sha256']}逐字一致。此次订单58才计调用端权威接受，旧取消分支58不计。

同resident下一callback25/order27从真实epoch7提交结果继续。一新私有物理推进至clock{dp['clock']}/event{dp['event_id']}/replan{dp['replan_count']}/21服务，当前-only临时接受27、原共同K10全20分支/C0/编译/原guard私有安装和plan_and_commit已完成；实际生成完整33911字节writer检查点，SHA{discarded['payload_sha256']}。新故障固定在未选槽0写入前{w['bytes_written']}字节（floor(33911/2)）后12秒延迟，stage4实际到达。未写新的HEADER、未切所选槽号、未发检查点事件或ACK；此时未选槽0仍带旧epoch6 HEADER，payload前半已经部分替换，不称该未选槽仍是完整旧状态或可恢复载荷。实际父从shared复制出的未选槽前缀与worker保存前缀及writer原字节前半逐字一致，实际未选HEADER与原初始HEADER逐字一致。

{h1['wall_s']:.9f}秒取消kill/reap（执行8/父与回收预留2/总10），无epoch8票据/ACK/调用端选择，所选槽仍1，RETAINED完整envelope与原所选epoch7逐字一致。worker真实被终止returncode{r['actual_worker_exit']}保留，监督和独立实际退出0，不伪称worker自然退出0。27临时接受、私有安装/当前plan_and_commit和1次推进被销毁，不计调用端权威执行；下一未决订单27应从actual epoch7恢复。本轮不另开恢复窗、未正式执行27，不证明完整后续恢复、任意发布位置取消、迟到/重复序号或真实外部不可逆操作。

19运行/2309另进程独立项0失败。独立以原接拒/原StrictOnlineEnv/完整C0/原SAA组装为参考，不调用新engine/projector或checkpoint worker：全部两次新请求40个K10分支及NumPy分组mean、接拒/逐物理事件/私有字段/公共剩余后缀/原编译/实际原pickle字节及SHA/谱系/半写前缀/保留槽/时长与源身份逐一复核。共3次物理推进中2次属于调用端已选正常提交，另1次属于丢弃私有分支，不能把3都计调用端权威推进。物理和canonical摘要共享原实现，不是第二物理实现；检查数量不等独立科学样本数。

正常开始/计算后/正常末端、第二私有计算/提交后至实际半写入口{s['files'].__len__()}文件/{len(s['module_files'])}映射一致；独立{i['files'].__len__()}文件/{len(i['module_files'])}映射起止闭合并核实际加载原字节。worker被kill不能写自然故障结束闭包。实际Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程，仅本无模型负载，不等正式科学完整依赖。首正常窗口未测回收，第二取消实际含终止回收；父同步小载荷归档、任意大小/native内部取消/慢磁盘/任意选择器位置/父崩溃持久化与撕裂发布未验。旧单缓冲实例没有被追加本故障，不把该新方案的本次成功升为旧完整硬实时资格。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False。下一新事前工程从实际所选epoch7真正未决27及后续事件继续，接齐其它发布边界、连续控制/回收预算与正式学习/同初始化已训CE、独立强动态接拒、全日10秒等耗时迭代/科学完整依赖；全部工程屏障满足后才另登记有限新鲜科学日，不按临时效用挑样。配送163/1886864理想输入组件沿用，第2族科学FAIL关闭训练/采集/审计/DEV/确认、第3族未启动。负结果31/183/C1、旧10890/4FAIL与缺失13800载荷不重整。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。

16当前入口保存旧字节后本地更新，本人进程已回收，无服务器/GPU/共享MD上传，最终取货20260927/28未用；没有新科学族/数据/模型或收益裁决。
''')
    boundary=R/'reproduction/INACTIVE_SLOT_CANCEL_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_INACTIVE_SLOT_CANCEL.json'),runtime_checks=19,independent_checks=2309,failed=0,selected_epoch=7,caller_authoritative_callback=24,caller_authoritative_order=58,discarded_callback=25,discarded_order=27,selected_slot=1,partial_inactive_slot=0,partial_bytes=16955,normal_payload_bytes=33907,normal_payload_sha256=normal['payload_sha256'],scoped_inactive_partial_write_cancel_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_inactive_slot_cancel_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_live_resume_saa_cancel_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12取货双缓冲半写取消：事前登记从actual epoch6计时内恢复，callback24/order58正式接受并选槽1/epoch7；同resident下一callback25/order27私有完整计算安装后向未选槽0写半载荷16955字节取消，槽号/新HEADER/事件未发布，已选epoch7原envelope逐字保留。19运行/2309独立项0失败，正常3.475295秒/取消8.021717秒；27非调用端接受，完整科学/全日10秒/强动态与已训CE仍待，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(inactive_slot_cancel_directory=str(O.relative_to(ROOT)),inactive_slot_cancel_runtime_checks=19,inactive_slot_cancel_independent_checks=2309,inactive_slot_cancel_independent_failed=0,scoped_inactive_slot_cancel_component_qualified=True,caller_selected_epoch=7,caller_selected_callback=24,caller_selected_order=58,discarded_pending_callback=25,discarded_pending_order=27,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat new19/2309 or old QA. Actual selected slot1 epoch7/callback24/order58 accepted; callback25/order27 accepted/installed only in discarded private branch after first-half inactive slot0 payload write cancellation. Next prospectively restore actual epoch7 for pending27 and subsequent events/remaining publication boundaries and reclaim budget. Formal learned/same-init trained CE/independent strong dynamic admission, full-day10s/equal-time iteration and full scientific loaded closure remain required before fresh science. Preserve old failures/missing13800 bytes.')
            obj['delivery']['next']=obj['delivery']['next'].split('Continue pickup ')[0]+'Continue pickup actual selected slot1 epoch7/callback24 after authoritative order58 commit; pending27 remains uncommitted after inactive-slot half-write cancellation. Formal CE/strong dynamic/full-day10s/dependency gaps continue.'
            obj['reproduction']['inactive_slot_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [真实恢复接续与未选缓冲区半写取消](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),runtime_checks=19,independent_checks=2309,failed=0,owned_processes=active)))
if __name__=='__main__':main()
