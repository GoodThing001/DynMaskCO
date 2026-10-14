"""Freeze new evidence and update the sixteen local entries after byte backups."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os,pickle
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/lineage_header_cancel_root_v1'
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_LINEAGE_HEADER_CANCEL.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(20,0,1818,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_exit']==1 and r['actual_worker_terminated']
    assert all(sha(p)==h for p,h in read(O/'REGISTRATION.json')['sources'].items()) and all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    assert sha(O/'RETAINED_PAYLOAD.bin')==sha(O/'ACTUAL_SHARED_PAYLOAD_13.bin')==sha(O/'ACTUAL_WRITER_PAYLOAD_13.bin')=='5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7'
    names=('build_pickup_lineage_header_cancel_v1.py','qualify_pickup_lineage_header_cancel_v1.py','audit_pickup_lineage_header_cancel_v1.py');active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in names for t in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        f=O/('FROZEN_'+n);assert not f.exists();shutil.copy2(ROOT/'tools'/n,f);frozen[str(f)]=sha(f)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True));create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,independent_actual_exit=0,actual_worker_terminated=True,actual_worker_exit=1,worker_natural_exit=False,owned_processes_active=active))
    report=E/'01_取货线/文档/结果与进度/取货_实际后续事务与发布谱系头错误取消组件_2026-10-12.md'
    create(report,'''# 实际后续事务与发布谱系头错误取消组件（2026-10-12）

pickup/lineage_header_cancel_root_v1事前登记，唯一来源为event_loss_resume_root_v1实际选定slot0 epoch12/callback29的38929字节原shared载荷，SHA3ae9e8f0348b15c544da7cfccfc8c03ce2e2154f3c7cca84fd4bacc70f1027b2。父shared预置、原运行库/200历史/80数据及仅旧epoch10环境预热在外；首10秒窗口实际SHA/epoch/base/pickle和全部私有字段恢复0.025727100088261068秒，恢复段0接拒/评分/物理。没有重做前阶段19/1807、1183/1FAIL及R2窄8、旧重复票据或旧发布故障QA。

真实callback30揭示订单48才首次调用端权威接受，own_incumbent未换路线。计时内恢复、现场私有公共投影编码解码、own_uncond_dynamic当前-only接拒、原200历史共同K10全部20分支、原完整C0认证编译、原guard安装、actual plan_and_commit、检查点生成发布、父selector有界SHA谱系校验及截止前选择/ACK/归档，共3.7245484000304714秒，resident保留未回收。两权威物理推进至clock8.159917831420898/event53/replan47/24服务；父选择slot1 epoch13。writer/actual shared/envelope/父保留42569字节原载荷逐字相符，SHA5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7。

同resident下一callback31/订单39以真实epoch13继续，5个私有物理推进至clock9.18341064453125/event58/replan50/28服务，当前-only临时接受39、共同K10全部20分支/完整C0/编译/原guard私有安装和当前plan_and_commit完成。实际writer51049字节SHAa82af6032de360536e5d8769c13b12a7b0864e10b8f47079def86fa4cae82909完整保存。新故障7仅将未选槽0新HEADER的base首字节异或1；payload与SHA/长度/epoch14/私有state digest保持正确，原payload内部base仍正确，因此该envelope谱系不一致。完整坏header写入并过原deadline/cancel门后切selector至0，Win32事件与ACK前12秒延迟，stage7实际到达。这是故障注入路径，不能称原正常worker产生错误谱系。

父新独占控制在当前base门直接拒绝，保存实际seen时间/正确base/错误base/slot/sequence/size；没有验证选择epoch14、没有新票据/ACK/调用端执行。执行8秒后取消kill/reap，共8.020578900002874秒（预留2秒/总10），RETAINED为父已选择epoch13的完整envelope，逐字等于首次选择的actual shared/envelope。取消时共享selector仍指向坏slot0 epoch14，不能把它称有效调用端检查点，不能从该槽恢复或假称selector被回滚；父保留票据/原缓存与共享选择器要分开。39临时接受、私有安装/plan及5个私有推进销毁，不计调用端权威执行。下一实际未决callback31/order39必须从父真实保留epoch13/slot1/原42569字节恢复，本轮未再恢复或正式执行39。正常48与两次物理推进不能因下一故障回退。

20运行/1818另进程独立项0失败，监督和独立实际退出0，worker真实kill returncode1单独保存，不写自然退出0。独立直接原接拒、全部40共同K10 float.hex/NumPy分组mean、完整C0/编译、7原物理事件、直接公共剩余后缀、全部私有字段、实际pickle字节SHA谱系、单比特header错误/正常父校验选择/故障拒绝时钟/完整保留envelope及实际进程退出核验；不调用新engine/projector/checkpoint worker作为参考，物理/canonical共享原实现。7物理中2权威、5取消私有；原始重放这些私有事实不将其升级为调用端执行。

生产正常计算提交后与故障入口517实际文件/263module map相同，审计523文件/251map起止闭合，仅本地Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型负载。worker被kill无自然故障末端全闭包。首正常未测回收；父有界复制/SHA/同步小载荷归档计时不证明任意大小/慢磁盘/父阶段抢占、任意header/selector撕裂或内存屏障、native任意指令点、父崩溃持久化或外部不可逆资格。仅这一个实际谱系头故障，不等完整对抗输入或整个科学控制链。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss均仍False。下一另事前工程从实际父保留epoch13/callback30/slot1原载荷接续真正未决31/order39，完成剩余预算/连续控制，再接正式学习、同初始化已训CE、独立强动态接拒、全日10秒等耗时迭代和科学完整依赖；全工程门满足后才另登记有限新鲜科学日，不按临时效用挑样。

配送动作张量2254/15160及理想动作链163/1886864资格沿用不重跑，下一实际评分模型与控制/C0/可中止预算；第2族FAIL封闭、三臂训练/采集/审计/DEV/确认关闭，第3族未启动。负结果31/183/C1与原10890/4FAIL/缺13800字节保持不重整。16入口先保存旧字节本地更新，无服务器/GPU/共享MD上传，最终20260927/28未用。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/LINEAGE_HEADER_CANCEL_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_LINEAGE_HEADER_CANCEL.json'),runtime_checks=20,independent_checks=1818,failed=0,authoritative_callback=30,authoritative_order=48,authoritative_epoch=13,authoritative_slot=1,retained_payload_sha256=sha(O/'RETAINED_PAYLOAD.bin'),retained_payload_bytes=42569,discarded_callback=31,discarded_order=39,shared_selector_bad_slot=0,shared_bad_header_epoch=14,header_base_bit_flip_rejected=True,authority_physical_events=2,discarded_private_events=5,actual_worker_terminated=True,actual_worker_returncode=1,scoped_lineage_fault_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_family=False,stable_online_gain=False))
    backup=R/'current_entries_before_lineage_header_cancel_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_action_tensor_binding_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        f=backup/x['path'];f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],f)
    note='\n> 2026-10-12取货发布谱系头错误组件：事前从actual epoch12恢复接续真实30/order48接受并选slot1 epoch13/42569字节；同resident31/order39临时接受和5私有推进后，故障仅将新HEADER base首字节异或1并切共享selector到坏slot0 epoch14，事件ACK前延迟。父实际拒绝、8.020579秒取消回收，保留先前选定13原envelope；坏共享selector不等调用端选择，39和5推进丢弃，48及2推进权威。20运行/1818独立项0失败，实际worker kill返回1保留，监督审计退出0。下一从父真实保留13接续未决31/order39；完整科学模型/已训CE/强动态/全日10秒与依赖仍待。配送2254/15160沿用不重跑，第2族FAIL封闭、第3族未启动，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];text=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(lineage_header_cancel_directory=str(O.relative_to(ROOT)),lineage_header_cancel_runtime_checks=20,lineage_header_cancel_independent_checks=1818,lineage_header_cancel_failed=0,scoped_lineage_header_cancel_qualified=True,current_authoritative_epoch=13,current_authoritative_callback=30,current_authoritative_order=48,current_authoritative_slot=1,current_authoritative_payload_sha256=sha(O/'RETAINED_PAYLOAD.bin'),current_authoritative_payload_bytes=42569,current_authoritative_source=str((O/'RETAINED_PAYLOAD.bin').relative_to(ROOT)),discarded_pending=dict(callback=31,order=39,private_epoch=14,reason='wrong_header_base_rejected_not_caller_selected'),next='Do not repeat20/1818 or prior completed QA. Actual caller retained slot1 epoch13/callback30/order48 payload42569 SHA5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7 is sole next source, equal normal actual shared13. Shared selector remained badslot0epoch14 with one-bit header base mismatch, explicitly NOT caller choice/source. Genuine pending callback31/order39 and5privatephysics were discarded;48 and2physics authoritative. New prospective engineering restore actual parent retained13 and continue31; formal models/same-init trainedCE/independentstrongdynamic/full-day10/equal-time/scientificclosure remain. No scientificday or oldQA rerun.')
            obj['delivery']['next']='Do not repeat2254/15160 binding or ideal163/1886864/closedfamily2 QA. Actual model score head/per-action provenance/control outputs, same-init trainedCE/task-trained random, fullC0 and interruptible full budget remain for new prospectively registered engineering before finitefamily3/newdata. Pickup cross-line pointer: actual parent retainedslot1epoch13/callback30/order48 raw42569; continue genuine pending31/order39, NOT invalid sharedslot0epoch14; five discarded private physics not caller execution.'
            obj['reproduction']['lineage_header_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2)
        else:
            new=note+text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if f.parent==E/'01_取货线':new+='\n- [实际后续事务与发布谱系头错误取消](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active))
    print(json.dumps(dict(entries=16,runtime=20,independent=1818,failed=0,active=active)))
if __name__=='__main__':main()
