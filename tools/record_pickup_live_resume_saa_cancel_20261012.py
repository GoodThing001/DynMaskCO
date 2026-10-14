"""Archive only the new scope; preserve earlier evidence and sixteen entry bytes."""
import hashlib,json,shutil,os,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/live_resume_saa_cancel_root_v1'
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
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_LIVE_RESUME_SAA_CANCEL.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(15,0,1757,0)
    assert read(O/'RUN_EXIT.json')['exit_code']==0 and r['actual_worker_terminated'] and r['actual_worker_exit']!=0
    for field,n in (('sources','REGISTRATION.json'),('inputs','AUDIT_REGISTRATION.json')):assert all(sha(p)==h for p,h in read(O/n)[field].items())
    names=('qualify_pickup_live_resume_saa_cancel_v1.py','audit_pickup_live_resume_saa_cancel_v1.py','build_pickup_live_resume_saa_cancel_v1.py')
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);assert not dest.exists();shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True))
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(supervisor_actual_exit=0,independent_actual_exit=0,worker_was_terminated=True,worker_actual_return_code=r['actual_worker_exit'],worker_natural_exit_0=False,owned_processes_active=active))
    sys.path.insert(0,str(ROOT/'tools'));import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_6.json'));w=decode(read(O/'SAA_CANCEL_WITNESS.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));normal=actual['records'][0];ep=actual['final'];h0,h1=r['host'];s=read(O/'SOURCE_START.json');i=read(O/'AUDIT_SOURCE_START.json')
    report=E/'01_取货线/文档/结果与进度/取货_选定恢复后真实提交与共同评分途中取消组件_2026-10-12.md'
    create(report,f'''# 选定恢复后真实提交与共同评分途中取消组件（2026-10-12）

pickup/live_resume_saa_cancel_root_v1在执行前登记，固定来自timed_restore_live_root_v1实际选定epoch5/callback22/event40的原shared载荷，SHA62f39e33eeade276e236060d0a34c9dd2b5016334e6f5f3c2bb6c6ed124548bf，28077字节。父初始shared预置、解释器/历史/原80订单数据和仅旧epoch3环境预热在计时外；选定epoch5原字节读入、SHA/epoch/base验证、可信pickle解码和全部封存私有字段恢复在第一个10秒窗口内，恢复段{restore['finished']-restore['started']:.9f}秒，0接拒/评分/物理推进。没有读取此前被取消order7副本的安装结果，没有重跑原13/1684或其它完成QA。

第一个真实后续callback23/order7在恢复后的私有所有者内正常执行：实际公开投影/编码解码、原own_uncond_dynamic当前-only接拒、own_incumbent同计划提议、原200历史共同K10/20完整分支、原完整C0物理关闭认证、编译/原JointPolicyOwner安装和原plan_and_commit；父固定共享控制/32字节选择实际epoch6票据并收ACK，正常{h0['wall_s']:.9f}秒含恢复、计算、安装、实际共享检查点发布及原字节归档，保留resident未回收。订单7接受、未换路线；原run_loop两个实际物理推进，clock{ep['clock']}/event{ep['event_id']}/replan{ep['replan_count']}/20服务。此次才是调用端权威order7接受，旧取消分支order7临时接受继续不计调用端执行。正常实际writer/shared/envelope载荷30927字节，SHA{sha(O/'ACTUAL_SHARED_PAYLOAD_6.bin')}，逐字一致。

同一个resident、同一个私有env/车队/轨迹/served/决策，以上述实际提交结果开始下一callback24/order58。原run_loop再推进两个私有物理事件至clock{w['private_at_reveal']['clock']}/event{w['private_at_reveal']['event_id']}/replan{w['private_at_reveal']['replan_count']}/21服务，当前-only接拒临时接受58，原base完整C0认证通过。工程故障只在原EnergyCompleteSAA._sim_scenario第五个分支调用入口：第一候选own_incumbent已有4个完整分支返回，保存真实分支结果和当时scoring_packet/原采样场景/原基证书/私有状态，然后12秒延迟。未调用第五分支，无第二候选评分/十项均值/择优/最终编译与owner.commit/当前揭示后的plan_and_commit/新检查点生成发布。owner.begin已建当前临时票据，不等已安装。实际入口stage3已到达，{h1['wall_s']:.9f}秒取消kill/reap且低于总10秒；执行8秒、回收/父处理预留2秒。原worker实际被终止、返回码{r['actual_worker_exit']}单独保留，监督和独立实际退出0，不能把被kill worker写成自然退出0。

取消没有epoch7票据/ACK/调用端选择，RETAINED完整envelope和payload与实际epoch6归档逐字一致，调用端保留callback23封存票据。取消副本中的订单58临时接受和两次推进被销毁，不计调用端权威执行；没有仍存活的resident处理下一order58。下一次恢复若继续order58必须以实际保留epoch6起点，不能使用取消副本状态。本组件没有恢复并正式执行order58，没有测试新重复序号、迟到发布或该保留载荷的另一恢复窗口，不冒充已覆盖这些缺口。

15运行检查/1757另进程独立检查0失败。独立以原StrictOnlineEnv/C0ReserveReplanner/UncondDynamic和原完整C0/SAA组装为参考，不调用新engine/projector/checkpoint worker：逐一复算原后续揭示/4个物理事件/所有私有字段/公共剩余后缀/接拒、正常全部20个K10分支与NumPy分组mean、故障第一候选前4个分支的float.hex、原场景及实际共享pickle字节、谱系/摘要/时长/加载身份和本人进程退出。故障4项分支是部分共同评分，不能写全部20项完成或策略已择优；不计算或利用不完整均值选样。物理评价和canonical摘要仍共享原实现，不是第二物理实现。

生产正常开始/计算后/完整正常末端与故障实际入口{s['files'].__len__()}文件/{len(s['module_files'])}映射相同；独立{i['files'].__len__()}文件/{len(i['module_files'])}映射起止闭合，并再核实际所有源字节。被kill worker没有自然故障后闭包。实际Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程，仅本无学习模型负载，不等正式完整科学依赖。只证明第五SAA分支调用入口可取消，并非pickle/native求解器内部任意指令点抢占或任意大小负载、慢磁盘/父同步归档阶段可抢占、父崩溃持久化/撕裂发布/外部真实不可逆效果。正常第一请求未测回收，取消请求实际测终止回收；不升级全日10秒资格。

full_live_request/full_loaded_dependency_closure/广义physical_ack_loss均False。正式学习/同初始化已训CE控制、独立强动态接拒、全日等耗时迭代与科学完整依赖待闭合。下一步另事前登记从实际epoch6恢复真正未决order58及后续事件，补齐其它计算/发布故障与回收预算；正式工程屏障全部满足后才另登记有限新鲜科学日，不以临时效用挑样。配送理想动作链163/1886864资格沿用，第2族FAIL及训练DEV确认关闭、第3族未启动。负结果31必要门/183/C1不重整，旧checkpoint10890/4FAIL和原13800字节载荷缺失保持。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。

16当前入口在保存先前字节后本地更新；无本人在跑本组件进程，无服务器/GPU/共享MD上传，最终取货20260927/28未用。本轮没有新科学族、训练权重或效用裁决。
''')
    boundary=R/'reproduction/LIVE_RESUME_SAA_CANCEL_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_LIVE_RESUME_SAA_CANCEL.json'),runtime_checks=15,independent_checks=1757,failed=0,caller_authoritative_callback=23,caller_authoritative_order=7,selected_epoch=6,payload_sha256=sha(O/'ACTUAL_SHARED_PAYLOAD_6.bin'),discarded_callback=24,discarded_order=58,completed_fault_candidate_branches=4,normal_complete_branches=20,scoped_live_resume_saa_cancel_qualified=True,first_normal_reap_included=False,cancel_reap_included=True,worker_actual_return_code=r['actual_worker_exit'],full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_live_resume_saa_cancel_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_directed_action_chain_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    note='\n> 2026-10-12取货新后续/评分取消组件：实际epoch5计时内恢复，正常callback23/order7正式接受并选择epoch6；同resident下一callback24/order58在共同K10第一候选已完成4分支、第五入口取消kill/reap，保留epoch6原envelope逐字一致。15运行/1757独立项0失败，正常3.747529秒、取消8.023378秒；order58仅丢弃私有分支接受，完整请求/科学依赖/强动态与已训CE/全日10秒未合格，在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(live_resume_saa_cancel_directory=str(O.relative_to(ROOT)),live_resume_saa_cancel_runtime_checks=15,live_resume_saa_cancel_independent_checks=1757,live_resume_saa_cancel_independent_failed=0,scoped_live_resume_saa_cancel_component_qualified=True,caller_selected_epoch=6,caller_selected_callback=23,caller_selected_order=7,discarded_pending_callback=24,discarded_pending_order=58,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat new15/1757 or old QA. Caller selected actual epoch6/callback23/order7; next callback24/order58 only accepted in discarded partial-scoring branch (first4 common K10 branches then fifth-entry cancellation), not caller execution. New prospective engineering should restore actual retained epoch6 for pending order58 and more events/remaining compute-publication faults with reclaim budget. Formal learned/same-init trained CE/independent strong dynamic admission, whole-day10s/equal-time iteration and full scientific dependencies remain required before fresh science. Preserve old failures/missing13800 bytes.')
            obj['reproduction']['live_resume_saa_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [实际检查点后续与共同评分取消组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),runtime_checks=15,independent_checks=1757,failed=0,owned_processes=active)))
if __name__=='__main__':main()
