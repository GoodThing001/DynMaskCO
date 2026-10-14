"""Archive normal restore-to-live scope without upgrading scientific qualification."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import sha
ROOT=old.ROOT;E=old.E;R=ROOT/'research/review_3080_20261010';O=old.P/'timed_restore_live_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    s=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(s)
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_TIMED_RESTORE_LIVE.json');assert(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(16,0,1778,0)
    assert all(sha(p)==h for p,h in read(O/'REGISTRATION.json')['sources'].items()) and all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    names={'qualify_pickup_timed_restore_live_v1.py','audit_pickup_timed_restore_live_v1.py'};owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not owned
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_5.json'));restored=decode(read(O/'RESTORE_IN_REQUEST.json'))
    metrics=dict(runtime_checks=16,independent_checks=1778,failed=0,walls_s=[h['wall_s'] for h in r['host']],selected_restore_s=restored['finished']-restored['started'],physical_events=len(actual['events']),writer_files=len(read(O/'SOURCE_START.json')['files']),writer_maps=len(read(O/'SOURCE_START.json')['module_files']),audit_files=len(read(O/'AUDIT_SOURCE_START.json')['files']),audit_maps=len(read(O/'AUDIT_SOURCE_START.json')['module_files']),records=[dict(callback=x['callback'],order=x['order'],decision=x['response']['scoring_packet']['admission_input']['current_decision'],clock=x['clock'],event=x['event_id'],replan=x['private_after']['replan_count'],services=sum(len(t.services) for t in x['private_after']['traces']),payload_size=x['payload_size'],payload_sha256=x['payload_sha256']) for x in actual['records']])
    frozen={}
    for n in (*sorted(names),'build_pickup_timed_restore_live_v1.py'):
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(production_actual_exit=0,independent_actual_exit=0,worker_actual_exit=r['actual_worker_exit'],owned_processes_active=owned,observed_from='exec_command/write_stdin actual completion'))
    report=E/'01_取货线/文档/结果与进度/取货_检查点恢复与现场连续请求同窗组件_2026-10-12.md'
    create(report,f'''# 检查点恢复与现场连续请求同窗组件（2026-10-12）

pickup/timed_restore_live_root_v1事前登记，实际从前阶段已选定callback20/event37/epoch3共享原载荷继续，不读取被取消epoch4私有分支来安装状态。解释器/原200历史/原80订单数据及仅旧epoch2环境预热在窗外；worker此前未恢复epoch3。第一请求计时开始后，实际读取父预置共享完整原payload及固定头，核SHA/epoch/base，pickle解码、恢复私有env/车队/served/轨迹/replanner/clock/event/replan，并继续原_run_loop。选定恢复段{metrics['selected_restore_s']:.9f}秒，处于同一个第一请求窗口；恢复本身0接拒/0评分/0物理推进，不再单开一个恢复预算窗。

首callback21/order37在前阶段只于未发布被丢弃副本接受，调用端权威并未接纳/推进该分支；本次从保留epoch3正常继续，同输入/原策略/种子处理仍真实未决的37，不是按效用重采或重跑旧16/2308取消QA。本次实际接拒/K10/原完整C0/编译/JointPolicyOwner保护与私有安装/原_plan_and_commit，接受且未换路线，clock6.703140735626221/event38/replan32/17服务；父在8秒截止前选择正常新epoch4票据并收回执。第一{metrics['walls_s'][0]:.9f}秒包含选定恢复、现场公共投影/编码解码、全部内生计算、提交、检查点原字节留存/共享发布/调用端选择和小载荷归档，保留resident未测该请求回收。

同一resident/env连续处理下一实际callback22/order16，不恢复第一旧起点。仍own_uncond_dynamic当前-only、own_incumbent同计划、原共同K10每请求20完整分支及原C0/编译/guard/实际计划提交，接受且未换路线，clock6.980449199676514/event40/replan34/18服务。两请求合计3个实际原物理推进，第二{metrics['walls_s'][1]:.9f}秒包含最终正常进程退出与回收、原共享载荷归档；不能称每请求均测回收。没有新延迟/取消故障，也没有把旧取消边界QA重做。

16运行检查、另进程1778独立项0失败，实际退出0。两份writer/shared/envelope实际原字节分别28138/28077字节逐字一致，SHA26a50225b734acbae4e912233c0cdd70bab094f9f1fb8b522281289d3fb467f7、62f39e33eeade276e236060d0a34c9dd2b5016334e6f5f3c2bb6c6ed124548bf。实际pickle对象memo不同可导致新载荷长度与旧取消副本不同；这里只验新实际writer/shared逐字，不声称旧缺失13800字节找回。完整私有状态/原路径字段与新独立原接拒/全部40共同分支/NumPy均值/原全C0/原3物理事件/公共剩余后缀/恢复源字段及时间/全部实际加载身份和进程退出均核验；不用新engine/projector/checkpoint worker作为参考，物理和canonical库共享原实现，不是第二物理模型。

生产实际517文件263module映射、独立审计516文件251映射起止一致，仅本无模型工作负载。父初始共享载荷预置及运行库预热仍在首请求外；父同步小载荷归档和固定共享控制未证明任意父阶段/慢磁盘可抢占。仅正常恢复后两请求，不覆盖恢复途中取消、任意发布/回收负载、父崩溃/持久化/撕裂发布/外部不可逆动作或全日完整控制。

只关闭“选定检查点恢复后真正继续现场处理，恢复和第一内生计算处于同一10秒窗口”这一有限工程缺口。full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False；正式学习/同初始化已训CE、独立强动态接拒、全日完整10秒和完整等耗时迭代仍待，稳定可归因MaskCO核心在线增益未建立。下一步另事前工程登记完善恢复与其它计算阶段取消/保留/回收，再接齐正式模型和强动态控制与完整科学依赖；全部屏障满足后才另登记有限新鲜研究日，不重复本16/1778或旧完成QA。

原checkpoint10890/4FAIL与旧缺失字节、continuous v1及审计v2启动FAIL均保持。配送第2族FAIL封闭不训练/重采/重审计/DEV/确认，第3族未启动、余30错配动作继续；负结果31必要门/183/C1原件不重整。16本地入口先备份更新，本人进程退出；无服务器/GPU/共享MD上传，最终seed20260927/28未用。
''')
    boundary=R/'reproduction/TIMED_RESTORE_LIVE_EVIDENCE_20261012.json';create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),metrics=metrics,registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_TIMED_RESTORE_LIVE.json'),scoped_timed_restore_live_component_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False,new_scientific_data=False))
    backup=R/'current_entries_before_timed_restore_live_20261012';backup.mkdir(exist_ok=False);updates=[]
    note='\n> 2026-10-12取货恢复与现场请求同窗组件：从选定epoch3原共享载荷在首请求计时内实际解码恢复（.015266秒）并正常处理此前取消未提交callback21/order37，随后同resident实际callback22/order16；两单接受/未换路线、3推进，3.371245/3.764415秒，第二含最终回收。16运行/1778独立项0失败，仅两个正常请求，无新故障/旧QA重做。全10秒/正式学习CE强动态/迭代及核心在线增益仍未建立。\n'
    for x in read(R/'current_entries_before_checkpoint_resume_cancel_20261012/UPDATE_IDENTITY.json')['files']:
        f=E/x['path'];assert sha(f)==x['after_sha256'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(timed_restore_live_directory=str(O.relative_to(ROOT)),timed_restore_live_runtime_checks=16,timed_restore_live_independent_checks=1778,timed_restore_live_independent_failed=0,scoped_timed_restore_live_component_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat16/1778 or old completed QA. Next prospective engineering must close selected restore and other computation cancellation stages/preservation/reap; then integrate formal learned/same-init trained CE/independent strong dynamic admission, whole-day10s and full scientific dependency identity before fresh science. Current normal same-window restore plus callback21/22 only. Preserve prior discarded branch and all old FAIL/missing13800 bytes.')
            obj['reproduction']['timed_restore_live_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [检查点恢复与现场请求同窗](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert len(updates)==16 and all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),metrics=metrics,owned_processes=owned)))
if __name__=='__main__':main()
