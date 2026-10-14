"""Archive two actual new cancellation entrances with narrow qualification."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import sha
ROOT=old.ROOT;E=old.E;R=ROOT/'research/review_3080_20261010';O=old.P/'restore_compute_cancel_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    s=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(s)
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_RESTORE_COMPUTE_CANCEL.json');assert(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(13,0,1684,0)
    assert all(sha(p)==h for p,h in read(O/'REGISTRATION.json')['sources'].items()) and all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    names={'qualify_pickup_restore_compute_cancel_v1.py','audit_pickup_restore_compute_cancel_v1.py'};owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not owned
    frozen={}
    for n in (*sorted(names),'build_pickup_restore_compute_cancel_v1.py','build_pickup_restore_compute_cancel_audit_v1.py'):
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    _,decode=old.codecs();w=decode(read(O/'CASE_2/FAULT_WITNESS.json'));p=w['details']['packet'];resp=w['details']['response'];metrics=dict(runtime_checks=13,independent_checks=1684,failed=0,walls_s=[h['wall_s'] for h in r['host']],source_epoch=5,attempt_sequence=6,proposed_order=p['admission_input']['current_order'],proposed_decision=resp['scoring_packet']['admission_input']['current_decision'],private_reveal_clock=p['snapshot']['clock'],private_event=p['snapshot']['event_id'],private_replan=p['snapshot']['replan_count'],discarded_private_physical_events=len(w['details']['events']),caller_new_publications=0,writer_files=[len(read(O/('CASE_'+str(i))/'SOURCE_START.json')['files']) for i in (1,2)],writer_maps=[len(read(O/('CASE_'+str(i))/'SOURCE_START.json')['module_files']) for i in (1,2)],audit_files=len(read(O/'AUDIT_SOURCE_START.json')['files']),audit_maps=len(read(O/'AUDIT_SOURCE_START.json')['module_files']))
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(production_actual_exit=0,independent_actual_exit=0,cancelled_worker_exit_codes=[h['exit_code'] for h in r['host']],owned_processes_active=owned,observed_from='exec_command/write_stdin actual completion; worker cancellation return codes distinct from supervisor qualification exit'))
    report=E/'01_取货线/文档/结果与进度/取货_选定恢复途中与计算后安装前取消组件_2026-10-12.md'
    create(report,f'''# 选定恢复途中与计算后安装前取消组件（2026-10-12）

pickup/restore_compute_cancel_root_v1事前登记，固定两份新私有副本，来源为timed_restore_live_root_v1实际已提交callback22/event40/epoch5共享原载荷。两次均总10秒/执行8秒/预留2秒、固定故障延迟12秒，非模型/科学数据/新收益；不重跑旧16/1778正常请求、16/2308取消或其它完成QA。父初始共享载荷预置、运行库/历史/原80订单数据及仅旧epoch3环境预热在计时外，选定epoch5恢复在请求计时内。

第一入口在实际原replanner.restore_state已完成，但选定env/车队/轨迹/served尚未全部恢复时暂停。实际选定全私有值树、已恢复replanner字段、恢复未完成标记与阶段/时钟保留，0接拒/0评分/0物理推进；工作副本此时可能混合选定replanner与预热旧env，不能称副本完整旧状态未变。8秒截止设置fence并kill/reap，{metrics['walls_s'][0]:.9f}秒含本人回收与小共享归档，无发布/回执/新票据，销毁半恢复副本。

第二入口正常完成选定epoch5恢复，继续原_run_loop至下一真实callback23/order7。实际私有副本经过2个原物理推进到clock7.237941741943359/event42/replan35；现场公共投影/编码解码、own_uncond_dynamic当前-only接拒提议接受、own_incumbent同计划提议、原200历史共同K10/20完整分支、原完整C0/编译已完成，source identity合格。完整输入/响应/原cert/原分支/私有现场状态和故障入口全部留存。随后在原JointPolicyOwner.commit安装前实际暂停：原owner.begin票据已创建，但尚未私有单指针安装、未调用本当前揭示后的_plan_and_commit、没有生成/发布新检查点。8秒截止kill/reap，{metrics['walls_s'][1]:.9f}秒含本人回收与共享归档。order7仍只是临时计算结果，调用端没有接受/拒绝发布；此前私有副本2次物理推进也不计调用端权威执行，不等真实外部不可逆操作。

两次实际共享INITIAL/RETAINED完整envelope逐字一致，payload逐字等于原epoch5实际共享文件（28077字节，SHA62f39e33eeade276e236060d0a34c9dd2b5016334e6f5f3c2bb6c6ed124548bf），原SHA/epoch/base/私有摘要核验，父权威保留callback22票据。这里保留的是完整封存票据/共享原字节；没有仍存活的旧resident，更没有新实际恢复并执行order7。未重复旧7/2630原字节组件或补造原缺失13800字节。

13运行、另进程1684独立项0失败，监督与独立进程实际退出0；两个故障worker按预声明被终止，原退出码另存，不伪称worker自然退出0。独立直接核原源状态/部分replanner恢复、原下一真实揭示与2私有物理事件、原接拒/全部20K10/NumPy均值/原C0/编译输出/公共剩余后缀/安装前实际私有字段、阶段耗时/原完整共享字节/原权威与实际进程退出，不以新engine/projector/checkpoint worker为参考。物理和canonical库共享原实现，不是第二物理模型。

每个生产计算至实际故障入口517加载文件/263module映射相同，独立513文件251映射起止闭合。因worker被kill，生产仅证明开始至故障入口的实际闭包，不能写为故障后自然结束完整闭包；延迟段实际终止且无迟到发布。仅这两个入口故障，不证明pickle/求解器/native内部任意指令点抢占、任意大小/负载/慢磁盘或完整科学控制链依赖。父同步小载荷复制归档未证明硬实时磁盘，父崩溃/持久化/撕裂发布/外部不可逆效果未验。

只两个固定取消入口及封存票据保留/本人回收组件合格，full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False。下一步将有界恢复/现场私有控制接回更完整事件链并处理剩余实际计算/发布阶段故障，接齐正式学习/同初始化已训CE与独立强动态接拒/全日10秒和科学完整依赖；全部屏障满足后才另登记有限新鲜研究日。不要重复本13/1684或旧完成QA，不按临时效用选样。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。

原所有失败/缺失载荷保留；配送第2族FAIL封闭不训练/重采/重审计/DEV/确认，第3族未启动、余30错配动作继续。负结果31必要门/183/C1原件不重整。16本地入口先备份更新，本人进程退出；无服务器/GPU/共享MD上传，最终seed20260927/28未用。
''')
    boundary=R/'reproduction/RESTORE_COMPUTE_CANCEL_EVIDENCE_20261012.json';create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),metrics=metrics,registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_RESTORE_COMPUTE_CANCEL.json'),scoped_two_cancel_entrances_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False,new_scientific_data=False))
    backup=R/'current_entries_before_restore_compute_cancel_20261012';backup.mkdir(exist_ok=False);updates=[]
    note='\n> 2026-10-12取货恢复/计算取消入口：固定epoch5两个新副本，恢复原replanner后/env恢复完成前，以及下一真实order7实际接拒/K10全20分支/C0编译后/安装前，12秒延迟均到达；8.016971/8.017962秒取消回收，0新票据/回执，共享完整epoch5原字节保留。第二副本2物理推进和临时接受order7不计调用端执行。13运行/1684独立项0失败，只这两个入口；全日/正式模型CE强动态/完整预算、迭代与核心在线增益仍未建立。\n'
    for x in read(R/'current_entries_before_timed_restore_live_20261012/UPDATE_IDENTITY.json')['files']:
        f=E/x['path'];assert sha(f)==x['after_sha256'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(restore_compute_cancel_directory=str(O.relative_to(ROOT)),restore_compute_cancel_runtime_checks=13,restore_compute_cancel_independent_checks=1684,restore_compute_cancel_independent_failed=0,scoped_two_cancel_entrances_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat13/1684 or old QA. New prospective engineering should reconnect bounded restore/live private ownership across more actual events and remaining computation/publication faults; formal learned/same-init trained CE/independent strong dynamic admission, whole-day10s and full scientific dependencies remain required before fresh science. Order7 only computed in discarded private branch, epoch5 caller ticket preserved; not an actual caller acceptance. Preserve old failures/missing13800 bytes.')
            obj['reproduction']['restore_compute_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [恢复途中与计算后安装前取消](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert len(updates)==16 and all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),metrics=metrics,owned_processes=owned)))
if __name__=='__main__':main()
