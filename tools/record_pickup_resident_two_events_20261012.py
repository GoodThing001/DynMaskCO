"""Archive combined qualification, keeping original independent FAIL files."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil,numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/resident_two_events_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def native(x):
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    return x
def create(p,v):
    text=v if isinstance(v,str) else json.dumps(native(v),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def main():
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');actual=decode(read(OUT/'RESIDENT_RESULT.json'));audit=read(OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json');r2=read(OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json');r3=read(OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R3.json')
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'],r2['checks_count'],r2['failed'],r3['checks_count'],r3['failed'])==(14,0,1216,1,29,1,30,0)
    assert audit['failure_names']==['new:1:original_private_admission_before'] and r2['failure_names']==['original_stored_physical_service52']
    assert audit['producer_sha256']==r3['producer_sha256']==sha(OUT/'RESULT.json') and audit['registration_sha256']==sha(OUT/'REGISTRATION.json') and all(sha(p)==h for p,h in reg['sources'].items())
    names={'qualify_pickup_resident_two_events_v1.py','audit_pickup_resident_two_events_v1.py','audit_pickup_resident_projection_recovery_r3.py'};active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])];assert not active
    frozen={}
    for p,h in reg['sources'].items():
        if Path(p).parent==ROOT/'tools':
            target=OUT/('FROZEN_'+Path(p).name);shutil.copy2(p,target);assert sha(target)==h;frozen[str(target)]=h
    for n in ('audit_pickup_resident_projection_recovery_r3.py','recover_pickup_resident_projection_r3.py'):
        source=ROOT/'tools'/n;target=OUT/('FROZEN_'+n);shutil.copy2(source,target);frozen[str(target)]=sha(source)
    frozen[str(OUT/'FROZEN_audit_pickup_resident_projection_recovery_r2.py')]=sha(ROOT/'tools/audit_pickup_resident_projection_recovery_r2.py')
    begin=read(OUT/'SOURCE_START.json');rows=actual['records'];metrics=native(dict(runtime_checks=14,runtime_failed=0,original_independent_checks=1216,original_independent_failed=1,original_successful_checks=1215,narrow_R2_checks=29,narrow_R2_failed=1,narrow_R3_checks=30,narrow_R3_failed=0,orders=[r['order'] for r in rows],accepted=[r['order'] in r['after']['accepted'] for r in rows],walls_s=[h['wall_s'] for h in raw['host']],physical_events=len(actual['events']),clocks=[r['clock'] for r in rows],event_ids=[r['event_id'] for r in rows],replan_counts=[r['private_after']['replan_count'] for r in rows],service_counts=[sum(len(t.services) for t in r['private_after']['traces']) for r in rows],loaded_files=len(begin['files']),module_files=len(begin['module_files']),pid=actual['pid']))
    report=E/'01_取货线/文档/结果与进度/取货_同一私有所有者两个连续揭示事务与投影核验恢复_2026-10-12.md'
    text=f'''# 同一私有所有者两个连续揭示事务与投影核验恢复（2026-10-12）

pickup/resident_two_events_root_v1事前登记，只从原第15次提交后实际event24封存状态继续两个新的真实揭示callback16/17，订单{metrics['orders']}。同一resident进程/同一私有env/车队/轨迹/served/决策状态持续推进，第二请求以第一实际提交后状态开始，没有恢复到旧快照、重复第15次接拒或筛选订单。14运行项0失败，两个请求均接受、同计划提议未换路线，无学习模型/增益/新科学日。

## 实际连续链

初始化从原endogenous_owner_transaction_root_v2/OWNER_RESULT_0.json的after_primary_advance读取原完整私有环境状态，原80节点数据只作为仿真私有真值。日初历史/解释器/私有恢复在两请求计时外。随后两个父固定共享控制启动窗中，worker直接原_run_loop推进到下一真实on_reveal，自己公开投影未决输入和编码解码、own_uncond_dynamic当前-only接拒、own_incumbent原计划提议、原200历史共同K10/20分支、完整C0/编译、实际原JointPolicyOwner单指针和原_plan_and_commit，记录固定32字节回执。父计时内不做公开投影/pickle/大状态guard。同一私有所有者保留第一结果并接续第二；现场公共输入由当前env生成，不依赖旧第16/17封存响应。

两个正常请求总10秒/执行8秒/回收预留2秒，计时包括实际事件推进、公共编码、完整接拒评分认证、依赖身份、实际车队提交、工件写入和固定回执；实际{metrics['walls_s'][0]:.6f}/{metrics['walls_s'][1]:.6f}秒。第一请求保持resident服务，不回收；第二含最终本人进程回收，因此不能称每请求都测了回收或故障余量。连续两次源开始/计算后/实际提交后{metrics['loaded_files']}文件、{metrics['module_files']}module映射一致。仅本无模型两请求负载，尚非正式模型链完整依赖资格。

clock={metrics['clocks']}、event={metrics['event_ids']}、重规划计数={metrics['replan_counts']}、累计服务={metrics['service_counts']}，两个实际物理推进全部留存。实际当前每次只有一个未决新单；未把旧派生双未决场景称真实同时揭示。正常循环在每次当前请求的真实计划提交后暂停等待下一固定命令，非完整日初至终局或所有事件全天候远程控制器。

## 独立结果与保留的核验器失败

原INDEPENDENT_RESIDENT_TWO_EVENTS.json为1216项1FAIL，原源码/报告/退出1与生产源字节冻结，不能改写0失败。其余1215项逐位相符：两次直接原UncondDynamic接拒、全部40个共同K10分支与NumPy1.26分组均值、原完整C0/关闭轨迹、直接公共字段、原实际run_loop/plan_and_commit、两物理事件和全部实际私有状态与终点、时间/源/本人进程。参考不调用新公共投影/事务/临时评分适配器，仍共享原物理实现。

唯一失败new:1:original_private_admission_before把原私有计划与公共剩余计划当成逐字同一对象。原封存event24已服务订单52，但私有_policy.plan仍保留52；下一on_reveal公共投影按served和不可变承诺重新组成后缀，移除52。接受集合、拒绝集合、预算、mode/status完全相同；接拒和物理结果不变。第一独立核验把“私有状态 == before == public.snapshot.state”串联，错误落在私有与公共状态的比较。

窄范围R2原29项1FAIL也保持：它已验证私有状态精确等于原sealed起点、非plan字段/独立剩余后缀正确，但误假定52是在本第一请求的新增物理event25刚服务。实际52是在起点所含原event24已服务，属于先前已合格物理推进，不应重跑该QA。新INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R3.json30项0失败，只读取原封存提交前后/当前工件及原AST，精确见证52已在起点served/原轨迹中；独立按当前served/承诺形成公共后缀，确认原_advance_fleet不访问replanner、原_run_loop在该唯一新揭示物理步与on_reveal之间尚未调用plan。没有执行接拒/求解器/SAA/原事件重放、修改投影器/生产源码或放宽容差。

组件资格由原1215成功项与R3窄范围私有/公共语义证据合成，原1216/1FAIL和R2 29/1FAIL不可写成全PASS。一次只读控制台输出NumPy int64 JSON序列化失败没有产生文件或改变结果，归档显式等值native转换；它也不是实验或效用重试。旧11/1144/1049、15/1548、35/2+11/4064、472/4711等QA不重复。

## 下一屏障

同一实际私有所有者两个新的连续正常请求通过限定组合资格；没有本阶段故障实验。物理推进/私有提交已发生后回执丢失、调用端与resident提交一致性、继续后的取消/回收与保留旧状态、完整日/正式模型/已训同初始化CE/独立强动态接拒仍待。正常小样本低于10秒不证明任意阶段或系统负载下硬实时。全部工程屏障满足后另登记有限新鲜科学日，不使用最终20260927/28。

本阶段CPU7/Python3.12.7/NumPy1.26.4，本人进程已退出，无服务器/GPU/共享MD上传。配送第2族FAIL封闭、三臂训练/重采/重审计/DEV/确认禁用，第3族未启动，余30错配与生产动作工程继续；负结果31必要门/183/C1封存保持。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
'''
    create(report,text)
    boundary=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,two_observed_resident_component_qualified=True,qualification='original1215 successful independent checks plus R3 narrow suffix evidence; original1216/1 and R2 29/1 FAIL preserved',physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,formal_learned_controls=False,stable_online_gain=False,new_scientific_data=False,registration_sha256=sha(OUT/'REGISTRATION.json'),result_sha256=sha(OUT/'RESULT.json'),original_independent_sha256=sha(OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json'),R2_sha256=sha(OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json'),R3_sha256=sha(OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R3.json'))
    bound=R/'reproduction/RESIDENT_TWO_EVENTS_EVIDENCE_20261012.json';create(bound,boundary)
    backup=R/'current_entries_before_resident_two_events_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_endogenous_owner_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货同一私有所有者新两个连续揭示：原第15次event24封存点继续callback16/17，订单77/56均接受、未换路线，14运行项0失败；原独立1216/1FAIL保留，1215其余项精确。私有已服务52前缀与公开剩余后缀区分经R3 30项0失败闭合；R2 29/1服务时点假设FAIL也保留，无接拒/评分/事件重跑。3.523153/4.010235秒，第二含最终回收；仅两个正常无模型请求，物理推进后回执丢失/故障和正式CE/强动态仍待，无稳定在线增益。\n'
    for item in entries:
        f=E/item['path'];saved=backup/item['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(resident_two_events_directory=str(OUT.relative_to(ROOT)),two_observed_resident_component_qualified=True,resident_two_runtime_checks=14,resident_original_independent_checks=1216,resident_original_independent_failed=1,resident_narrow_R3_checks=30,resident_narrow_R3_failed=0,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,next='Close original physical/commit completion versus lost or delayed acknowledgment, exactly-once resident/client agreement and preserved old state with cancellation/recovery budget. Two normal new actual callbacks16/17 in the same private env passed14 runtime plus original1215 successes/R3 30; original1216/1 and R2 29/1 FAIL preserved; do not repeat. Full day/formal learned/same-init trained CE/independent strong admissions and formal chain dependencies still missing; fresh science only after barriers. Delivery family2 closed FAIL, family3 not started.')
            s['reproduction']['resident_two_events_boundary']='research/review_3080_20261010/reproduction/RESIDENT_TWO_EVENTS_EVIDENCE_20261012.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [同一私有所有者两个连续事务](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=item['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),boundary_sha256=sha(bound),registered_sources_exact=True,owned_processes_active=active,original_independent_FAIL_preserved=True,full_live_request_qualified=False))
    print(json.dumps(dict(metrics=metrics,updated_entries=len(updates),stable_online_gain=False)))
if __name__=='__main__':main()
