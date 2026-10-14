"""Archive new engineering evidence; preserve previous current-entry bytes."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';V1=R/'pickup/parent_decode_guard_shadow_root_v1';OUT=R/'pickup/parent_decode_guard_shadow_root_v2'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    text=x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def main():
    original=read(V1/'RESULT.json');raw=read(OUT/'RESULT.json');reg=read(OUT/'REGISTRATION.json');audit=read(OUT/'INDEPENDENT_PARENT_SHADOW.json')
    assert original['checks_count']==35 and original['failed']==2 and raw['checks_count']==11 and raw['failed']==audit['failed']==0
    assert audit['producer_sha256']==sha(OUT/'RESULT.json') and audit['registered_sha256']==sha(OUT/'REGISTRATION.json')
    assert all(sha(p)==h for p,h in reg['sources'].items())
    active=[dict(pid=p.pid,cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any('qualify_pickup_parent_decode_guard_shadow_v' in x or 'audit_pickup_parent_decode_guard_shadow_v' in x for x in p.info['cmdline'] or [])]
    assert not active
    frozen={}
    for p,h in reg['sources'].items():
        if Path(p).parent==ROOT/'tools':
            target=OUT/('FROZEN_'+Path(p).name);shutil.copy2(p,target);assert sha(target)==h;frozen[str(target)]=h
    walls=[x['wall_s'] for x in raw['records']];faultwalls=[x['wall_s'] for x in original['records'][2:]]
    closure=[read(OUT/('SOURCE_START_'+str(n)+'.json')) for n in (0,1)]
    for n,begin in enumerate(closure):
        end=read(OUT/('SOURCE_END_'+str(n)+'.json'));assert begin['files']==end['files'] and begin['module_files']==end['module_files']
    metrics=dict(original_runtime_checks=35,original_runtime_failed=2,normal_recovery_runtime_checks=11,normal_recovery_failed=0,independent_checks=audit['checks_count'],independent_failed=0,normal_requests=2,original_fault_stages=6,original_faults_rerun=False,normal_wall_s=walls,fault_cancel_and_reap_wall_s=faultwalls,normal_cap_s=10.,normal_reserve_s=2.,fault_cap_s=.4,fault_reserve_s=.15,loaded_and_registered_file_counts=[len(c['files']) for c in closure],loaded_module_file_counts=[len(c['module_files']) for c in closure])
    report=E/'01_取货线/文档/结果与进度/取货_封存响应可中止解码与私有状态提交保护组件_2026-10-12.md'
    text=f'''# 封存响应可中止解码与私有状态提交保护组件（2026-10-12）

本阶段只处理已观察日后续的两个已封存端点响应：固定首/末callback15/80，选择依据为序列端点而非数值或结果。旧接拒、共同K10、完整C0、求解器、物理事件、模型和训练均未重新执行，不生成科学日，不重跑472/4711或旧QA。新独占pickup/parent_decode_guard_shadow_root_v1在两个正常请求和六个延迟入口前登记来源、计时与有限范围。

每个请求的旧报文/响应在计时前写入有界2MiB共享输入，解释器、一次输入解码及真实C0状态副本初始化在外。可终止worker在组件计时内执行第二次可信pickle解码、原公开报文/快照摘要、实际JointPolicyOwner.begin及commit保护条件、实际私有副本单指针_policy发布和32字节摘要返回。原JointPolicyOwner与安装模块磁盘字节未改；进程内包装只插入阶段标记/延迟。调用端只接收固定32字节摘要并更新候选缓存标记，不解码或安装真实仿真计划。这是私有副本上的实际guard/指针组件，不能声称把真实车辆执行所有权交给worker。已封存响应仅将deadline等值类型重定时至本组件的前瞻工程截止点，不重新认证C0，不用于实际现场发布。

六个故障分别到达输入解码、公开摘要、owner.signature、prepared状态摘要、私有指针发布前和返回摘要入口；延迟2秒，组件总0.4秒、执行窗0.25秒、终止回收预留0.15秒。实际全部取消并回收，耗时{min(faultwalls):.6f}–{max(faultwalls):.6f}秒，调用端仍为旧缓存且无返回结果。第六阶段在子进程私有指针已更新后暂停，取消通过销毁该副本隔离；不能声称所有故障均发生在指针更新前，也不声称真实现场状态已经具备这套事务所有权。仅这些已到达入口负载被验收，不外推任意阶段负载或硬实时保证。

## 原失败与窄范围环境恢复

v1运行35项中2项环境闭包失败，正常guard/摘要与六个取消边界均通过；原RESULT、RUN_EXIT=1、起止清单、源码及全部取消记录保持原字节。两个正常进程原摘要操作第一次调用时懒加载scripts/correction_engineering/audit_io.py及既存pyc，源文件本身无变化。故障并非数值或guard失败，不修改全局依赖。

保存v1四份原工具后，v2另目录/事前登记同两个端点、原guard、掩码无关输入和原计时参数，只在起始封存前初始化实际audit_io。六个已完成故障不再运行。两正常请求11项0失败，真实计时含worker解码/guard/封存结束与返回、调用端固定接收及本人进程回收，{min(walls):.6f}–{max(walls):.6f}秒；总10秒、执行窗8秒、预留2秒。输入写入/日初副本设置均未计入此组件，因而不能把该时长称完整请求时长。

两个正常进程实际已加载module文件/既存字节码、内存映射且有磁盘身份的native/DLL、解释器与登记输入源联合清单各{len(closure[0]['files'])}文件，开始结束无变化或新增；实际module文件映射各{len(closure[0]['module_files'])}项。无磁盘映射标签及缺失可选pyc另记，不伪造身份。仅本封存副本工作负载的实际文件闭包完成，父现场公开投影/编码、真实控制全链、独立进程全部未来加载依赖和科学完整闭包未因此合格。

## 独立验收与后续边界

另进程INDEPENDENT_PARENT_SHADOW.json共{audit['checks_count']}项0失败：不重跑解码/guard/求解器，用原端点工件独立canonical状态组装与SHA重算、原六取消记录、精确时间差、固定执行窗/回收预留、实际已退出本人worker命令身份，以及全部登记/加载文件逐SHA核对。资格是v1其余33成功项加v2两正常11项及本独立证据的组合，原35/2FAIL不能改成0FAIL。新独立审计不等再次执行guard，也不是另一物理认证实现。

本地Python3.12.7/NumPy1.26.4/CPU7，无服务器/GPU或共享MD上传。实际本人进程已退出；完整前后读取原输入SHA保持。父可中止公开投影/编码、真实候选及事件所有权切换、现场提交与完整请求回收、全链加载依赖仍需接线验收。正式学习提议、同初始化已训CE、独立强动态接拒与完整10秒控制仍待；全部工程屏障满足才另登记有限新鲜科学日，最终20260927/28禁用。配送第2族FAIL封闭，不采集/训练/审计/DEV/确认重跑；第3族未启动，余30错配/动作评分继续。负结果31必要门/183历史/C1原件不重复整理。

稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。
'''
    create(report,text)
    boundary=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,prepared_shadow_component_qualified=True,actual_shadow_guard_pointer_qualified=True,normal_shadow_loaded_files_qualified=True,live_owner_handoff_qualified=False,parent_public_projection_encode_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,new_scientific_data=False,stable_online_gain=False,shared_MD_uploaded=False,v1_result_sha256=sha(V1/'RESULT.json'),registration_sha256=sha(OUT/'REGISTRATION.json'),result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_PARENT_SHADOW.json'))
    create(R/'reproduction/PARENT_SHADOW_COMPONENT_EVIDENCE_20261012.json',boundary)
    backup=R/'current_entries_before_parent_shadow_20261012';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_observed_day_extension_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note=f'\n> 2026-10-12取货可中止封存响应组件：固定两个旧端点的worker解码/原guard/私有指针和固定摘要返回已验收，原六实际入口取消含回收均落0.4秒；原35/2懒加载闭包FAIL保留，v2仅补两正常11项0失败，独立{audit["checks_count"]}项0失败。原audit_io预加载后两个正常文件闭包一致；没有真实现场所有权切换或全请求资格，父投影编码/正式模型CE/强动态和完整10秒仍待，无收益。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(parent_decode_guard_shadow_directory=str(OUT.relative_to(ROOT)),prepared_shadow_component_qualified=True,normal_shadow_loaded_files_qualified=True,parent_shadow_original_checks=35,parent_shadow_original_failed=2,parent_shadow_recovery_checks=11,parent_shadow_recovery_failed=0,parent_shadow_independent_checks=audit['checks_count'],parent_shadow_independent_failed=0,live_owner_handoff_qualified=False,parent_public_projection_encode_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,
                next='Wire truly interruptible public projection/encode and live event/vehicle ownership with actual commit, whole-request cleanup reserve and loaded dependencies. Two sealed response private replicas passed decode/actual guard/pointer with fixed capsule; no live ownership promotion. Keep original35/2 FAIL and v2 normal11 plus independent evidence; do not repeat six faults or completed old QA. Formal model and same-initialization trained CE plus independent strong dynamic admissions still missing; fresh scientific days only after all barriers. Delivery family2 closed FAIL, family3 not started.')
            s['reproduction']['parent_shadow_boundary']='research/review_3080_20261010/reproduction/PARENT_SHADOW_COMPONENT_EVIDENCE_20261012.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [封存响应可中止解码与私有状态提交保护组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    assert all(sha(p)==h for p,h in reg['sources'].items())
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),boundary_sha256=sha(R/'reproduction/PARENT_SHADOW_COMPONENT_EVIDENCE_20261012.json'),registered_inputs_still_exact=True,owned_workers_active=active,full_live_request_qualified=False))
    print(json.dumps(dict(metrics=metrics,updated_entries=len(updates),stable_online_gain=False)))
if __name__=='__main__':main()
