"""Append engineering evidence and back up the sixteen current entry files."""
import json,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/endogenous_owner_transaction_root_v2'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def native(v):
    if isinstance(v,np.ndarray):return native(v.tolist())
    if isinstance(v,np.generic):return native(v.item())
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v
def create(p,v):
    text=v if isinstance(v,str) else json.dumps(native(v),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def main():
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();raw=read(OUT/'RESULT.json');reg=read(OUT/'REGISTRATION.json');audit=read(OUT/'INDEPENDENT_PRIVATE_OWNER.json');extra=read(OUT/'INDEPENDENT_STORED_FAULT_CLOSURE.json');normal=decode(read(OUT/'OWNER_RESULT_0.json'));fault=decode(read(OUT/'CANDIDATE_RESULT_1.json'))
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'],extra['checks_count'],extra['failed'])==(11,0,1144,0,1049,0)
    assert audit['producer_sha256']==extra['producer_sha256']==sha(OUT/'RESULT.json') and audit['registration_sha256']==sha(OUT/'REGISTRATION.json')
    assert all(sha(p)==h for p,h in reg['sources'].items())
    names={'qualify_pickup_endogenous_owner_transaction_v1.py','qualify_pickup_endogenous_owner_transaction_v2.py','audit_pickup_endogenous_owner_transaction_v2.py'}
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])];assert not active
    frozen={}
    for p,h in reg['sources'].items():
        if Path(p).parent==ROOT/'tools':
            target=OUT/('FROZEN_'+Path(p).name);shutil.copy2(p,target);assert sha(target)==h;frozen[str(target)]=h
    source=ROOT/'tools/audit_pickup_endogenous_owner_stored_fault_v1.py';target=OUT/('FROZEN_'+source.name);shutil.copy2(source,target);frozen[str(target)]=sha(source)
    start=read(OUT/'SOURCE_START_0.json');compute=read(OUT/'SOURCE_COMPUTE_END_0.json');end=read(OUT/'SOURCE_END_0.json');assert start['files']==compute['files']==end['files'] and start['module_files']==compute['module_files']==end['module_files']
    a=normal['after_commit'];b=normal['after_primary_advance'];response=normal['response'];order=normal['public_packet']['admission_input']['current_order']
    metrics=native(dict(runtime_checks=11,independent_normal_checks=1144,stored_fault_checks=1049,failed=0,current_order=order,current_accepted=order in a['state']['accepted'],route_changed=response['accepted_route'],normal_wall_s=raw['records'][0]['wall_s'],post_selection_fault_wall_s=raw['records'][1]['wall_s'],clock_before=normal['public_packet']['snapshot']['clock'],clock_after=b['clock'],event_after=b['event_id'],service_counts=[sum(len(t.services) for t in a['traces']),sum(len(t.services) for t in b['traces'])],replan_counts=[a['replan_count'],b['replan_count']],loaded_files=len(start['files']),module_files=len(start['module_files']),branches=sum(len(x['branch_values']) for x in response['common_scores']['candidate_scores'].values())))
    report=E/'01_取货线/文档/结果与进度/取货_内生接拒评分与私有所有者事务组件_2026-10-12.md'
    text=f'''# 内生接拒评分与私有所有者事务组件（2026-10-12）

pickup/endogenous_owner_transaction_root_v2完成固定已观察callback15/当前订单{order}的内生请求：实际私有环境公共提取、编码解码/原摘要、own_uncond_dynamic当前-only接拒、own_incumbent原计划提议、原200历史共同K10/20完整分支、完整C0认证、编译、原JointPolicyOwner保护安装、原_plan_and_commit真实车队处理、调用端选择实际所有者、一次原自然物理推进及回收。生产11项、另进程1144项0失败；追加仅对已存故障候选/依赖清单的1049项0失败，不重跑接拒/评分/求解器。正常{metrics['normal_wall_s']:.6f}秒，订单接受、未换路线，服务{metrics['service_counts']}，重规划计数{metrics['replan_counts']}，clock到{metrics['clock_after']}、event{metrics['event_after']}。没有学习模型、策略收益、新科学日或第3族。

## 固定方法与实际边界

先登记新独占工程目录，同一原首14提交后完整私有80订单状态，原_run_loop只搬运中间物理状态到下一on_reveal入口，仍未决，无拒绝占位。恢复/历史预加载/解释器初始化在请求计时外；未重复已合格前14接拒与评分。公共投影发生在worker实际拥有的env，而调用端计时内只固定共享控制与32字节回复，不做公共JSON/pickle编码、解码或大对象guard。此次响应由实际engine.compute内生生成，认证与source_qualified在实际原guard安装前完成，未复用封存响应认证或仅重定deadline。

原科学/物理参数、K10、共同seed202610111301、CPU7、Python3.12.7/NumPy1.26.4不变。生产方法仍显式own_incumbent同计划提议，不包含学习输出；无同初始化已训CE、正式五臂或独立强动态接拒。当前输入的公共摘要仍核已封存原请求身份，是固定已观察状态工程，不能称新鲜在线科学日。旧临时pending评价分区和原完整日覆盖门保持，完整日末必须pending为空。本请求其余pending为空，原完整C0字段与全部实际关闭轨迹可直接复算。

## 原失败与工程恢复

v1已登记并执行一次，首次真实接拒评分懒加载既存rolling_optimizer.py及既存pyc未记起始清单；其他起始源均不变。实际source_qualified=False，原JointPolicyOwner拒绝unqualified，未发布到调用端，未到所有者选择或新故障试验。原FAILURE/WORKER_FAILURE/退出1/开始与计算结束清单/冻结源码保持。v1没有在guard前保存完整响应，因此其原接拒/SAA数值输出不可追溯补造；此缺口明确保留。

核本人v1进程已退出后，v2事前封存相同输入和方法，新源码只预加载已定位模块并在原guard前保存完整响应尝试。没有改接拒、教师、种子、评价器、容差或按效用挑选；是缺失起始依赖身份的工程恢复，不是科学效用重试。v2两次正常计算段开始/结束实际{metrics['loaded_files']}联合文件与{metrics['module_files']}module映射一致；正常一次物理推进后结束也一致。包括实际加载源码/既存字节码、解释器、文件对应native/DLL与注册输入。仅封存本无模型单请求负载，不升级全部科学控制链依赖资格。

## 新交接后取消

同当前状态另一个固定副本正常完成内生接拒/评分/认证/安装及_plan_and_commit，父已选择该resident进程为实际所有者之后，在下一物理推进入口延迟12秒。总10秒、执行8秒、回收预留2秒；实际到达stage3，8秒执行窗结束时父恢复旧实际所有者引用、kill/reap销毁新私有分支，含回收{metrics['post_selection_fault_wall_s']:.6f}秒。原所有者完整env/state/fleet/traces/served摘要不变，无物理推进回执或后到OWNER_RESULT。它已在子私有分支安装计划和处理车队，因此不能称全部故障发生在提交前或从未选择所有者。

此回滚由保留旧当前点副本和隔离新私有分支实现，尚未处理对外不可逆操作、物理推进已开始/完成后回执丢失、连续跨事件所有者切换或任意负载。正常与故障的已存公共字段、接拒状态、20分支、完整C0/编译/实际提交后私有状态完全相同；额外1049项只复核已有工件与源，不重新执行旧QA。记录中继承的cancel_actual_stage_without_owner_promotion检查名称不表示从未选择：其布尔条件明确要求promotion存在、末端引用回滚。

## 独立范围与下一步

1144项独立另进程直接组装原UncondDynamic/原SAA，完整复算当前接拒、全部20分支/NumPy1.26分组均值、原完整C0轨迹、直接公共字段、canonical编译SHA、同实际私有基线原环境提交与自然物理推进；参考不调用新投影/事务程序/新pending适配器作为答案。共享原物理/仿真实现，不是另一物理模型。独立1144还包含实际开始文件哈希；1049补核两个计算段实际全部结束文件/module、故障已存响应及本人进程退出。原15/1548、35/2+11/4064、472/4711等QA不重复、不改写。

该单请求的无模型内生链和一次交接后、推进入口前取消组件通过；尚非连续全请求或硬实时证明。下一步连续实际事件所有权与物理推进后迟到/故障语义、预算回收和全链依赖，再完善正式模型/同初始化已训CE/独立强动态接拒；全部工程屏障满足后另登记有限新鲜科学日。最终20260927/28未用，无服务器/GPU/共享MD上传。配送第2族FAIL封闭、不训练/重采/重审计/DEV/确认，第3族未启动；负结果31必要门/183/C1封存不重整。

稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
'''
    create(report,text)
    boundary=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,endogenous_observed_owner_component_qualified=True,post_selection_pre_physical_entry_cancel_qualified=True,actual_loaded_single_request_qualified=True,full_live_request_qualified=False,continuous_live_owner_qualified=False,full_loaded_dependency_closure_qualified=False,formal_learned_controls=False,stable_online_gain=False,previous_failure_preserved=True,registration_sha256=sha(OUT/'REGISTRATION.json'),result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_PRIVATE_OWNER.json'),stored_fault_audit_sha256=sha(OUT/'INDEPENDENT_STORED_FAULT_CLOSURE.json'))
    bound=R/'reproduction/ENDOGENOUS_OWNER_COMPONENT_EVIDENCE_20261012.json';create(bound,boundary)
    backup=R/'current_entries_before_endogenous_owner_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_private_owner_handoff_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货内生私有所有者组件：实际公共提取→当前接拒→原计划提议→共同K10/全C0/编译→实际计划安装→所有者选择/一物理推进，11运行/1144独立项0失败；交接后推进入口延迟取消8.018522秒含回收落10秒，追加已存故障/依赖1049项0失败。v1懒加载rolling_optimizer源/pyc导致guard拒绝原件保留，v2同输入参数预加载恢复；单个已观察请求无模型/收益，连续/推进后故障与正式CE/强动态仍待。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(endogenous_owner_directory=str(OUT.relative_to(ROOT)),endogenous_observed_owner_component_qualified=True,post_selection_pre_physical_entry_cancel_qualified=True,actual_loaded_single_request_qualified=True,endogenous_runtime_checks=11,endogenous_independent_checks=1144,stored_fault_closure_checks=1049,full_live_request_qualified=False,full_live_owner_transaction_qualified=False,full_loaded_dependency_closure_qualified=False,next='Continuous actual event ownership; define delayed/failure semantics after physical advance and retention/rollback of old private owner within total budget, seal full formal workload. Single observed current69 endogenous own_uncond_dynamic/own_incumbent/K10/C0/compile/private owner and one physical advance11/1144 plus stored fault1049 passed; do not repeat. v1 lazy-load source-gate FAIL preserved. Formal model/same-init trained CE/independent strong admissions still missing; fresh science only after all barriers. Delivery family2 closed FAIL, family3 not started.')
            s['reproduction']['endogenous_owner_boundary']='research/review_3080_20261010/reproduction/ENDOGENOUS_OWNER_COMPONENT_EVIDENCE_20261012.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [内生私有所有者事务组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),boundary_sha256=sha(bound),registered_inputs_still_exact=True,owned_processes_active=active,full_live_request_qualified=False))
    print(json.dumps(dict(metrics=metrics,updated_entries=len(updates),stable_online_gain=False)))
if __name__=='__main__':main()
