"""Archive prepared-only actual private owner evidence, preserve current entry bytes."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/private_owner_handoff_root_v1'
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
def create(p,x):
    text=x if isinstance(x,str) else json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def main():
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs();raw=read(OUT/'RESULT.json');reg=read(OUT/'REGISTRATION.json');audit=read(OUT/'INDEPENDENT_PRIVATE_OWNER.json');actual=decode(read(OUT/'OWNER_RESULT_0.json'))
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'])==(15,0,1548,0)
    assert audit['producer_sha256']==sha(OUT/'RESULT.json') and audit['registration_sha256']==sha(OUT/'REGISTRATION.json') and all(sha(p)==h for p,h in reg['sources'].items())
    active=[dict(pid=p.pid,cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(x).name in {'qualify_pickup_private_owner_handoff_v1.py','audit_pickup_private_owner_handoff_v1.py'} for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for p,h in reg['sources'].items():
        if Path(p).parent==ROOT/'tools':
            target=OUT/('FROZEN_'+Path(p).name);shutil.copy2(p,target);assert sha(target)==h;frozen[str(target)]=h
    a=actual['after_commit'];b=actual['after_primary_advance'];begin=read(OUT/'SOURCE_START_0.json');end=read(OUT/'SOURCE_END_0.json');assert begin['files']==end['files'] and begin['module_files']==end['module_files']
    metrics=native(dict(runtime_checks=15,runtime_failed=0,independent_checks=1548,independent_failed=0,current_order=actual['public_packet']['admission_input']['current_order'],normal_clock=a['clock'],normal_event_id=a['event_id'],next_physical_clock=b['clock'],next_event_id=b['event_id'],replan_counts=[a['replan_count'],b['replan_count']],service_counts=[sum(len(t.services) for t in a['traces']),sum(len(t.services) for t in b['traces'])],normal_wall_s=raw['records'][0]['wall_s'],fault_wall_s=[r['wall_s'] for r in raw['records'][1:]],public_encoded_bytes=actual['encoded_bytes'],loaded_and_registered_files=len(begin['files']),loaded_module_files=len(begin['module_files'])))
    report=E/'01_取货线/文档/结果与进度/取货_私有所有者公开投影编码与封存计划交接组件_2026-10-12.md'
    text=f'''# 私有所有者公开投影编码与封存计划交接组件（2026-10-12）

新独占pickup/private_owner_handoff_root_v1先登记，固定原observed_day_extension的首callback15封存响应，不按结果筛选。唯一正常交接加公开投影/编码两个新延迟入口；不是原六guard取消或15/80影子响应组件的重复QA。旧接拒、共同K10/SAA、完整C0认证、候选求解器、模型与训练没有重新执行，最终seed20260927/28未用，不生成新科学日/新研究族。

起点为原live_event_prefix第14次_plan_and_commit之后的实际封存状态。恢复真实私有全80订单环境/冷链车队/轨迹/served/决策与计数，直接原_run_loop推进到下一on_reveal入口即停止，当前订单仍未决，没有先写拒绝占位，也未再执行该入口接拒。这段解释器/私有恢复与当前点初始化在组件计时外；父旧所有者及三个worker恢复到相同实际当前点，clock={metrics['normal_clock']}/event_id={metrics['normal_event_id']}，公共当前订单{metrics['current_order']}。

计时开始后，worker从自己拥有的当前真实私有env投影未决公共输入，实际pickle编码再解码并核原封存公共摘要，复用原响应实际解码与JointPolicyOwner.begin/commit保护条件，原C0计划单指针安装；随后调用原StrictOnlineEnv._plan_and_commit真实处理车队。原响应的deadline只重定时到本工程截止点，不重新调用认证器，既存cert/source资格只来自原已封存证据。父端计时内只发固定共享控制/收32字节身份，不做公开投影/可信pickle解码/guard；正常收到精确原compiled SHA后，以一次process-reference选择该resident进程为下一实际所有者。

父随后实际命令选中的进程继续推进一个原自然物理事件，原事件最小值规则选择下一揭示/服务完成/返仓/预约截止；原_advance_fleet实际更新冷链/车队/served，继而停止于物理更新完成、下一控制处理之前。本例clock推进到{metrics['next_physical_clock']}/event_id={metrics['next_event_id']}，累计服务{metrics['service_counts']}，重规划计数{metrics['replan_counts']}。这一选择的是实际持有env/车队/决策的进程并由其继续物理推进，不是上一组件只更新摘要缓存。组件结束后正常所有者也退出回收，不称连续运行的全日远程控制器。

## 运行与独立证据

15运行项0失败。正常从固定启动控制到公开投影/编码/实际计划交接、所有者选择、一个物理推进及本人进程回收{metrics['normal_wall_s']:.6f}秒，总10秒/执行窗8秒/回收预留2秒。公开输入实际{metrics['public_encoded_bytes']}字节。公开投影和编码入口各延迟2秒，总0.4秒、执行窗0.25秒/回收预留0.15秒，实际{min(metrics['fault_wall_s']):.6f}–{max(metrics['fault_wall_s']):.6f}秒均取消回收，无返回响应/所有者选择；父原实际env状态摘要逐位不变，正常旧所有者也不被子副本改动。这两项只是已到达入口的有限负载验证，不证明任意系统负载下的硬实时。未测所有者选择后的迟到/故障、连续候选进程切换或完整现场回滚，因此总体现场所有权事务仍未资格。

正常worker实际已加载module源码/既存字节码、磁盘对应native/DLL、解释器与注册输入联合{metrics['loaded_and_registered_files']}文件/ {metrics['loaded_module_files']}module文件映射，开始结束一致、无新增；两个故障有实际开始清单，因终止没有结束闭包。此闭包仅本prepared-only所有者工作负载，未覆盖真实接拒/评分/模型科学控制链。

另进程INDEPENDENT_PRIVATE_OWNER.json共1548项0失败。它从同一原封存起点独立组装原C0环境，直接原_run_loop定位当前点，以原响应canonical状态字典独立SHA和restore_state安装参考计划，执行原_plan_and_commit/原下一自然物理事件；没有调用新投影器/交接程序/JointPolicyOwner作为参考，也没有重跑接拒/SAA/C0认证。逐公共已观察字段、原公共摘要、原私有起始身份、实际计划提交前后和物理推进的全部state/fleet/traces/served/clock/event/replan字段、预算时间、两取消无交接、全部源与加载文件逐SHA相符。共享原仿真/热学实现，不是另一物理模型。原15/80影子35/2FAIL及11/4064恢复、472/4711日后续等所有已完成QA原件不动、不重跑。

## 尚未闭合的现场请求

本阶段复用已存在的响应，不能声称内生新订单接拒/学习提议/共同K10/完整C0认证进入同一可中止事务。私有全环境预恢复/预加载在计时外；原接拒/评分本次并不发生。完整新请求、原真实事件所有权连续接线、全链依赖、每请求取消回收及所有者切换后故障仍待。正式模型、已训同初始化CE、独立强动态接拒和完整10秒控制仍未齐；工程屏障全满足后另登记有限新鲜科学日。

本地Python3.12.7/NumPy1.26.4/CPU7，本人进程已回收，无服务器/GPU/共享MD上传。配送第2族FAIL封闭、无采集/训练/审计/DEV/确认重跑，第3族未启动；余30错配/生产动作评分工程继续。负结果31必要门/183历史/C1原件不重整。

prepared-only私有所有者的公开投影/编码、原计划交接与一次实际物理推进组件通过。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
'''
    create(report,text)
    boundary=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,prepared_private_owner_handoff_qualified=True,actual_private_public_projection_encoding_qualified=True,actual_prepared_policy_commit_and_one_physical_advance_qualified=True,normal_owner_loaded_files_qualified=True,prepared_response_reused=True,new_admission_or_SAA=False,new_scientific_data=False,full_live_owner_transaction_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,shared_MD_uploaded=False,registration_sha256=sha(OUT/'REGISTRATION.json'),result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_PRIVATE_OWNER.json'))
    create(R/'reproduction/PRIVATE_OWNER_HANDOFF_EVIDENCE_20261012.json',boundary)
    backup=R/'current_entries_before_private_owner_handoff_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_parent_shadow_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货prepared-only私有所有者组件：原首callback15当前点worker公开投影/编码、原封存响应guard安装与真实_plan_and_commit，父选择resident所有者后实际推进一物理事件；15运行/1548独立项0失败，两新投影编码入口取消含回收落0.4秒、旧所有者不变。仅复用响应、初始化在外，内生接拒/K10/认证/模型、连续所有权与交接后故障未合成完整请求，无收益；原35/2等旧QA保持。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(private_owner_handoff_directory=str(OUT.relative_to(ROOT)),prepared_private_owner_handoff_qualified=True,actual_private_public_projection_encoding_qualified=True,actual_prepared_policy_commit_and_one_physical_advance_qualified=True,private_owner_runtime_checks=15,private_owner_independent_checks=1548,normal_owner_loaded_files_qualified=True,full_live_owner_transaction_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,
                next='Integrate endogenous live admission/proposal/shared K10/full C0/compile into the actual private environment owner; add continuous event ownership and post-promotion cancellation/rollback with whole-request cleanup reserve and loaded dependencies. Prepared-only callback15 projection/encode/actual plan commit plus one physical advance passed15/1548; do not repeat these or old QA. Formal learned/same-initialization trained CE/independent strong dynamic admissions still missing, fresh science only after barriers. Delivery family2 closed FAIL, family3 not started.')
            s['reproduction']['private_owner_handoff_boundary']='research/review_3080_20261010/reproduction/PRIVATE_OWNER_HANDOFF_EVIDENCE_20261012.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [私有所有者公开投影编码与封存计划交接组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates) and all(sha(p)==h for p,h in reg['sources'].items())
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),boundary_sha256=sha(R/'reproduction/PRIVATE_OWNER_HANDOFF_EVIDENCE_20261012.json'),registered_inputs_still_exact=True,owned_workers_active=active,full_live_request_qualified=False))
    print(json.dumps(dict(metrics=metrics,updated_entries=len(updates),stable_online_gain=False)))
if __name__=='__main__':main()
