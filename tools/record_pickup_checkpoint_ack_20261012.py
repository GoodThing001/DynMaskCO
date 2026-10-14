"""Preserve runtime PASS and independent audit FAIL, no qualification upgrade."""
import hashlib,json,pickle,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=old.ROOT;E=old.E;R=ROOT/'research/review_3080_20261010';OUT=old.P/'checkpoint_ack_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def native(v):
    if isinstance(v,np.ndarray):return native(v.tolist())
    if isinstance(v,np.generic):return native(v.item())
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v
def create(p,v):
    s=v if isinstance(v,str) else json.dumps(native(v),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(s)
def main():
    raw=read(OUT/'RESULT.json');a=read(OUT/'INDEPENDENT_CHECKPOINT_ACK.json');reg=read(OUT/'REGISTRATION.json');ar=read(OUT/'AUDIT_REGISTRATION.json');encode,decode=old.codecs()
    assert (raw['checks_count'],raw['failed'],a['checks_count'],a['failed'])==(24,0,10890,4)
    assert a['failure_names']==['0:payload_identity','1:payload_identity','2:payload_identity','audit_loaded_closure']
    assert all(sha(p)==h for p,h in reg['sources'].items()) and all(sha(p)==h for p,h in ar['inputs'].items())
    s=read(OUT/'AUDIT_SOURCE_START.json');e=read(OUT/'AUDIT_SOURCE_END.json');new=sorted(set(e['files'])-set(s['files']));changed=[p for p,h in s['files'].items() if e['files'].get(p)!=h]
    demonstration=[]
    for i in range(3):
        row=decode(read(OUT/('PHYSICAL_ATTEMPT_'+str(i)+'.json')));c=row['checkpoint'];rebuilt=pickle.dumps(c,protocol=5);roundtrip=decode(encode(c));twice=pickle.dumps(roundtrip,protocol=5)
        demonstration.append(dict(case=i,original_logged_size=row['payload_size'],original_logged_sha256=row['payload_sha256'],tree_rebuilt_size=len(rebuilt),tree_rebuilt_sha256=hashlib.sha256(rebuilt).hexdigest(),second_tree_sha256=hashlib.sha256(twice).hexdigest(),semantic_digest_before=old.pf.digest(c),semantic_digest_after=old.pf.digest(roundtrip),semantic_tree_equal=encode(c)==encode(roundtrip)))
    diag=dict(created_utc=datetime.now(timezone.utc).isoformat(),failure_names=a['failure_names'],original_audit_checks=10890,original_audit_failed=4,successful_other_checks=10886,source_changed=changed,new_loaded_files=new,new_module_files=sorted(set(e['module_files'])-set(s['module_files'])),demonstration=demonstration,interpretation='Original shared pickle bytes were not archived. An audit JSON tree is value lossless but does not carry Python object alias/memo identity, so a rebuilt pickle is not an independently recovered original byte stream. Specific cause of original size/hash difference has not been uniquely isolated. New SHCore.dll appeared in audit loaded maps, existing source files unchanged; its first-load cause not uniquely located. Both original barriers remain FAIL.',raw_payload_identity_qualified=False,independent_audit_qualified=False,full_live_request_qualified=False)
    create(OUT/'FAILURE_MECHANISM_DIAGNOSIS.json',diag)
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in ('qualify_pickup_checkpoint_ack_v1.py','audit_pickup_checkpoint_ack_v1.py','run_pickup_checkpoint_ack_bounded_v1.py') for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for name in ('qualify_pickup_checkpoint_ack_v1.py','run_pickup_checkpoint_ack_bounded_v1.py','audit_pickup_checkpoint_ack_v1.py'):
        f=ROOT/'tools'/name;dest=OUT/('FROZEN_'+name);shutil.copy2(f,dest);frozen[str(dest)]=sha(f);assert sha(dest)==sha(f)
    recovery=[read(OUT/('RECOVERY_HOST_'+str(i)+'.json')) for i in (0,2)];metrics=dict(runtime_checks=24,runtime_failed=0,independent_checks=10890,independent_failed=4,independent_other_successful=10886,walls_s=[r['wall_s'] for r in raw['records']],recovery_walls_s=[r['wall_s'] for r in recovery],clock=a['clock'],event_id=a['event_id'],service_count=a['service_count'],normal_files=len(read(OUT/'SOURCE_START_0.json')['files']),normal_modules=len(read(OUT/'SOURCE_START_0.json')['module_files']),payload_bytes=13800)
    report=E/'01_取货线/文档/结果与进度/取货_物理推进后回执丢失检查点组件与独立屏障失败_2026-10-12.md'
    text=f'''# 物理推进后回执丢失检查点组件与独立屏障失败（2026-10-12）

pickup/checkpoint_ack_root_v1事前登记，仅从原callback17/event26封存实际私有状态推进下一自然物理边界。三个新工程副本：正常、实际推进完成后/发布前延迟12秒、完整检查点发布后/回执前延迟12秒。24运行项0失败，独立10890项4FAIL原件保留，其余10886项相符；整体独立验收未通过，不能升级physical_ack_loss/full_live_request资格。没有接拒、模型、提议、SAA、完整C0重新认证或科学数据/收益，不重复之前已合格的两个揭示请求。

## 实际行为和时间

原StrictOnlineEnv._advance_fleet在完整私有80数据/实际车队与轨迹恢复实例上运行一次，clock推进至{a['clock']}、event={a['event_id']}，累计服务{a['service_count']}。这里只处理原自然物理边界，不处理该时点揭示订单或再次_plan_and_commit。三个副本的原物理字段逐项相符，未将仿真推进称现实硬件不可逆交付。

新完整私有检查点上限2MiB，实际13800字节。worker自己序列化、计算SHA、写完整共享载荷和固定epoch/base/digest头，再用命名Win32事件发布。父只非阻塞读取事件与固定头，核epoch/base，在8秒执行截止前一次选择权威票据；未选择的私有推进副本可销毁，旧封存状态保持。完整检查点候选发布与权威票据选择分开，父选择才是本组件逻辑提交。

正常含本人回收{metrics['walls_s'][0]:.6f}秒；实际推进后尚未发布的延迟取消{metrics['walls_s'][1]:.6f}秒，父保留旧摘要、无票据/回执；实际检查点已发布后丢失回执{metrics['walls_s'][2]:.6f}秒，父在截止前选定新票据并终止回收延迟本人worker，完整共享载荷保留供另一进程恢复。三个窗总10秒/执行8秒/预留2秒；均实测含回收低于10秒，单小负载不是任意负载硬实时保证。延迟确实在物理推进之后，不能称故障均发生在物理动作之前。

正常与丢回执副本各新进程在独立恢复请求中实际解码共享载荷、验证其SHA/epoch/base、恢复完整私有环境/车队/轨迹/served/replanner，返回原已提交状态摘要。恢复窗分别{metrics['recovery_walls_s'][0]:.6f}/{metrics['recovery_walls_s'][1]:.6f}秒含本人回收；识别原票据并直接返回当前摘要，0物理推进，不执行接拒/SAA。它不是第二个真实在线订单请求，也不能称通用重复请求协议或完整resident exactly-once。解释器/原完整数据预加载和最初私有恢复在组件窗外；恢复窗预热运行库在外，checkpoint解码与实际私有恢复在内。恢复请求是单独10秒窗，不能把它与原8秒取消合计声称同一个10秒请求。

生产正常及两个故障计算段、两个恢复段实际加载文件起止一致（正常{metrics['normal_files']}文件/{metrics['normal_modules']}模块映射），只本无学习负载，不等完整科学链封存。旧状态由封存完整字典保留，权威引用选择为共享检查点票据；本组件不是原同一resident所有者连续控制器，尚未把检查点协议接回内生接拒/评分事务。

## 独立失败和证据边界

另进程参考不调用新checkpoint工作器、恢复器或票据发布器，直接原C0ReserveReplanner.restore_state/原环境恢复和原_advance_fleet，独立计算下一自然时点、逐位比较完整私有数组/数据类/浮点/轨迹/状态、用独立编码器复核摘要、时长、epoch/base、实际进程退出、生产/恢复加载文件及输入身份。共享原物理实现，不是另一物理模型。

原INDEPENDENT_CHECKPOINT_ACK.json四FAIL：三项payload_identity把JSON树反解后的checkpoint重新pickle，要求原生产载荷字节身份；另一audit_loaded_closure发现审计自身新增C:/Windows/System32/SHCore.dll，既有文件/模块无变化，但缺该DLL起始加载身份。原报告/冻结源码保留，失败不写0。

原共享载荷实际被恢复worker读取校验，运行记录保存其size/SHA及恢复后全部字段；但生产没有把原共享pickle字节另存原件。JSON显式编码保留值类型而不封存原Python对象别名和pickle memo，重建字节不能当原载荷独立恢复。新只读FAILURE_MECHANISM_DIAGNOSIS.json保留三组原记录与重建size/SHA，并展示树往返语义摘要一致；未唯一定位原字节差异的对象层原因，不补造原载荷、不更改检查规则或据此宣布原payload gate通过。SHCore.dll首次加载具体调用亦未唯一定位，不把无既有源变化升级审计闭包PASS。没有重跑原生产或独立复算。

下一步只针对未闭合的原始检查点字节保存/独立载荷核验及审计实际DLL起始封存，在新的独占工程注册与完整原件保存条件下恢复；旧24/10890本轮不重复，旧FAIL不改。然后才能将本逻辑提交/恢复协议接回完整内生事务、连续现场事件和提交后的故障/取消/重复处理，验证完整链计时与回收。磁盘持久化、父进程崩溃、撕裂发布/对抗篡改、真实外部不可逆效果和任意阶段负载未验。

正式模型/同初始化已训CE、独立强动态接拒、完整日10秒及正式控制链依赖继续缺；全部工程屏障合格后才另登记有限新鲜科学日。无服务器/GPU/共享MD上传，最终20260927/28未用。配送第2族FAIL封闭不训练/重采/重审计/DEV/确认，第3族未启动；负结果31必要门/183/C1原件保持。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
'''
    create(report,text);boundary=R/'reproduction/CHECKPOINT_ACK_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,runtime_component_behaviors_observed=True,independent_audit_qualified=False,raw_payload_identity_qualified=False,physical_ack_loss_qualified=False,full_live_request_qualified=False,external_irreversible_effects_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,new_scientific_data=False,registration_sha256=sha(OUT/'REGISTRATION.json'),runtime_sha256=sha(OUT/'RESULT.json'),independent_FAIL_sha256=sha(OUT/'INDEPENDENT_CHECKPOINT_ACK.json'),diagnosis_sha256=sha(OUT/'FAILURE_MECHANISM_DIAGNOSIS.json')))
    backup=R/'current_entries_before_checkpoint_ack_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_resident_two_events_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货物理推进后检查点/回执故障新组件：三个新副本实际推进event27，24运行项0失败；发布前取消保留旧票据、发布后丢ACK另进程恢复不再推进。原独立10890项4FAIL保持（3原pickle载荷未归档/重建字节不一致、1审计新增SHCore.dll缺起始身份），10886其余项相符；不升级完整验收/在线收益。完整内生连续事务、正式CE/强动态及全10秒仍待。\n'
    for item in entries:
        f=E/item['path'];dest=backup/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(checkpoint_ack_directory=str(OUT.relative_to(ROOT)),checkpoint_runtime_checks=24,checkpoint_runtime_failed=0,checkpoint_independent_checks=10890,checkpoint_independent_failed=4,checkpoint_independent_qualified=False,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,next='Preserve checkpoint_ack_root_v1 24 runtime/10890 independent 4 FAIL. New exclusive prospective engineering must retain actual raw shared pickle bytes before disposal and seal audit actual DLL initial identity; no original payload reconstruction or gate relaxation, no repetition of completed old QA. Then integrate checkpoint logical commit/recovery into endogenous admission/K10/C0 and continuing private owner, fault/duplicate/cleanup and complete formal chain. Formal learned/same-init trained CE/independent strong dynamic admissions/full-day10s remain missing; no fresh science until barriers.')
            obj['reproduction']['checkpoint_ack_boundary']='research/review_3080_20261010/reproduction/CHECKPOINT_ACK_EVIDENCE_20261012.json';updated=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [物理推进后检查点组件与独立FAIL](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=item['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,original_independent_FAIL_preserved=True,full_live_request_qualified=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    print(json.dumps(dict(updated_entries=len(updates),metrics=metrics,independent_qualified=False,owned_processes=active)))
if __name__=='__main__':main()
