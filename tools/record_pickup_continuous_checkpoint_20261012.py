"""Archive new two-request checkpoint scope, including preserved recovery failures."""
import json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil,numpy as np
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import sha
ROOT=old.ROOT;E=old.E;R=ROOT/'research/review_3080_20261010';O=old.P/'continuous_checkpoint_root_v2';F=old.P/'continuous_checkpoint_root_v1';D=old.P/'event27_settlement_diagnosis_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
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
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_CONTINUOUS_CHECKPOINT.json');_,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_2.json'));assert(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(17,0,2314,0)
    assert read(F/'RUN_EXIT.json')['exit_code']==1 and read(old.P/'checkpoint_ack_root_v1/INDEPENDENT_CHECKPOINT_ACK.json')['failed']==4
    for dr in (F,O):
        assert all(sha(p)==h for p,h in read(dr/'REGISTRATION.json')['sources'].items())
    assert all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    recovery=read(O/'AUDIT_BOOTSTRAP_RECOVERY_REGISTRATION.json');assert sha(recovery['new_audit_source'])==recovery['new_source_sha256']
    owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in ('qualify_pickup_continuous_checkpoint_v1.py','qualify_pickup_continuous_checkpoint_v2.py','audit_pickup_continuous_checkpoint_v3.py') for x in p.info['cmdline'] or [])];assert not owned
    frozen={}
    for n in ('qualify_pickup_continuous_checkpoint_v2.py','audit_pickup_continuous_checkpoint_v3.py','recover_pickup_continuous_checkpoint_v2.py','recover_pickup_continuous_audit_v3.py'):
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'AUDIT_RUN_EXIT_RECEIPT.json',dict(actual_exit_code=0,checks=2314,failed=0,source_sha256=sha(ROOT/'tools/audit_pickup_continuous_checkpoint_v3.py'),observed_from='exec_command/write_stdin actual process completion'))
    create(O/'AUDIT_V2_STARTUP_FAILURE_RECEIPT.json',dict(actual_exit_code=1,observed_traceback="Traceback (most recent call last):\n  File \"D:\\PyCharm_\\MASKCO-Main\\tools\\audit_pickup_continuous_checkpoint_v2.py\", line 6, in <module>\n    import hard_admission,rolling_optimizer\nModuleNotFoundError: No module named 'hard_admission'\n",before_audit_computation=True,source_sha256=sha(ROOT/'tools/audit_pickup_continuous_checkpoint_v2.py')))
    create(D/'RUN_EXIT_RECEIPT.json',dict(actual_exit_code=0,observed_from='exec_command completion',source_sha256=sha(ROOT/'tools/diagnose_pickup_event27_settlement_v1.py')))
    metrics=dict(runtime_checks=17,runtime_failed=0,independent_checks=2314,independent_failed=0,walls_s=[h['wall_s'] for h in r['host']],recovery_wall_s=r['recovery_host']['wall_s'],physical_events=len(actual['events']),records=[dict(callback=x['callback'],order=x['order'],decision=x['response']['scoring_packet']['admission_input']['current_decision'],clock=x['clock'],event_id=x['event_id'],replan_count=x['private_after']['replan_count'],services=sum(len(t.services) for t in x['private_after']['traces']),payload_bytes=x['payload_size'],payload_sha256=x['payload_sha256']) for x in actual['records']],writer_files=len(read(O/'SOURCE_START.json')['files']),writer_modules=len(read(O/'SOURCE_START.json')['module_files']),audit_files=len(read(O/'AUDIT_SOURCE_START.json')['files']),audit_modules=len(read(O/'AUDIT_SOURCE_START.json')['module_files']))
    report=E/'01_取货线/文档/结果与进度/取货_内生连续事务检查点与丢回执恢复组件_2026-10-12.md'
    create(report,f'''# 内生连续事务检查点与丢回执恢复组件（2026-10-12）

pickup/continuous_checkpoint_root_v2固定从原event27实际封存私有状态继续callback18/19：订单53接受、26拒绝，同一resident私有env/车队/轨迹/接拒决定连续，第二请求以第一实际提交结果开始。17运行项、另进程2314独立项0失败，实际退出0。仅两个已观察新工程请求，无模型/科学日/在线收益，未重跑旧callback15-17或7/2630、24/10890等已完成QA。

## 新实际事务与丢回执

worker计时内实际私有公开投影/编码解码、own_uncond_dynamic当前-only接拒、own_incumbent提议、原200历史共同K10每请求20完整分支、完整C0/编译/原JointPolicyOwner guard/私有单指针安装及原_plan_and_commit。两次均未换路线；strong_rolling与正式学习/已训同初始化CE未运行，不能把此接拒当独立强动态臂。原_run_loop继续7个实际物理事件。callback18 clock4.731592178344727/event31/replan26/13服务，callback19 clock5.317551136016846/event34/replan28/15服务；原有接拒与服务不重做。

每次实际提交后worker序列化完整私有检查点并独占保存原writer二进制，再写共享载荷/固定epoch、base、size、原字节SHA和私有摘要，经命名Win32事件发布。调用端仅读固定头，在8秒执行截止前一次选择新票据；结束该序号后另保存共享原envelope/payload，独立与writer逐字一致，分别18884/21456字节。调用端选择票据是本组件逻辑权威，原运行worker里私有安装与检查点候选发布不自动等于调用端提交。

第一正常请求{metrics['walls_s'][0]:.9f}秒含原字节归档，但保留resident，未测每请求回收。第二在完整内生提交/检查点已发布后实际延迟回执12秒，调用端已在截止前选定新票据；8秒截止设置fence并终止回收本人worker，{metrics['walls_s'][1]:.9f}秒含原共享归档与回收，无回执且保留已提交票据。不能称该故障发生在提交前，也没有重新计算接拒/SAA来找回结果。

另一恢复进程在独立10秒窗内实际读原共享载荷、验证SHA/epoch/base、pickle解码及恢复完整私有env/车队/served/轨迹/replanner，返回选定第二票据摘要。重复请求seq2被识别为已提交状态，0物理推进/0接拒/0评分；该窗含回收{metrics['recovery_wall_s']:.9f}秒。恢复窗与原8秒取消窗分开，不能合计声称同一个10秒请求。解释器/历史/原完整数据和起点恢复在日初工程预热外；恢复进程预热原初始env在外，实际选定检查点解码/恢复在窗内。尚未运行恢复后的第三个揭示请求，不能把重复票据返回等同恢复后全部连续控制合格。

生产两个计算/提交段及恢复段实际加载514文件/263module映射起止相同；审计自身522文件/251映射起止相同，SHCore.dll/process_iter在开始封存前预初始化。只此无模型负载，不等正式科学全链依赖。父固定头读、原共享小载荷归档仍是同步文件操作，测得低于10秒不证明任意大小父阶段/磁盘延迟可抢占或完整硬实时；磁盘持久化/父崩溃/撕裂发布/真正外部不可逆效果亦未验。

## 失败保存与工程恢复

原continuous_checkpoint_root_v1首次接入退出1：event27来源只做原_advance_fleet，缺同一原事件随后_plan_and_commit；直接继续下一揭示时engine在原admitted incumbent完整物理认证处拒绝，尚未K10/安装/检查点发布。原注册/源码/WORKER_FAILURE/FAILURE/RUN_EXIT保持，原未保存失败输入和base证书输出的缺口不补造。

新event27_settlement_diagnosis_root_v1事前登记固定同一原始状态，保留未补/补一次原_plan_and_commit两支，随后只捕获第一个实际揭示及原当前-only接拒/base完整C0，不跑K10或guard。未补支订单53/clock4.731592178344727/event28，base hard=False/customer_tw_violation；补支相同订单/clock、event31、source replan24至25，base hard=True/0失败。原AST见证_run_loop先推进物理再做原_plan_and_commit；限定这一起点控制边界遗漏，有具体成对证据，不推断所有状态/算法全局正确。

v2新独占目录在首请求计时内补一次原event27计划提交后继续两个固定实际揭示，科学输入、原策略、K10/种子/完整物理规则/预算不改，不按效用换样。新注册绑定原失败与成对诊断源SHA。补旧边界不是新接拒或重做callback17/物理event27，且独立参考也按原循环补该阶段。

独立审计v2工具首次在路径bootstrap之前导入hard_admission失败，实际退出1/源码及startup receipt保留，未开始独立复算。另封存v3仅调换导入顺序至原bootstrap后，使用同一实际生产/原字节输入，新AUDIT_BOOTSTRAP_RECOVERY_REGISTRATION绑定源和生产身份，无生产重跑。v3直接原接拒、所有40共同分支/NumPy均值、完整C0、原事件/车队/计划提交参考，独立原公共剩余后缀（不把已服务私有前缀当公共剩余计划）、固定头/原载荷SHA/epoch/base/状态/时间/全部加载源与实际进程退出核验；不以新engine/projector/checkpoint工作器作为参考。物理与canonical摘要库仍共享原实现，不是第二物理模型。

## 资格与下一步

仅本两个内生请求与已提交后丢回执/原字节恢复/重复票据组件合格。原checkpoint_ack_root_v1 10890/4FAIL与原13800载荷缺失保持，byte-retention7/2630和其它原件不改。full_live_request/full_loaded_dependency_closure/广义physical_ack_loss均False；没有部署正式全五臂或新科学日/训练。

下一步新事前工程登记中继续恢复后的实际新揭示请求，验证未发布取消保留上个已提交状态、重复/迟到序号边界及回收余量，把完整正式学习/同初始化已训CE/独立强动态接拒/全日10秒和正式全链依赖接齐。全部工程屏障满足后才另登记有限新鲜科学日，禁止重跑本17/2314或旧已合格QA。配送第2族FAIL封闭，不训练/重采/重审计/DEV/确认，第3族未启动，余30错配动作继续；負结果31必要门/183/C1原件保持。16入口本地备份更新，本人进程退出，无服务器/GPU/共享MD上传，最终20260927/28未用。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/CONTINUOUS_CHECKPOINT_EVIDENCE_20261012.json';create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),metrics=metrics,scoped_two_endogenous_checkpoint_component_qualified=True,preserved_producer_v1_exit1_sha256=sha(F/'RUN_EXIT.json'),preserved_audit_v2_startup_failure=True,producer_registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_CONTINUOUS_CHECKPOINT.json'),physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,new_scientific_data=False,stable_online_gain=False))
    backup=R/'current_entries_before_continuous_checkpoint_20261012';backup.mkdir(exist_ok=False);items=read(R/'current_entries_before_checkpoint_bytes_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货内生连续检查点v2：新callback18/19订单53接受/26拒绝，同resident真实7推进，17运行/2314独立项0失败；正常3.771585秒，第二提交后丢ACK8.018785秒含回收，另恢复窗.630232秒重复票据0接拒/推进。原v1遗漏event27计划提交导致base时间窗失败原件保留，成对诊断后v2首请求计时内补原阶段；独立v2导入前失败保持，v3只修bootstrap未重跑生产。实际原payload已存。两请求组件不等全日/正式CE强动态/完整预算，原checkpoint10890/4FAIL不改，核心在线增益仍未建立。\n'
    for x in items:
        f=E/x['path'];assert sha(f)==x['after_sha256'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(continuous_checkpoint_directory=str(O.relative_to(ROOT)),continuous_checkpoint_runtime_checks=17,continuous_checkpoint_independent_checks=2314,continuous_checkpoint_independent_failed=0,scoped_two_endogenous_checkpoint_component_qualified=True,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,next='Do not repeat 17/2314 continuous checkpoint or old completed QA. Next new prospective engineering must resume selected committed checkpoint into actual subsequent reveal; cancel before new publication preserves prior committed state, duplicate/late sequence and cleanup budgets. Formal learned/same-init trained CE/independent strong dynamic admission, whole-day complete10s and formal loaded dependency closure remain absent; no fresh science before all barriers. Preserve v1 event27 missing-plan failure, audit v2 bootstrap exit1, prior checkpoint10890/4 and absent old payloads.')
            obj['reproduction']['continuous_checkpoint_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [内生连续事务检查点与丢回执恢复](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True,full_live_request_qualified=False));print(json.dumps(dict(updated_entries=len(updates),metrics=native(metrics),owned_processes=owned)))
if __name__=='__main__':main()
