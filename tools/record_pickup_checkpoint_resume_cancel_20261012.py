"""Archive limited new checkpoint cancellation scope and preserve prior bytes."""
import json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import sha
ROOT=old.ROOT;E=old.E;R=ROOT/'research/review_3080_20261010';O=old.P/'checkpoint_resume_cancel_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    text=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def main():
    result=read(O/'RESULT.json');audit=read(O/'INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json');assert (result['checks_count'],result['failed'],audit['checks_count'],audit['failed'])==(16,0,2308,0)
    for dr in (O,old.P/'continuous_checkpoint_root_v2',old.P/'continuous_checkpoint_root_v1'):
        assert all(sha(p)==h for p,h in read(dr/'REGISTRATION.json')['sources'].items())
    assert all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    names={'qualify_pickup_checkpoint_resume_cancel_v1.py','audit_pickup_checkpoint_resume_cancel_v1.py'}
    owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(a).name in names for a in p.info['cmdline'] or [])];assert not owned
    assert read(old.P/'checkpoint_ack_root_v1/INDEPENDENT_CHECKPOINT_ACK.json')['failed']==4
    frozen={}
    for n in (*sorted(names),'build_pickup_checkpoint_resume_cancel_v1.py'):
        target=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,target);frozen[str(target)]=sha(target)
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(production_actual_exit=0,independent_actual_exit=0,observed_from='exec_command/write_stdin actual completion',owned_processes_active=owned))
    _,decode=old.codecs();actual=decode(read(O/'OWNER_STATE_4.json'));metrics=dict(runtime_checks=16,runtime_failed=0,independent_checks=2308,independent_failed=0,walls_s=[h['wall_s'] for h in result['host']],separate_recovery_wall_s=result['recovery_host']['wall_s'],selected_callback=20,selected_order=11,cancelled_private_callback=21,cancelled_private_order=37,selected_events=3,discarded_private_events=1,writer_files=len(read(O/'SOURCE_START.json')['files']),writer_modules=len(read(O/'SOURCE_START.json')['module_files']),audit_files=len(read(O/'AUDIT_SOURCE_START.json')['files']),audit_modules=len(read(O/'AUDIT_SOURCE_START.json')['module_files']))
    report=E/'01_取货线/文档/结果与进度/取货_选定检查点后续揭示与未发布取消组件_2026-10-12.md'
    text=f'''# 选定检查点后续揭示与未发布取消组件（2026-10-12）

pickup/checkpoint_resume_cancel_root_v1在执行前登记，从continuous_checkpoint_root_v2实际共享seq2原始载荷恢复已提交callback19/event34状态，继续两个新的真实揭示。16运行项、另进程2308独立项0失败，实际退出0。没有重跑callback18/19或17/2314、7/2630、24/10890等旧QA；原失败与缺失字节保持。没有新科学日/模型/训练/服务器/GPU。

第一请求callback20/order11实际接受，原_run_loop连续3个物理推进至clock6.671595573425293/event37/replan31/17服务。worker计时内现场私有公共投影/编码解码、own_uncond_dynamic当前-only接拒、own_incumbent提议、原200历史共同K10/20完整分支、原完整C0/编译/JointPolicyOwner保护/私有安装与原_plan_and_commit。原writer25369字节、共享payload/envelope逐字留存，SHA850d7d74a5fc32ae114d8eaa756a0b9b357f9800dda8c3e6d1193776be56fd18；父在8秒截止前选择epoch3票据，{metrics['walls_s'][0]:.9f}秒含原字节归档，保留resident未回收。

同一resident收到两项固定控制边界：重复当前已提交epoch3返回原摘要，迟到epoch2明确拒绝（零摘要/拒绝状态）。两项均不进入原_run_loop/接拒/SAA/物理执行，记录数仍1、物理数仍3、私有摘要与调用端权威均不变。只是这两个实际序号和当前协议入口，不证明任意网络重排、重叠请求或对抗性输入的一次执行语义。

第二请求callback21/order37实际在同私有副本中接受并完成接拒/K10全20分支/C0/编译/原_plan_and_commit，clock6.703140735626221/event38/replan32/17服务，另一个物理事件也仅发生于该副本。writer26951字节原载荷与完整响应/输入/私有状态已留存，SHA73664bb1eb2ef9f62dc1f225bd3bf67346f816f419f8b9fa8b8b924674e4625e。随后实际进入共享发布前延迟12秒；8秒截止取消kill/reap，{metrics['walls_s'][1]:.9f}秒含回收与原共享归档，无epoch4检查点发布/回执/调用端选择。共享完整envelope逐字保持第一已提交epoch3，调用端权威仍是callback20，而非私有已接受order37的副本。不能称此故障在私有提交前，也不能把被丢弃副本的物理推进计入调用端权威轨迹或真实对外不可逆行为。

另一个独立10秒恢复窗内读取实际保留共享epoch3原字节，验证SHA/epoch/base、pickle解码并恢复私有env/车队/轨迹/served/replanner/clock/event/replan，{metrics['separate_recovery_wall_s']:.9f}秒含本人回收，0接拒/0评分/0物理再推进。它与原取消窗分开，不合计成同一个10秒请求。恢复的实际所有字段等于第一已提交状态，明确不同于第二被取消副本；尚未让恢复进程继续下一实际揭示。初始实际选定seq2环境恢复和恢复进程运行库预热仍在请求计时外；恢复窗内选定seq3解码恢复已计时。

生产/恢复实际516文件263module映射、独立审计522文件251映射起止一致，仅该无模型工作负载。原始小共享载荷归档仍同步，正常/故障实测低于10秒不证明任意大小、慢磁盘或父阶段可抢占。未验父崩溃/持久化/撕裂发布、任意故障阶段及真实外部不可逆操作。

独立核验直接原接拒、所有40共同分支与NumPy均值、原完整C0、原_run_loop与4个实际私有物理事件、公共剩余后缀、实际原writer/shared字节、私有状态/票据/时间/加载依赖与进程退出；不以新engine/projector/checkpoint worker作为参考。物理和canonical摘要仍共享原实现，不是独立第二物理模型。

只本正常后续揭示、两个序号边界及共享发布前取消/保留/恢复组件合格。正式学习/同初始化已训CE、独立强动态接拒、完整等耗时迭代、全日完整10秒控制与科学全依赖仍缺；full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False，稳定可归因MaskCO核心在线增益未建立。下一步另事前工程登记，将恢复后真实未决揭示接回连续控制，区分取消分支与调用端最终执行，完善其它取消阶段/回收和正式控制链，不能以旧分支效用挑样。配送第2族FAIL封闭、第3族未启动，旧负结果31必要门/183/C1不改不重整。16本地入口先备份后更新，最终seed20260927/28未用，共享MD未上传。
'''
    create(report,text)
    boundary=R/'reproduction/CHECKPOINT_RESUME_CANCEL_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),metrics=metrics,registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_CHECKPOINT_RESUME_CANCEL.json'),scoped_resume_prepublication_cancel_component_qualified=True,selected_actual_state_sha256=actual['records'][0]['resident_after'],cancelled_private_state_sha256=actual['records'][1]['resident_after'],full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False,old_failures_preserved=True))
    backup=R/'current_entries_before_checkpoint_resume_cancel_20261012';backup.mkdir(exist_ok=False);items=read(R/'current_entries_before_continuous_checkpoint_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货检查点后续取消组件：新callback20/order11接受并选择epoch3（3.360646秒、3权威推进）；callback21/order37只在私有副本接受/推进后、共享发布前延迟取消，8.019228秒含回收，调用端仍保留epoch3，第二副本推进不计权威执行。重复epoch3/迟到epoch2无接拒或推进；另独立恢复窗.665088秒恢复epoch3完整状态。16运行/2308独立项0失败，仅本组件，原FAIL保留；完整10秒/正式CE强动态/迭代证据与核心在线增益仍未建立。\n'
    for x in items:
        f=E/x['path'];assert sha(f)==x['after_sha256'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(checkpoint_resume_cancel_directory=str(O.relative_to(ROOT)),checkpoint_resume_cancel_runtime_checks=16,checkpoint_resume_cancel_independent_checks=2308,checkpoint_resume_cancel_independent_failed=0,scoped_resume_prepublication_cancel_component_qualified=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,next='Do not repeat16/2308 or old completed QA. New prospective engineering must continue restored selected epoch3 into actual pending reveal, distinguish discarded cancellation branch from caller-authoritative execution; complete other cancellation stages/cleanup, formal learned/same-init trained CE/independent strong dynamic admission and whole-day complete10s/scientific closure before fresh science. Preserve all old FAIL and absent original13800-byte payload.')
            obj['reproduction']['checkpoint_resume_cancel_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':new+='\n- [检查点后续揭示与未发布取消](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=len(updates),metrics=metrics,owned_processes=owned)))
if __name__=='__main__':main()
