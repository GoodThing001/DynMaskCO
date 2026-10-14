"""Seal the scoped continuous last50+terminal engineering evidence and entries."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
import qualify_pickup_joint_policy_transaction_root_v3 as old
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/resident_day_tail_root_v1';L=R/'pickup/resident_day_tail_launch_root_v1'
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_RESIDENT_DAY_TAIL.json');assert r['failed']==a['failed']==0 and read(O/'RUN_EXIT.json')['exit_code']==read(L/'EXIT.json')['exit_code']==0 and r['actual_worker_exit']==0
    assert all(sha(p)==h for p,h in read(O/'REGISTRATION.json')['sources'].items()) and all(sha(p)==h for p,h in read(O/'AUDIT_REGISTRATION.json')['inputs'].items())
    _,decode=old.codecs();state=decode(read(O/'OWNER_STATE_64.json'));term=decode(read(O/'TERMINAL.json'));restore=decode(read(O/'RESTORE_IN_REQUEST.json'));final=state['final'];ev=term['evaluation'];records=state['records'];last=records[-1];b=read(O/'SOURCE_START.json');ab=read(O/'AUDIT_SOURCE_START.json')
    assert len(records)==50 and [x['callback'] for x in records]==list(range(31,81)) and r['terminal_epoch']==64 and records[0]['order']==39
    names=('build_pickup_resident_day_tail_v1.py','qualify_pickup_resident_day_tail_v1.py','audit_pickup_resident_day_tail_v1.py','run_pickup_resident_day_tail_bounded_v1.py');live=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in names for t in p.info['cmdline'] or [])];assert not live;frozen={}
    for n in names:
        p=O/('FROZEN_'+n);assert not p.exists();shutil.copy2(ROOT/'tools'/n,p);frozen[str(p)]=sha(p)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True));create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,supervisor_actual_exit=0,independent_actual_exit=0,worker_actual_exit=0,worker_natural_exit=True,owned_processes_active=live))
    facts=dict(runtime_checks=r['checks_count'],independent_checks=a['checks_count'],failed=0,new_requests=50,terminal_operations=1,physics=len(state['events']),first_order=records[0]['order'],last_order=last['order'],accepted=len(final['state']['accepted']),rejected=len(final['state']['rejected']),served=int(final['served'].sum()),clock=float(final['clock']),event=final['event_id'],replan=final['replan_count'],vehicle_statuses=[v.status for v in final['fleet']],return_arrivals=[t.return_arrival for t in final['traces']],utility=float(ev['utility']),normal_min=min(h['wall_s'] for h in r['host'][:-1]),normal_max=max(h['wall_s'] for h in r['host'][:-1]),terminal_wall=r['host'][-1]['wall_s'],restore_s=restore['finished']-restore['started'],production_files=len(b['files']),production_modules=len(b['module_files']),audit_files=len(ab['files']),audit_modules=len(ab['module_files']),final_payload_bytes=(O/'RETAINED_PAYLOAD.bin').stat().st_size,final_payload_sha256=sha(O/'RETAINED_PAYLOAD.bin'),supervisor_seconds=read(L/'EXIT.json')['wall_s'])
    create(O/'SEALED_STAGE_FACTS.json',facts)
    report=E/'01_取货线/文档/结果与进度/取货_有效检查点接续五十揭示至返仓终局的连续控制工程_2026-10-12.md'
    create(report,f'''# 有效检查点接续五十揭示至返仓终局的连续控制工程（2026-10-12）

pickup/resident_day_tail_root_v1执行前登记，唯一来源为lineage_header_cancel_root_v1父真正保留epoch13/callback30的42569字节原载荷，SHA5d9aa477e539d0f350e09e96970f288d8e50dac16f7a3ecba5674613ae03b7b7。上一错误共享selector/header epoch14与取消order39私有分支没有作为恢复来源。新缓冲区初始slot0装载原epoch13，不改变前阶段调用端选定slot1的事实；父shared预置、原库/200历史/80数据和仅旧epoch12环境预热在外。首实际请求窗口内原SHA/epoch/base/pickle/全部私有字段恢复{facts['restore_s']}秒，恢复段0接拒/评分/物理。

同一resident/env/车队/轨迹/served/策略从真正callback31/order39连续至callback80/order{facts['last_order']}，50次新揭示全部保留，没有按效用选状态、恢复旧取消分支、换样/换种子或重复前30。每次实际私有公共提取编码解码、own_uncond_dynamic当前-only接拒、own_incumbent同计划提议、原200历史共同K10/20完整分支、原完整C0认证编译/guard安装、actual plan_and_commit、原pickle检查点/共享selector发布、父有界SHA/base/稳定selector验证与8秒截止前唯一选择、事件ACK和原字节归档均计时内。第二及后续均以紧前实际提交结果开始，50次全部未换路线；没有正式学习模型或新科学数据。

50个请求各自10秒窗口（执行8/预留2），实际正常{facts['normal_min']}至{facts['normal_max']}秒，resident持续保留，不声称每次均测回收。最后另一个10秒终局窗口，原_run_loop从第80次实际plan之后执行剩余物理推进/返仓，没有新接拒或K10；原完整coldchain_evaluator_a1直接完整80数据与实际全轨迹。终局检查点epoch64/新slot1发布选择，{facts['terminal_wall']}秒含自然退出0/本人回收。51窗口分别计时，不能合称一个10秒日；整轮监督600秒未触发，实际{facts['supervisor_seconds']}秒。本轮无新故障注入或重复票据/丢通知QA。

本次实际新增{facts['physics']}个权威原物理推进，末端clock{facts['clock']}/event{facts['event']}/replan{facts['replan']}。全状态累计{facts['accepted']}接受且全部实际服务/{facts['rejected']}拒绝，末端所有80订单已决定、pending为空，车队状态{facts['vehicle_statuses']}、返仓时刻{facts['return_arrivals']}。原完整评价hard/0失败，效用{facts['utility']}仅本已观察日own_uncond_dynamic/own_incumbent工程描述，不与旧cond_hist947.40554当策略收益或复现；也不根据其是否正向决定保留。来源前30已封存，当前工程独立执行后50和尾部，不能称日初单进程全部80的新独立资格。

每次writer/actual shared/envelope原字节全部保留；终局载荷{facts['final_payload_bytes']}字节SHA{facts['final_payload_sha256']}，RETAINED与终局writer/shared/envelope逐字一致。{facts['runtime_checks']}运行/{facts['independent_checks']}另进程独立项0失败，监督/生产/审计和resident实际自然退出0；独立从同一原epoch13恢复，直接原接拒/全部1000共同K10分支float.hex和NumPymean/原完整C0/编译/每次原物理事件/公共剩余后缀/全部私有字段/50实际检查点SHA谱系/终局完整原评价与字节/父选择时钟/自然回收/所有输入源身份复算，不以新engine/projector/checkpoint worker作参考，物理/canonical仍共享原实现。没有原31私有取消/前30/旧472/4711/20/1818等QA重跑。

生产起止{facts['production_files']}实际文件/{facts['production_modules']}module map一致，独立{facts['audit_files']}/{facts['audit_modules']}起止一致，仅本地Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型连续剩余日负载。代码生成预执行首次边界定位导致IndentationError，未生成运行程序或启动实验；原生成器字节另保存，修正定位后才登记执行，不是实验失败后重跑。所有实际实验原件与旧FAIL/缺13800字节保持。

该工程关闭同resident正常剩余日接续至返仓终局及原完整评价/51实际窗口的窄门；没有学习模型、已训同初始化CE、独立强动态接拒、等耗时MaskCO迭代或科学完整依赖，不能将无模型加载闭包升级全科学链。父有界复制/SHA与同步归档计时不证明任意大小/慢磁盘/父阶段抢占、native任意指令点、selector内存屏障、父崩溃持久化或外部不可逆操作；没有新故障覆盖。full_live_request/full_loaded_dependency_closure/广义physical_ack_loss均仍False。

此已观察日当前控制后续已到终局，无未决第31或后续订单；不要继续从旧epoch13重复余日，不能启动新鲜科学日。下一优先接正式模型提议/同初始化已训CE/同任务训练随机控制、独立强动态接拒与等耗时迭代，封存全科学实际依赖和整请求预算；屏障全满足后才另登记有限科学日。配送2254/15160和理想163/1886864沿用不重跑，实际评分头/各控制/C0/预算待，第2族FAIL封闭、第3族未启动。负结果31/183/C1原件不重整。16入口先保存旧字节本地更新，无服务器/GPU/共享MD上传，最终20260927/28未用。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/RESIDENT_DAY_TAIL_EVIDENCE_20261012.json';create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_RESIDENT_DAY_TAIL.json'),facts=facts,scoped_normal_remaining50_terminal_controller_qualified=True,whole80_from_initial_single_resident_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,new_scientific_family=False,stable_online_gain=False))
    backup=R/'current_entries_before_resident_day_tail_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_lineage_header_cancel_20261012/UPDATE_IDENTITY_R2.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note=f"\n> 2026-10-12取货连续剩余日终局：另事前从父actual retained13恢复，同一resident真正callback31/order39至80全部50揭示及另终局返仓窗口，{facts['runtime_checks']}运行/{facts['independent_checks']}独立项0失败。{facts['physics']}新增权威物理推进，累计{facts['accepted']}接受全服务/{facts['rejected']}拒绝、pending空、车辆返仓closed、原完整80评价hard。50正常{facts['normal_min']:.6f}..{facts['normal_max']:.6f}秒/终局{facts['terminal_wall']:.6f}秒含自然0退出回收，各10秒窗分别计时。实际终局epoch64/raw{facts['final_payload_bytes']}保存；仅封存前30+新连续后50，不等日初单进程全80/正式模型/完整科学全请求。此旧观察日后续已终局，不再从13重做；下一正式学习/已训CE/强动态接拒/等耗时迭代/全链依赖预算，科学日仍关闭。配送第2族FAIL封闭、第3族未启动，稳定在线增益未建立。\n"
    updates=[]
    for x in entries:
        f=E/x['path'];text=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(resident_day_tail_directory=str(O.relative_to(ROOT)),resident_day_tail_runtime_checks=facts['runtime_checks'],resident_day_tail_independent_checks=facts['independent_checks'],resident_day_tail_failed=0,scoped_normal_remaining50_terminal_controller_qualified=True,current_authoritative_epoch=64,current_authoritative_callback=80,current_authoritative_order=facts['last_order'],current_authoritative_slot=1,current_authoritative_payload_sha256=facts['final_payload_sha256'],current_authoritative_payload_bytes=facts['final_payload_bytes'],current_authoritative_source=str((O/'RETAINED_PAYLOAD.bin').relative_to(ROOT)),caller_selected_epoch=64,caller_selected_callback=80,caller_selected_order=facts['last_order'],caller_selected_slot=1,discarded_pending=None,discarded_pending_callback=None,discarded_pending_order=None,next_resume_callback=None,current_observed_controller_at_terminal=True,next='Do not repeat resident_day_tail50+terminal or previous completed QA. Actual caller retained epoch64/callback80 terminal raw'+str(facts['final_payload_bytes'])+' SHA'+facts['final_payload_sha256']+'; all80 decisions and accepted service/returns complete, pending empty. Actual callback31/order39 executed authoritatively after valid13 restoration; no remaining resume callback in this old observed-day controller. Scope is sealed first30 plus new same-resident last50/terminal, not day-start whole80/scientific full-run. Next prospectively integrate formal learned proposals/same-init trainedCE/task-trainedrandom/independentstrongdynamic admission/equal-timeMaskCOiteration, actual full scientific closure and whole-budget barriers before separately preregistering any fresh science. No extra no-model old-day QA or finalseeds.')
            obj['delivery']['next']='Do not repeat2254/15160 tensors or ideal163/1886864/closedfamily2. Actual model score head/per-action provenance/control outputs, same-init trainedCE/task-trainedrandom, fullC0 and interruptible full request remain for new prospective engineering before finitefamily3/newdata. Pickup crossed from valid13 through actual31..80/terminal64; old observed controller now complete/pending empty, do NOT replay13 or assume next callback. Prior bad14 preserved.'
            obj['reproduction']['resident_day_tail_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2)
        else:
            new=note+text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if f.parent==E/'01_取货线':new+='\n- [有效检查点接续五十揭示至返仓终局](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=live));print(json.dumps(dict(entries=16,facts=facts,active=live),ensure_ascii=True))
if __name__=='__main__':main()
