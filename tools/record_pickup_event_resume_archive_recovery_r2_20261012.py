"""Append narrow archive checker correction; preserve all first-pass bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/event_loss_resume_root_v1'
a=read(O/'FINAL_ARCHIVE_VERIFICATION_RECOVERY_R2.json');assert a['checks_count']==8 and a['failed']==0
report=E/'01_取货线/文档/结果与进度/取货_丢通知恢复组件归档空错误日志核验更正_2026-10-12.md'
create(report,'''# 丢通知恢复组件归档空错误日志核验更正（2026-10-12）

event_loss_resume_root_v1原生产19/独立1807项0失败、监督/审计/worker实际自然退出0保持。最后只读归档检查1183项1FAIL原件和检查器源码保存，唯一失败是检查器新增“每个工件必须非空”的假定把真实0字节worker.stderr.txt判失败；其余1182项输入/实际加载身份/原始字节/16入口备份/封存源/自动化与进程检查通过。空stderr表示没有错误输出，不应要求制造日志内容来满足这一检查器假定。

执行前另登记窄R2，8项0失败，核实际空stderr与自然0退出/原19和1807结果/原1183唯一失败与其余1182成功项/原文件身份不变。资格由原1182成功项和本R2窄语义证据组合，不把原1183/1写成整体PASS。没有重跑生产、独立审计、原归档检查或已合格旧QA，没有改变科学屏障、容差或工件原字节。另一次只读控制台默认GBK读取该UTF8中文结果失败，改为显式UTF8后读取；没有修改结果或实验。

当前actual slot0 epoch12/callback29/order76原shared38929字节仍为权威起点，下一真正callback30身份未观察。full_live_request/full_loaded_dependency_closure/广义physical_ack_loss仍False；正式模型/同初始化已训CE/独立强动态接拒/全日10秒等耗时迭代和科学完整依赖仍待，稳定可归因MaskCO核心在线增益与真实强动态证据未建立。配送第2族FAIL封闭、第3族未启动；旧FAIL/原13800缺失载荷/31与183历史字节保持，无服务器/GPU/共享MD上传，最终20260927/28未用。
''')
boundary=R/'reproduction/EVENT_LOSS_RESUME_ARCHIVE_RECOVERY_R2_20261012.json'
create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='archive empty stderr assumption only',original_final_sha256=sha(O/'FINAL_ARCHIVE_VERIFICATION.json'),original_checks=1183,original_failed=1,retained_successful_items=1182,narrow_registration_sha256=sha(O/'FINAL_ARCHIVE_RECOVERY_R2_REGISTRATION.json'),narrow_result_sha256=sha(O/'FINAL_ARCHIVE_VERIFICATION_RECOVERY_R2.json'),narrow_checks=8,narrow_failed=0,producer_checks=19,independent_checks=1807,production_rerun=False,original_FAIL_preserved=True,stable_online_gain=False))
before=R/'current_entries_before_event_loss_resume_20261012';backup=R/'current_entries_before_event_loss_resume_archive_r2_20261012';backup.mkdir(exist_ok=False);entries=read(before/'UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
for x in entries:
    dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
note='\n> 2026-10-12丢通知恢复归档窄更正：原生产19/独立1807合格保持；归档1183/1FAIL唯一误将实际空stderr判失败，原件保留。另登记R2 8项0失败，资格由原1182成功身份项+窄语义证据组合，不改写原FAIL，不重跑实验/审计/归档QA。当前actual epoch12/callback29→callback30及科学缺口不变。\n'
updates=[]
for x in entries:
    f=E/x['path'];text=f.read_text(encoding='utf-8')
    if f.suffix=='.json':
        obj=json.loads(text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(event_loss_resume_archive_original_checks=1183,event_loss_resume_archive_original_failed=1,event_loss_resume_archive_r2_checks=8,event_loss_resume_archive_r2_failed=0,event_loss_resume_archive_qualification='original1182successful_identity_items+narrowR2_empty_stderr_semantics; original1183/1FAIL preserved; no rerun');obj['reproduction']['event_loss_resume_archive_recovery_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2)
    else:
        new=note+text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
        if f.parent==E/'01_取货线':new+='\n- [归档空错误日志窄更正](文档/结果与进度/'+report.name+')\n'
    f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
checks=[dict(name=x['path'],passed=sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256']) for x in updates]
create(O/'ARCHIVE_R2_UPDATE_IDENTITY.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=16,failed=sum(not x['passed'] for x in checks),report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_index_sha256=sha(backup/'UPDATE_IDENTITY.json'),old_report_and_archive_and_failure_preserved=True))
assert all(x['passed'] for x in checks);print(json.dumps(dict(entries=16,failed=0,original_final_failed_preserved=1,narrow_recovery_checks=8)))
