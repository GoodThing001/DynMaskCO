"""Archive only the new byte-retention component; preserve old FAIL."""
import json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import sha
ROOT=old.ROOT; E=old.E; R=ROOT/'research/review_3080_20261010'; OUT=old.P/'checkpoint_bytes_retention_root_v1'; PREV=old.P/'checkpoint_ack_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    s=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(s)
def main():
    r=read(OUT/'RESULT.json'); a=read(OUT/'INDEPENDENT_BYTES.json'); w=read(OUT/'WRITER.json')
    assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(7,0,2630,0)
    assert read(OUT/'RUN_EXIT.json')['exit_code']==read(OUT/'AUDIT_RUN_EXIT.json')['exit_code']==0
    for name in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):
        reg=read(OUT/name);assert all(sha(p)==h for p,h in reg.get('sources',reg.get('inputs')).items())
    assert read(PREV/'INDEPENDENT_CHECKPOINT_ACK.json')['failed']==4
    for name,key in (('REGISTRATION.json','sources'),('AUDIT_REGISTRATION.json','inputs')):
        assert all(sha(p)==h for p,h in read(PREV/name)[key].items())
    owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in ('qualify_pickup_checkpoint_bytes_v1.py','audit_pickup_checkpoint_bytes_v1.py') for x in p.info['cmdline'] or [])]; assert not owned
    closures={n:read(OUT/n) for n in ('SOURCE_START.json','SOURCE_END.json','AUDIT_SOURCE_START.json','AUDIT_SOURCE_END.json')}
    dll=str(Path('C:/Windows/System32/SHCore.dll'));assert all(dll in x['files'] for x in closures.values())
    frozen={}
    for n in ('qualify_pickup_checkpoint_bytes_v1.py','audit_pickup_checkpoint_bytes_v1.py'):
        dest=OUT/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    report=E/'01_取货线/文档/结果与进度/取货_检查点原始字节留存与窄独立核验_2026-10-12.md'
    create(report,f'''# 检查点原始字节留存与窄独立核验（2026-10-12）

pickup/checkpoint_bytes_retention_root_v1事前登记，仅使用checkpoint_ack_root_v1原PHYSICAL_ATTEMPT_0已封存event27完整私有值树生成一个新载荷。7运行项、另进程2630独立项均0失败，实际退出均0。没有物理推进、接拒、求解器、SAA、模型、科学数据或旧24/10890 QA重复。它关闭新正常载荷字节保存组件的缺口，不能升级原检查点故障组件整体资格。

实际worker用pickle protocol5新序列化，先独占保存ACTUAL_WRITER_PAYLOAD.bin原字节，再写共享载荷/固定epoch、size、base、payload SHA及state摘要头。调用端等有界worker退出后，保存实际ACTUAL_SHARED_ENVELOPE.bin和ACTUAL_SHARED_PAYLOAD.bin，比较共享与writer字节、SHA、长度及头。实际{w['size']}字节，SHA={w['sha256']}，完整私有状态摘要={w['state_digest']}。从发出GO到三个字节工件保存及本人回收{r['wall_s']:.9f}秒，总10/执行8/预留2；仅此小正常负载，初始化/值树解码/运行库预热在外，不是现场全请求或发布故障计时验收。父有界等待退出后同步复制/写小载荷，未证明任意大小父阶段可抢占、磁盘硬实时或持久化事务。

另进程直接读取实际二进制文件，独立解析固定头、计算原字节SHA、核writer/shared/envelope逐字相同、直接pickle反解后逐字段核数组dtype/shape/原字节、数据类/序列/映射/浮点hex及独立canonical摘要。参考是旧封存状态值树，不重新pickle参考树要求旧字节相同。两个新的内存边界：实际载荷末字节翻转与epoch改变均被独立固定头/SHA验证拒绝；这是独立验证器边界，不是生产运行时故障、撕裂发布或对抗协议资格。没有恢复仿真env、物理事件、重复票据控制、内生接拒或连续事件执行。

writer实际加载起止{len(closures['SOURCE_START.json']['files'])}文件/{len(closures['SOURCE_START.json']['module_files'])}模块映射相同；审计实际起止{len(closures['AUDIT_SOURCE_START.json']['files'])}文件/{len(closures['AUDIT_SOURCE_START.json']['module_files'])}映射相同。两个进程在封存前显式加载系统SHCore.dll并预初始化process_iter/closure，实际DLL字节在四份起止清单中一致。此方法封存本组件实际负载，不唯一解释原审计何处首次加载SHCore.dll，亦不等完整正式科学链依赖资格。

本新载荷来自旧JSON显式值树的重新序列化，{w['size']}字节与原13800字节不同，明确不是原v1载荷找回或补造。原checkpoint_ack_root_v1三payload_identity FAIL及audit_loaded_closure FAIL、10890/4和24/0原件/source/input SHA均保持。原载荷仍缺，原physical_ack_loss/checkpoint独立资格仍False；本新字节组件合格不能倒推旧故障全过程通过。没有新故障/原物理重跑/原审计重跑或放宽旧门。

下一步在新的事前工程登记中把实际原字节保留接入内生当前接拒/提议/共同K10/完整C0认证编译、连续现场私有所有者及逻辑提交/丢回执/取消/重复一致性。须核整请求回收余量和完整正式依赖，正式学习模型、同初始化已训CE、独立强动态接拒与全日10秒仍待；全部工程屏障满足后才另登记有限新鲜科学日。配送第2族FAIL封闭，不训练/重采/重审计/DEV/确认，第3族未启动，余30错配动作工程继续。负结果31必要门/183/C1原件保持。本人worker已退出，无服务器/GPU/共享MD上传，最终20260927/28未用。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/CHECKPOINT_BYTES_RETENTION_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),runtime_checks=7,runtime_failed=0,independent_checks=2630,independent_failed=0,new_payload_bytes_qualified=True,new_payload_bytes=w['size'],new_payload_sha256=w['sha256'],wall_s=r['wall_s'],original_v1_payloads_recovered=False,original_independent_FAIL_sha256=sha(PREV/'INDEPENDENT_CHECKPOINT_ACK.json'),physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,new_scientific_data=False,registration_sha256=sha(OUT/'REGISTRATION.json'),independent_sha256=sha(OUT/'INDEPENDENT_BYTES.json')))
    backup=R/'current_entries_before_checkpoint_bytes_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_checkpoint_ack_20261012/UPDATE_IDENTITY.json')['files'];updates=[]
    note='\n> 2026-10-12取货检查点新正常字节留存窄组件：7运行/2630独立项0失败，新实际15459字节writer/shared/envelope均留存且逐字一致，.533397秒含回收。审计和writer实际SHCore.dll及加载文件起止封存。本次无物理/接拒/SAA重跑；不是原13800字节找回，原checkpoint 10890/4FAIL保持。内生连续控制/故障接线、正式CE/强动态及完整10秒仍待，在线核心增益未建立。\n'
    for item in entries:
        f=E/item['path'];assert sha(f)==item['after_sha256'];dest=backup/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['pickup'].update(checkpoint_bytes_directory=str(OUT.relative_to(ROOT)),checkpoint_bytes_runtime_checks=7,checkpoint_bytes_independent_checks=2630,checkpoint_bytes_independent_failed=0,new_checkpoint_bytes_qualified=True,checkpoint_independent_qualified=False,physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,next='Preserve original checkpoint_ack_root_v1 24/10890 4 FAIL; actual old payloads remain absent. New checkpoint_bytes_retention_root_v1 7 runtime/2630 independent 0 FAIL only qualifies newly serialized normal raw byte archiving and fixed header audit, no physical replay. Next prospectively integrate raw retention and checkpoint logical commit/recovery into endogenous admission/proposal/common K10/full C0/compile and continuous private owner; fault/duplicate/cancel/reap and complete request/loaded dependency barriers remain. Formal learned/same-init trained CE/independent strong dynamic admissions/full-day10s still missing; no fresh science before all barriers.')
            obj['reproduction']['checkpoint_bytes_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');updated=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [检查点原始字节留存窄核验](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');updates.append(dict(path=item['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(OUT/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,original_independent_FAIL_preserved=True,full_live_request_qualified=False))
    print(json.dumps(dict(updated_entries=len(updates),runtime_checks=7,independent_checks=2630,failed=0,owned_processes=owned,old_failed=4)))
if __name__=='__main__':main()
