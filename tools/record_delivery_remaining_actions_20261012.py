"""Archive new observed action diagnosis and isolated guard; old artifacts immutable."""
import hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';P=D/'family2_remaining_action_diagnosis_root_v1';O=D/'retained_edge_action_guard_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    r=read(P/'RESULT.json');a=read(P/'INDEPENDENT_REMAINING_ACTIONS.json');g=read(O/'RESULT.json');ga=read(O/'INDEPENDENT_EDGE_OCCURRENCES.json')
    assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'],g['checks_count'],g['failed'],ga['checks_count'],ga['failed'])==(875,0,1480,0,217,0,223,0)
    for base in (P,O):
        for name in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):
            assert all(sha(p)==h for p,h in read(base/name)['inputs'].items())
        assert read(base/'RUN_EXIT.json')['exit_code']==read(base/'AUDIT_RUN_EXIT.json')['exit_code']==0
    names=['diagnose_delivery_remaining_actions_v1.py','audit_delivery_remaining_actions_v1.py','delivery_retained_edge_action_guard_v1.py','qualify_delivery_retained_edge_guard_v1.py','audit_delivery_retained_edge_guard_v1.py']
    owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not owned
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(diagnostic_actual_exit=0,coordinate_audit_actual_exit=0,guard_actual_exit=0,guard_audit_actual_exit=0,owned_processes_active=owned,observed_from='actual exec_command/write_stdin completions; no remote dispatch'))
    report=E/'02_配送线/文档/结果与进度/配送_剩余30错配插入动作与单订单保留边检查_2026-10-12.md'
    create(report,'''# 配送剩余30错配：插入动作与单订单保留边检查（2026-10-12）

在新独占工程目录事前登记后，完整覆盖第2族既有审计中30项不同无向邻接的固定探针错配。仅流式读取旧STATES及已保存插入记录，不重新调用重构器、教师/求解器、模型、route_ok或物理评价器；不是第2族重复探针/审计，也没有新科学族、训练、开发或确认。原第2族注册SHA562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176、9819/35FAIL、96248/36注册屏障失败和科学放行False保持。

875项记录动作/评分/源身份检查0失败；另进程1480项独立检查0失败。独立采用60位Decimal.from_float坐标算术、逐车目标秩递增和归属约束，未导入新诊断器/解码器/物理或模型。原实际服务器delivery_equal_schedule.py及delivery_baselines.py SHA与本地冻结源逐字一致，评分式为标准化距离增量减标准化目标无向边增量。新工程算术绝对1e−10界限事前登记，仅核局部评分算术，不修改任何原科学精确门；实际与原保存分数最大差1.7763568394002505e−15，不声称浮点逐位重现。独立实际63加载文件起止一致，仅本标准库工作负载，不等正式科学全依赖闭包。

以当前插入骨架为教师逐车路线的子序列定义结构一致性，30项首个偏离都在零基插入15至24步，且均把当前订单插入教师目标以外车槽。比较同一个订单在唯一教师兼容位置与原实际选择位置的局部分数：16项目标动作分数更低，14项原选择分数更低，没有1e−10界限内平分。仅同订单位置比较，没有重新算其它订单的regret竞争，不能由此认定所有错配因果。

其中26项目标动作被原`if adjacency[a,b] > 0: continue`无条件排除。26个实际骨架的目标车槽均只剩一个订单b，原仓库无向边{0,b}在路径0→b→0中有两次出现。向b的一侧插入新订单只移除一项边出现，另一侧仓库边仍在；所有原掩码保留边、未掩码原车和严格顺序均保持。原边集合已折叠这两个出现，直接禁插比最终保留边联合不变量更严。这是一项具体结构过度限制，与四组路线反向别名不同；不是仅从行归一差异推测错误。

| 首个目标动作情况 | 数量 | 分数比较 |
|---|---:|---|
| 单订单仓库边直接禁插，目标分数更低 | 16 | 更低分仍被结构检查排除 |
| 单订单仓库边直接禁插，原选择分数更低 | 10 | 既有结构排除，也有评分偏离 |
| 目标动作未被该边检查排除 | 4 | 原选择分数均更低，不能归因于该禁插 |

新独占delivery_retained_edge_action_guard_v1结构组件先构造一次插入后的完整车槽路线，再检查全部原要求无向边仍在联合边集合中。217组件检查0失败：30个固定目标动作全部保留原边/原车顺序，26个单订单双出现均2→1；内部保留边、非单订单唯一仓库边真正被破坏时仍拒绝，另重复订单/域外槽/位置边界拒绝。组件同时以独立边出现计数更新核对，不调用物理评价。另进程223项独立核验0失败，直接从原保留矩阵/有重复的有序边出现列表/目标秩推导全部30与26双出现；未导入新guard或重构器。此工程源与输入起止哈希一致，四进程实际退出0，实际本人进程已结束。

该局部组件尚未接入生产greedy重构器。知道目标动作后证实它保留结构，不等模型能预测、该动作通过生产物理过滤、后续重构必然精确、最终成本提升或全部34错配关闭。原当前状态记录没有完整每轮可行候选集合，无法据本局部分数重算所有regret选择。即使修正禁插，14项首偏离时原选择分数更低，其中4项不被此禁插影响；无向目标在临时插入锚点上的增量也可能为0或−1。后续应另事前工程登记实际生产动作适配与有方向/车槽/锚点评分，完整保留失败分支与物理门、各控制和预算；不能直接升级第2族、降门或按成本挑重构结果。科学第3族尚未启动。

取货选定恢复/真实事件/其它计算与发布取消、正式学习/同初始化已训CE、独立强动态接拒、全日10秒与科学完整依赖仍待；不重跑13/1684或其它完成QA。负结果只新增本边界，31必要门/183历史与C1失败原件不重整。稳定、可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。

工件为research/review_3080_20261010/delivery/family2_remaining_action_diagnosis_root_v1和retained_edge_action_guard_root_v1；原STATES未整文件载入内存，30份选定既有记录独占保存。16本地入口先备份后更新，无服务器/GPU/共享MD上传，最终取货seed20260927/28未用。
''')
    boundary=R/'reproduction/REMAINING_ACTION_EDGE_GUARD_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),diagnosis_directory=str(P.relative_to(ROOT)),guard_directory=str(O.relative_to(ROOT)),diagnostic_checks=875,independent_coordinate_checks=1480,guard_checks=217,independent_occurrence_checks=223,failed=0,original_singleton_excess_skip=26,unblocked_score_preference_cases=4,first_divergences_all_wrong_vehicle=True,isolated_structural_guard_qualified=True,production_decoder_qualified=False,physical_or_gain_qualified=False,family2_FAIL_preserved=True,scientific_family3_started=False,stable_online_gain=False,identities={str(p):sha(p) for p in (P/'REGISTRATION.json',P/'RESULT.json',P/'INDEPENDENT_REMAINING_ACTIONS.json',O/'REGISTRATION.json',O/'RESULT.json',O/'INDEPENDENT_EDGE_OCCURRENCES.json')}))
    backup=R/'current_entries_before_remaining_actions_20261012';backup.mkdir(exist_ok=False);updates=[]
    note='\n> 2026-10-12配送余30动作错配诊断：首次偏离均后段换出教师车槽；26项目标插入被单订单仓库无向边禁插过度排除（双出现2→1仍全保留），其余4项原评分更优。875记录/1480独立算术/217隔离guard/223独立边出现项0失败。仅旧记录和结构组件，未改生产/重跑探针/物理认证/训练/新第3族；第2族FAIL保持，在线增益仍未建立。\n'
    entries=read(R/'current_entries_before_restore_compute_cancel_20261012/UPDATE_IDENTITY.json')['files']
    # Complete identity validation before changing any shared current entry.
    assert all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['delivery'].update(remaining_action_diagnosis_directory=str(P.relative_to(ROOT)),remaining_action_record_checks=875,remaining_action_independent_checks=1480,remaining_action_independent_failed=0,remaining_first_divergences_wrong_vehicle=30,observed_singleton_excess_edge_skips=26,unblocked_score_preference_cases=4,retained_edge_guard_directory=str(O.relative_to(ROOT)),retained_edge_guard_checks=217,retained_edge_guard_independent_checks=223,isolated_retained_edge_guard_qualified=True,modified_production_decoder_qualified=False,current_family2_FAIL_preserved=True,next='Do not repeat875/1480/217/223 or family2 closed QA. Prospective isolated engineering should integrate complete retained-edge invariant with production feasible insertion and directed-slot/anchor scoring; four unblocked score deviations and full outcome qualification remain. No family3/new science until engineering/preregistration; continue pickup timed restoration and remaining live cancellation/formal model/control/full-day dependency gaps.')
            obj['reproduction']['remaining_action_guard_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'02_配送线':new+='\n- [剩余30错配与单订单保留边检查](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert len(updates)==16 and all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=len(updates),record_checks=875,coordinate_checks=1480,guard_checks=217,occurrence_checks=223,failed=0,owned_processes=owned)))
if __name__=='__main__':main()
