"""Archive the new local feasible-option primitive with explicit evidence limits."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'feasible_action_options_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_FEASIBLE_ACTION_OPTIONS.json');assert(r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(163,0,11316,0)
    for n in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):assert all(sha(p)==h for p,h in read(O/n)['inputs'].items())
    assert read(O/'RUN_EXIT.json')['exit_code']==read(O/'AUDIT_RUN_EXIT.json')['exit_code']==0
    names=('delivery_feasible_action_options_v1.py','qualify_delivery_feasible_action_options_v1.py','audit_delivery_feasible_action_options_v1.py')
    owned=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not owned
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,independent_actual_exit=0,owned_processes_active=owned,observed_from='actual local exec_command completion; no remote or GPU deployment'))
    report=E/'02_配送线/文档/结果与进度/配送_保留边修正接入实际可行插入候选枚举_2026-10-12.md'
    m=r['metrics'];mi=dict(capacity=min(x['capacity_margin'] for x in a['details']),tw=min(x['min_tw_margin'] for x in a['details']),return_=min(x['return_margin'] for x in a['details']))
    create(report,f'''# 保留边修正接入实际可行插入候选枚举（2026-10-12）

delivery/feasible_action_options_root_v1在实际枚举前登记，以此前固定30个已观察首偏离骨架/当前订单为输入，每个骨架只做一次当前订单的全部车槽/位置枚举。新独占原语将完整保留边联合检查与实际原delivery_baselines.matrices/route_ok及原评分分组接线；所有1525个位置、排除原因、分数、同订单最低分和目标动作都保存。没有完整regret客户循环/重构探针、教师求解、模型、全C0评价或最终成本计算，不是已封闭第2族重复实验，也不改原生产解码器或任何旧文件。

163运行检查0失败，另进程11316项独立核验0失败，两个实际退出0。30目标动作全部通过实际原容量、时间窗和返仓route_ok，以及完整保留边联合；原实际选择动作仍全部合格。这关闭了此前“26个被边检查排除的动作是否可能被原可行过滤排除”的局部不确定性：在这些固定骨架上答案为否。目标车槽插入后容量余量至少{mi['capacity']:.6f}，客户时间窗余量至少{mi['tw']:.9f}小时，返仓余量至少{mi['return_']:.9f}小时。余量是本局部路线过滤的量，不等完整温度/品质/占舱物理余量。

修正结构排除后，在同一已封存骨架和同一个订单上仅12/30的最低分选项等于教师兼容车槽/位置，另外18仍选择其它位置。此计数不等12条完整路线精确恢复或最终收益；未来客户regret竞争和后续骨架没有运行。原16项目标比实际选择分数低，不保证它优于新开放的所有其它候选。因此不能认为改一条边判断就关闭所有目标错位；需继续有方向车槽和临时锚点评分的实际目标适配。四组反向别名、第1族距离劣势与学习分布问题亦未因此关闭。

本地实际Python3.12.7/NumPy1.26.4/CPU7，BLAS单线程，未套用3080 NumPy2.2.6环境资格。原语调用实际本地原矩阵及route_ok，源字节保持且此前已与原服务器封存身份匹配；原global binary guide1评分及分组未改，logits仅旧教师已知二值邻接、不是模型预测。独立不导入新候选原语/guard/原route_ok或物理评价器，直接坐标距离、逐订单容量/ready/reveal/服务开始TW/返仓时钟、全部原保留矩阵覆盖与原评分公式复算所有1525位置。原route_ok的1e−9边界沿用；新工程评分算术绝对1e−10事前登记，不放宽科学精确门，最大保存评分差{a['max_score_arithmetic_error']:.17g}。保存浮点分数的最低分/车槽/位置平分规则另外逐字核对，不声称另一NumPy逐位实现。

生产实际439加载文件/218module映射起止一致，独立{len(read(O/'AUDIT_SOURCE_START.json')['files'])}加载文件/{len(read(O/'AUDIT_SOURCE_START.json')['module_files'])}映射起止一致，并逐项核原生产加载身份；包含本原语实际NumPy/native/解释器和输入。只有这一个非模型候选枚举工作负载，full_scientific_loaded_closure仍False。每次局部枚举{min(m['walls_s'])*1000:.6f}至{max(m['walls_s'])*1000:.6f}毫秒；输入读取/已知二值标签创建/报告与闭包封存在局部计时外，没有可中止worker/回收/完整10秒请求资格。

仅实际可行候选原语和固定30状态资格。新组件未接入完整生产greedy循环、实际MaskCO输出/有方向目标头/训练或动态在线控制；没有fullC0/最终成本/新科学数据，不证明数值正向或全局可达。下一步另事前工程登记有方向车槽/插入锚点的实际评分目标、原完整动作链/各控制/预算，再另登记有限科学族；第3族尚未启动。不要重复本163/11316或旧875/1480/217/223等QA，不按临时结果选择模型、路线或样本。

原第2族注册SHA562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176、生产35FAIL/独立36注册屏障失败、训练/DEV/确认关闭保持。取货有界恢复/更多真实事件/剩余计算发布取消、正式学习/同初始化已训CE/独立强动态接拒/全日10秒及科学完整依赖仍待，旧13/1684等不重跑。负结果31必要门/183历史/C1原件保留，仅新增本证据边界。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。

16本地入口先备份更新，两个本人进程已退出，无服务器/GPU/共享MD上传，最终取货seed20260927/28未用。
''')
    boundary=R/'reproduction/FEASIBLE_ACTION_OPTIONS_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),runtime_checks=163,independent_checks=11316,failed=0,metrics=m,local_route_filter_and_retained_union_qualified=True,production_full_decoder_qualified=False,full_C0_physics_qualified=False,learned_or_online_qualified=False,scientific_family2_FAIL_preserved=True,new_scientific_family=False,stable_online_gain=False,registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_FEASIBLE_ACTION_OPTIONS.json')))
    backup=R/'current_entries_before_feasible_action_options_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_remaining_actions_20261012/UPDATE_IDENTITY.json')['files'];assert all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    note='\n> 2026-10-12配送实际候选原语：新事前登记固定30首偏离骨架，完整保留边guard接实际原容量/TW/返仓过滤，1525位置全保留；30目标动作全合格，但同订单最低分仅12匹配目标，18仍偏离。163运行/11316独立项0失败，仅原语/局部过滤，无完整重构探针/模型/C0/最终成本/第3族，原第2族FAIL保持，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['delivery'].update(feasible_action_options_directory=str(O.relative_to(ROOT)),feasible_action_options_runtime_checks=163,feasible_action_options_independent_checks=11316,feasible_action_options_independent_failed=0,feasible_action_candidate_positions=1525,target_local_route_filter_qualified=30,target_new_local_best_count=12,local_feasible_option_primitive_qualified=True,modified_production_decoder_qualified=False,full_scientific_loaded_closure_qualified=False,next='Do not repeat163/11316 or closed family2/old QA. New prospective engineering must align directed-slot and temporary insertion-anchor scoring with actual masked action chain;18 of30 local best options remain non-target even with all30 target route filters passing. No full reconstruct/new science/third family until prospective engineering qualification and scientific registration. Continue pickup bounded restore/actual events/remaining cancellations/formal models CE and strong dynamic/full-day10s/full dependency gaps.')
            obj['reproduction']['feasible_action_options_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'02_配送线':new+='\n- [保留边修正与实际可行候选枚举](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(dest),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert len(updates)==16 and all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=owned,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),runtime_checks=163,independent_checks=11316,failed=0,owned_processes=owned)))
if __name__=='__main__':main()
