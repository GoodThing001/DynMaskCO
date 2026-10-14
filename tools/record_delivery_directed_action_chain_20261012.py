"""Scoped engineering archive, immutable prior artifacts, sixteen local entries."""
import hashlib,json,shutil,os
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'delivery/directed_action_chain_root_v1'
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
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_DIRECTED_ACTION_CHAIN.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(163,0,1886864,0)
    for n in ('REGISTRATION.json','AUDIT_REGISTRATION.json'):assert all(sha(p)==h for p,h in read(O/n)['inputs'].items())
    assert read(O/'RUN_EXIT.json')['exit_code']==read(O/'AUDIT_RUN_EXIT.json')['exit_code']==0
    names=('delivery_directed_action_chain_v1.py','qualify_delivery_directed_action_chain_v1.py','audit_delivery_directed_action_chain_v1.py')
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name in names for x in p.info['cmdline'] or [])];assert not active
    frozen={}
    for n in names:
        dest=O/('FROZEN_'+n);shutil.copy2(ROOT/'tools'/n,dest);frozen[str(dest)]=sha(dest)
    create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,independent_actual_exit=0,owned_processes_active=active,observed_from='Actual local exec completion; no server/GPU'))
    report=E/'02_配送线/文档/结果与进度/配送_有方向车槽理想评分接入完整掩码动作链组件_2026-10-12.md'
    walls=[x['wall_s'] for x in r['metrics']];s=read(O/'SOURCE_START.json');i=read(O/'AUDIT_SOURCE_START.json')
    create(report,f'''# 有方向车槽理想评分接入完整掩码动作链组件（2026-10-12）

delivery/directed_action_chain_root_v1在执行前登记，固定使用此前30个不同邻接错配记录的原完整掩码，每例25订单；一次新的隔离动作链调用，从各例原initial去掉全部masked订单开始，不以原首偏离骨架作为新起点。750次实际插入、370500个所有剩余订单/车槽/位置候选逐步保留，包含排除原因、方向锚点、评分、每客户最佳候选、regret、选择、前后状态。ACTION_CHAINS.jsonl原字节22307834字节，逐例写入并另进程流式复核。没有读取新科学样本，没有调用原注册reconstruct/二值邻接guide1探针、教师求解器、模型或C0/最终成本，没有重跑原163/11316等完成QA。

新组件调用先前已验的可行候选原语，将每个真实临时骨架上的(customer,vehicle,position,predecessor,successor)输入交给有限标量评分接口。此次唯一评分来源是明确的已知教师理想fixture：车槽一致、插入后与目标方向秩一致时为1，其它为0。它使用目标完整路线，不能用于实际部署，更不是现有MaskCO模型输出。排序先最大动作分数，再最小距离增量/原距离尺度，最后车槽/位置；regret为前两候选动作分数差，只有一个可行候选优先，客户平分取ID最小。每步重新构造当前锚点和评分，未用最终固定边代替临时插入动作。保留边采用完整插入后联合检查，实际原route_ok容量/TW/返仓过滤仍沿用。

163运行检查0失败，1886864独立核验项0失败，两个实际进程退出0。30/30动作链均完成、逐车槽有方向顺序精确等于目标、每例25个掩码订单各一次，所有被选动作理想分数为1。独立不用新动作链/评分fixture/候选原语/guard/原route_ok或matrices作参考，直接逐候选坐标距离、容量/ready/reveal/服务开始TW/返仓时钟、全保留边集合和插入后目标序列过滤核验；保存分数、车槽位置平分、regret及客户竞争/每步转移精确复算。工程距离算术绝对1e−10事前登记，最大差{a['max_geometry_error']:.17g}，不修改原科学精确门或原route_ok1e−9。1886864是大量候选字段/转移/身份检查数量，不是独立科学样本数。

只证明在这30个已观察输入上，新动作消费者存在这组已知目标的理想输入且完整顺序可执行；不证明现有模型能生成它、其它状态可达、学习分布覆盖或正向增益。与旧解码器相比，联合保留边判断、动作表示、评分及regret均有变化，未做隔离因果对照；不能单独归因于方向编码或一处guard修正。此新组件未替换生产解码器，不修复原第2族的特定二值输入FAIL，不允许用本结果放行旧族训练。四组方向别名原83/104与30项875/1480、217/223及163/11316证据沿用，不重复。

本地实际Python3.12.7/NumPy1.26.4/CPU7、BLAS单线程；生产{s['files'].__len__()}加载文件/{len(s['module_files'])}映射与独立{len(i['files'])}文件/{len(i['module_files'])}映射起止一致，并核验所有生产加载原字节与实际解释器。仅本无模型动作链负载闭包，不套用3080环境、不升级正式科学完整依赖资格。每例动作计算{min(walls):.9f}至{max(walls):.9f}秒；读取已封存输入、已知教师fixture准备、闭包封存和归档在组件计时外。合作式10秒仅循环边界检查，没有worker抢占/回收/任意评分内部取消/慢磁盘资格。接口非法分数和预算失败分支没有本轮运行验收，不声称完整部署预算合格。无原完整C0温度品质认证、择优部署或效用统计。

下一步正式评分来源/目标张量与临时锚点逐一绑定、各控制/实际模型完整加载、认证和可中止整请求预算工程；科学第3族须另数据前登记，不能把这组教师理想输入当科学训练或学习收益。取货仍从选定epoch5继续真正未决order7、剩余计算/发布故障、正式已训同初始化CE/独立强动态接拒和全日10秒/科学完整依赖待闭合；取消副本order7临时接受不计调用端执行。负结果31必要子集/183历史/C1FAIL封存不动。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。

原第2族注册SHA562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176及生产35FAIL/独立36注册屏障失败保持，科学三臂训练/DEV/确认关闭，第3族未启动。16当前入口保存旧字节后更新，两个本人进程退出，无服务器/GPU/共享MD上传，最终取货20260927/28未用。
''')
    boundary=R/'reproduction/DIRECTED_ACTION_CHAIN_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_DIRECTED_ACTION_CHAIN.json'),requests=30,actions=750,candidate_positions=370500,runtime_checks=163,independent_checks=1886864,failed=0,ideal_known_teacher_fixture_only=True,action_consumer_scoped_qualified=True,production_decoder_modified=False,learned_output_qualified=False,C0_or_final_cost_calls=0,hard_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,scientific_family2_FAIL_preserved=True,new_scientific_family=False,stable_online_gain=False))
    backup=R/'current_entries_before_directed_action_chain_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_feasible_action_options_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        f=E/x['path'];dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    note='\n> 2026-10-12配送新动作链组件：事前登记固定30旧完整掩码，理想有方向车槽/临时锚点评分接新隔离客户regret循环；750插入/370500候选全保留，30精确重现目标，163运行/1886864独立项0失败。仅已知教师理想输入，评分/表示/guard/regret同时变化，无学习/因果归因/完整C0/可中止预算或新科学族；生产及第2族FAIL保持，在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(prior);obj['updated']=datetime.now(timezone.utc).isoformat();obj['delivery'].update(directed_action_chain_directory=str(O.relative_to(ROOT)),directed_action_chain_runtime_checks=163,directed_action_chain_independent_checks=1886864,directed_action_chain_independent_failed=0,directed_action_chain_requests=30,directed_action_chain_actions=750,directed_action_chain_positions=370500,directed_action_chain_target_exact=30,directed_action_consumer_ideal_fixture_qualified=True,directed_action_learned_scorer_qualified=False,modified_production_decoder_qualified=False,full_scientific_loaded_closure_qualified=False,next='Do not repeat new163/1886864 directed action chain or old QA/closed family2. Fixed30 ideal teacher-conditioned directed-slot scalar scores reproduced750 full masked actions,370500 candidates; this is not learned output or isolated causal direction/guard effect. Next qualify score-source/target tensors and temporary anchor binding, actual models/controls, full C0 and interruptible request budget before separately preregistering family3/new data. Prior family2 stays FAIL with training/DEV/confirmation disabled. Continue pickup epoch5 actual pending order7 and remaining faults/formal CE/strong dynamic/full-day10s/dependency gaps.')
            obj['reproduction']['directed_action_chain_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'02_配送线':new+='\n- [有方向车槽理想评分与完整动作链组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True));print(json.dumps(dict(entries=len(updates),runtime_checks=163,independent_checks=1886864,failed=0,owned_processes=active)))
if __name__=='__main__':main()
