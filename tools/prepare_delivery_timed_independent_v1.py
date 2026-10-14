"""Freeze a new source-only checker/evidence draft; never import runtime modules."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
OLD=R/'delivery/timed_actual_request_preparation_root_v1';O=R/'delivery/timed_request_independent_preparation_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    previous=read(OLD/'PREPARATION_REGISTRATION.json')
    assert sha(OLD/'PREPARATION_REGISTRATION.json')=='f3be1985d254a2fd652384fb1d5a4e573cf3a4e21f58b099d665789d42b89035'
    for n,h in previous['package_files'].items():assert sha(OLD/n)==h
    names=['delivery_timed_actual_engine_v2.py','delivery_timed_shared_worker_v2.py','delivery_timed_shared_parent_v2.py','audit_delivery_timed_request_v1.py']
    deps=['delivery_actual_score_chain_v1.py','delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py','delivery_actual_action_head_v1.py','action_head_runtime_v1.py','features_objective.py','CHAIN_INPUT.json']
    for n in names:ast.parse((ROOT/'tools'/n).read_text(encoding='utf-8'),filename=n)
    for n in deps:
        assert sha(OLD/n)==previous['package_files'][n]
        if n.endswith('.py'):ast.parse((OLD/n).read_text(encoding='utf-8'),filename=n)
    audit=(ROOT/'tools/audit_delivery_timed_request_v1.py').read_text(encoding='utf-8')
    imports=[]
    for node in ast.walk(ast.parse(audit)):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    forbidden=['delivery_timed_actual_engine','delivery_timed_shared_worker','delivery_timed_shared_parent','delivery_actual_action_head','delivery_actual_score_chain','delivery_feasible_action_options','delivery_retained_edge_action_guard','features_objective']
    assert not any(m.startswith(f) for m in imports for f in forbidden)
    row=read(OLD/'CHAIN_INPUT.json');assert row['key']==[12,3] and len(row['remaining'])==25
    latest=R/'current_entries_before_timed_actual_request_preparation_20261012/UPDATE_IDENTITY.json'
    entries=read(latest)['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    O.mkdir(exist_ok=False)
    for n in names:shutil.copy2(ROOT/'tools'/n,O/n)
    for n in deps:shutil.copy2(OLD/n,O/n)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Source-only independent timed partial/full replay and durable stage evidence; no actual execution.',
        previous_preparation_sha256=sha(OLD/'PREPARATION_REGISTRATION.json'),previous_package_files=previous['package_files'],
        package_files={n:sha(O/n) for n in names+deps},input_sha256=sha(O/'CHAIN_INPUT.json'),input_digest=previous['input_digest'],
        key=[12,3],masked_count=25,arms=previous['arms'],head_seed=20261012,optimizer_steps=0,
        total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,cpu=15,minimum_available_ram_gib=8,
        proposed_requests=[dict(arm=a,fault_stage=0) for a in previous['arms']]+[dict(arm='pretrained_initial',fault_stage=s) for s in range(1,9)],
        proposed_requests_are_not_execution_authorization=True,
        changes=['FEATURES/incumbent C0/encoded/chain stop/C0 selection durable before matching completed-stage markers.',
            'Stage markers bind already saved evidence; fault-only loaded closure and actual delay-entry clock recorded before fixed12s delay inside unchanged8s cutoff.',
            'Original serialized response retained before stage7; publication record precedes stage8, selected postpublish output cannot be rolled back.',
            'Caller separate writer/validation/selection/kill/reap/deadline clocks; actual pid/create_time/cmdline/owner evidence, PID reuse distinguished.',
            'Caller mmap/log cleanup included before end timestamp; receipt file creation explicitly outside elapsed, caller synchronous copy/SHA/JSON/archive limitations retained.',
            'Independent original parameter/head reload/full10D/anchors/fresh full forward versus saved cache/JAX scores plus2e-5 NumPy arithmetic, local filters/regret/partial transition/fullC0/selected bytes/stages/timing/source replay source.'],
        algorithm_unchanged=True,original_engine_v1_not_modified=True,old_initial_scores_reused=False,
        audit_imports=sorted(set(imports)),independent_timed_checker_source_prepared=True,
        independent_checker_actual_execution_validated=False,exact_execution_registration_sealed=False,
        exact_driver_supervision_transport_sealed=False,source_review_complete=False,execution_authorized=False,
        model_imported=False,forward_executed=False,C0_executed=False,timed_request_executed=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,learned_output_qualified=False,
        formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        numpy_head_max_abs_error=0.00002,
        limitations=['Only AST/bytes prepared, not an exercised audit or timed budget qualification.',
            'Original shared Transformer/JAX and C0 physics remain shared reference implementations.',
            'Fault label without actual delay entrance is unexercised; preserve unreached fault/partial writes/timeout without retry or extra warmup.',
            'A process killed during NPZ/JSON write may leave an incomplete artifact; preserve raw bytes and audit failure, do not fabricate completion.',
            'Source-end and fault-entry snapshots differ; killed worker has no natural postfailure closure.',
            'Parent prepared public input/model startup outside window; no live private event projection/vehicle installation.',
            'Caller synchronous copy/SHA/JSON/archive/cleanup do not prove arbitrary-load/slow-disk/native instruction preemption.',
            'Fault evidence capture consumes the registered time; no time is added to reach later stages.',
            'Formal trained CE/task-random/other controls/equal-time iteration/independent strong dynamic/full scientific chain still absent.'])
    create(O/'PREPARATION_REGISTRATION.json',plan)
    note='\n> 2026-10-12 配送计时核查源码准备：timed_request_independent_preparation_root_v1另封存v2工程证据接线和独立timed/partial/full核查源码，旧v1登记及十文件原字节保持。完整10D/原参数/同seed头/当前锚点和保存新请求前向拟独立复算；局部过滤/保留边/regret/部分链/原全C0/截止前选择与实际kill-reap拟核。完成标记前保存特征、原计划C0、encoder、链停点及候选C0；故障前实际闭包/延迟入口计入8秒不加时。父明确writer/validation/selection/kill/reap与cleanup计时，PID复用按create_time和命令区分，receipt文件创建仍在记录时长外。只有AST与字节准备，0模型导入/forward/C0/计时执行/服务器上传，不是旧QA重跑；实际checker资格/源码复核/准确执行登记、driver/监督/派发下载仍缺，execution_authorized=False。未到故障入口如实未验，残写原件和审计失败保留，发布后截止前已选结果不能回滚。仅预备公共快照、同步父处理/慢磁盘/native任意指令/现场安装和科学全链仍未合格；第2族FAIL封闭、第3族未启动、正式已训CE/任务随机/等耗时/独立强动态仍待。取货权威终局64/pending空无81，旧QA/FAIL/ledger及13800缺口不改；稳定可归因核心在线增益仍未建立，无GPU/共享MD/最终种子使用。\n'
    report=E/'02_配送线/文档/结果与进度/配送_计时请求独立核查与阶段原件留存源码准备_2026-10-12.md'
    create(report,'# 计时请求独立核查与阶段原件留存源码准备\n'+note+'\n静态检查发现旧草案若在特征/encoder/链停点/完整C0已完成入口被终止，相关结果可能还未落盘。新独占v2先保存这些实际结果，再发布对应阶段标记；不更改模型、头、掩码、目标、动作选择或8秒执行预算。保存文件与故障入口身份采集也消耗计时预算，不能据此要求延长期限取得完整链。旧草案没有执行，本次也仅源码准备，不能把源码问题写成已观察运行故障或收益。\n\n独立核查器不用新engine、head、consumer、过滤器、feature mapper或共享worker/caller作参考。拟从原checkpoint重载234叶/76,263,424参数和fresh888、同seed四叶头，独立构建可见10D与每个当前骨架锚点；仅复算新计时请求实际保存的前向、已完成插入及原全C0。原JAX/Transformer和物理评价器仍共享，不属于另一模型或另一物理实现。2e-5仅沿用头的NumPy/JAX跨库算术界限，参数/特征/latent/JAX评分仍精确，不改旧科学门。\n\n核查器逐项即时落盘并单独报告未到达故障，不能靠指定标签或最后stage编号宣称延迟已进入。父保存实际进程身份、单独选择时间与kill/reap，终止后的记录不得伪称自然0退出。父归档与receipt创建、共享资源日初准备的计时范围明示；任意父负载或磁盘硬实时、真正现场所有权/车辆安装与正式科学控制仍不能从这份源码获得资格。准确执行登记/限定driver/有界监督/派发下载与进一步静态审查尚未封存，禁止启动任何草案。\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    boundary=R/'reproduction/TIMED_REQUEST_INDEPENDENT_PREPARATION_BOUNDARY_20261012.json'
    create(boundary,dict(directory=str(O),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),source_only=True,
        execution_authorized=False,model_executed=False,C0_executed=False,old_QA_repeated=False,
        full_request_budget_qualified=False,formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False))
    backup=R/'current_entries_before_timed_request_independent_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    next_action='New timed_request_independent_preparation_root_v1 seals source-only v2 durable completed-stage evidence and independent timed/partial/full checker. No model/forward/C0/request executed. Exact source review/execution registration/driver/supervision/dispatch/download still missing; do not run draft. Marker/actual delay arrival distinct, partial/torn artifacts kept, selected postpublish output not rolled back; fixed10/8/2, no sample warmup/deadline extension. Before actual execution check CPU15/RAM8GiB and real owned process tree. Public-only/caller synchronous archive and wholebudget/science/formaltrainedCE/taskrandom/equal-time/strongdynamic remain unqualified. Family2 closedFAIL/family3 unstarted; pickup terminal64/pendingempty/no81. No oldQA repeat or stableonlinegain.'
    changes=[]
    for x in entries:
        p=E/x['path'];text=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(text);w['updated']=datetime.now(timezone.utc).isoformat()
            w['delivery'].update(timed_request_independent_preparation_directory=str(O.relative_to(ROOT)),
                timed_request_independent_preparation_sha256=sha(O/'PREPARATION_REGISTRATION.json'),
                independent_timed_checker_source_prepared=True,independent_timed_checker_executed=False,next=next_action)
            w['reproduction']['timed_request_independent_preparation_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+text if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else text+note
            if p.parent==E/'02_配送线':new+='\n- [计时请求独立核查源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),report_sha256=sha(report),
        boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),model_executed=False))
    print(json.dumps(dict(directory=str(O),files=len(plan['package_files']),entries=len(changes),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),execution_authorized=False)))
if __name__=='__main__':main()
