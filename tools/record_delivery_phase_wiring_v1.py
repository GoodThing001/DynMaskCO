"""Read/AST-review prepared bytes, record narrow scope; execute no workload."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/phase_timing_wiring_preparation_root_v1'
OLD=R/'delivery/timed_actual_request_execution_root_v1/downloaded'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def method(tree,name):
    return next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)

def main():
    reg=read(O/'PREPARATION_REGISTRATION.json')
    assert reg['execution_authorized'] is False
    assert all(sha(O/n)==h for n,h in reg['package_files'].items())
    assert all(sha(ROOT/n)==h for n,h in reg['basis_files'].items())
    trees={p.name:ast.parse(p.read_text(encoding='utf-8')) for p in sorted(O.glob('*.py'))}
    checks=[]
    def check(n,v):
        checks.append(dict(name=n,pass_condition=bool(v)))
        if not v:raise ValueError(n)
    oldtree=ast.parse((OLD/'delivery_timed_actual_engine_v3.py').read_text(encoding='utf-8'))
    check('No sample warmup or initialization modification',ast.dump(method(oldtree,'__init__'))==ast.dump(method(trees['delivery_phase_actual_engine_v1.py'],'__init__')))
    measured=[]
    for n,t in trees.items():
        for x in ast.walk(t):
            if isinstance(x,ast.Call) and x.args and isinstance(x.args[0],ast.Constant):
                f=x.func
                if isinstance(f,ast.Name) and f.id=='m' or isinstance(f,ast.Attribute) and f.attr=='measure':
                    measured.append(dict(file=n,line=x.lineno,phase=x.args[0].value))
    check('All nineteen phases wired in source',set(x['phase'] for x in measured)==set(range(1,20)))
    engine=(O/'delivery_phase_actual_engine_v1.py').read_text(encoding='utf-8')
    check('Encoder actual device sync explicit', 'jax.block_until_ready(encoded)' in engine)
    check('Decoder/head actual device and host sync explicit','jax.block_until_ready((scores,latent))' in engine and 'return np.asarray(scores),np.asarray(latent)' in engine)
    check('End source and both parameter checks inside request before report return',
          engine.index('end=m(14,0,self.closure')<engine.index('unchanged=m(15,0,parameter_gate)')<engine.index('return m(16,0,source_report_gate)'))
    check('Full model and head inventories evaluated', 'model_identity=self.inventory(self.model);head_identity=self.inventory(self.head)' in engine)
    parent=(O/'delivery_phase_shared_parent_v1.py').read_text(encoding='utf-8')
    check('Separate worker and caller trace storage',"tagname=spec['trace']" in parent and 'caller_buffer=bytearray(64+64*MAX_RECORDS)' in parent)
    check('Normal worker raw collection after actual wait/reap',parent.index('child.wait(timeout=max(')<parent.index("save_bytes(base/(tag+'_WORKER_TRACE.bin')"))
    check('Normal both trace archives and cleanup before finished',parent.index("tag+'_CALLER_TRACE.bin'")<parent.index('finished=time.perf_counter()'))
    check('Unchanged eight plus two seconds', 'deadline=started+8.;total_deadline=started+10.' in parent)
    check('Selection preserved through post-publication stall','Selected output remains selected' in parent)
    parser=trees['independent_phase_trace_parser_v1.py']
    imports=[]
    for x in ast.walk(parser):
        if isinstance(x,ast.Import):imports += [n.name for n in x.names]
        elif isinstance(x,ast.ImportFrom):imports.append(x.module)
    check('Independent parser uses only standard library',set(imports)=={'hashlib','math','struct','fractions'})
    check('No actual execution registration present',not (O/'EXECUTION_REGISTRATION.json').exists())
    review=dict(kind='Static source inspection only, not runtime QA',checks=checks,measured_call_sites=measured,
        source_identity={n:sha(O/n) for n in reg['package_files']},
        execution_authorized=False,trace_or_parser_executed=False,model_executed=False,
        notable_new_engine_semantics='Both full inventories evaluated rather than old boolean short-circuit; numerical method/parameters unchanged',
        known_static_blockers=[
            'Inherited caller exception/finally path lacks guaranteed raw trace retention and full post-cleanup failure clock; normal path only is wired. Must fix in a new source copy before execution; this is source inspection, not observed runtime failure.',
            'Phase19 includes waiting for worker until deadline, then kill/reap/collection; it is not pure kill latency. Final caller trace write/archive/map close are outer-clock gaps, not included in narrow phase19 operation span.',
            'Stage evidence enumeration/hash/marker JSON and some argument host materialization are outer-clock gaps. NPZ/write and enumeration/hash closure remain combined; do not claim internal separation.',
            'Independent parser source has not run; no torn/kill/open/nested/overflow runtime validation.',
            'Full actual new request replay audit, precise limited execution/supervision/transport and real owner/resource checks not sealed.'],
        source_review_is_budget_or_science_qualification=False)
    create(O/'SOURCE_REVIEW.json',review)
    prior=read(R/'current_entries_before_phase_timing_preparation_20261012/UPDATE_IDENTITY.json')['files']
    assert len(prior)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in prior)
    backup=R/'current_entries_before_phase_timing_wiring_20261012';backup.mkdir(exist_ok=False)
    for x in prior:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    report=E/'02_配送线/文档/结果与进度/配送_分阶段计时实际接线与独立字节解析源码准备_2026-10-12.md'
    prose='''# 分阶段计时实际接线与独立字节解析源码准备

新增独占phase_timing_wiring_preparation_root_v1，仅新源码/AST与逐字身份。固定旧首项key[12,3]/25masked、两initial臂/同headseed20261012/0更新、8秒执行+2秒回收，全部纯当前输入与head/消费者/可行过滤/评价模块原字节保持。没有导入或运行trace/parser、模型、C0、solver、旧请求/QA，没有上传研究包或共享MD。

新engine把输入、完整10D投影、incumbent C0、实际同步encoder、当前全部可行动作/锚点、实际同步decode/final_norm/head及host转换、NPZ保存、regret消费、JSON保存/SHA、停止链、候选C0与严格择优、完整实际依赖与全参数核验接19阶段。初始化AST与原engine一致，无当前样本模型/C0预热。新参数段显式计算两份完整inventory，不沿用原boolean短路；它增加核查覆盖，不改变模型/参数/数值门。依赖/参数资格门仍在请求内、合格发布前，不能挪出时窗宣称速度。

新worker单writer共享buffer与caller私有buffer分开。正常路径实际wait/reap后才复制保存worker原buffer，并把caller trace保存/映射日志清理计入外层elapsed；阶段18保留原SHA/source/C0严格择优与截止前选择，已选输出不因发布后stall回滚。阶段19包括等待截止、终止/回收/收集，不是纯kill延迟；最后记录写、caller trace归档/关闭及若干marker/参数构造成本是外层计时可见间隙，不能将operation和冒充完整请求。NPZ序列化+写、依赖枚举+hash暂为组合段，未证明内部已分离。

独立parser源码不导入writer/engine/模型，用stdlib逐字schema/上限/序号/19阶段/状态/保留padding/未使用尾部/外层时钟边界检查，用binary64精确有理数算区间并集与外层间隙，避免嵌套求和。open不算完成、非法/残写拒绝，close不等候选/源/科学资格。尚未运行parser或验证任何实际trace，不证明并发、内存原子性或任意native点。

静态审查明确留有执行阻挡：原caller异常finally路径尚无可靠raw trace归档和清理后失败时钟，须在新副本修正；正常路径接线不冒充异常路径闭合。完整新实际请求重放核查器、parser运行验证、准确限定执行登记/监督传输与实际资源/本人树核查仍未封存。当前没有EXECUTION_REGISTRATION，execution_authorized=False，禁止盲跑源码。此项不是观察到的运行bug或预算PASS。

正式已训同初始化CE/同任务随机、各控制、等耗时MaskCO迭代、独立强动态接拒/科学完整依赖与整请求预算仍缺；第2族FAIL封闭，第3族未启。取货已权威终局64/callback80/pending空，无81；不重复旧无模型QA。所有旧FAIL、ledger/13800字节缺口、31必要门/183/C1保持，最终20260927/28未用。稳定可归因MaskCO核心在线增益仍未建立。
'''
    create(report,prose)
    boundary=R/'reproduction/PHASE_TIMING_WIRING_PREPARATION_BOUNDARY_20261012.json'
    summary=dict(directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),source_review_sha256=sha(O/'SOURCE_REVIEW.json'),
        engine_worker_caller_source_wired=True,independent_parser_source_prepared=True,
        trace_or_parser_executed=False,model_executed=False,abnormal_cleanup_source_closed=False,
        execution_authorized=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,family3_started=False,stable_online_gain=False)
    create(boundary,summary)
    note='\n> 2026-10-12 分阶段计时接线源码准备：新phase_timing_wiring_preparation_root_v1封存engine/worker/caller十九阶段和独立stdlib字节parser，仅AST/字节未运行；正常路径两份trace收集清理计入外层8+2，JAX同步显式、依赖与两份完整参数核验仍在请求内。静态发现继承异常清理尚缺原buffer保存及清理后失败钟，须新副本修正；parser/完整新请求核查与准确执行登记未齐，execution_authorized=False，不盲跑。未运行模型/C0/旧QA、无上传/科学族；第2族FAIL封闭，第3族未启，取货终局64/pending空无81，稳定可归因在线增益仍未建立。\n'
    next_action='Source-only phase_timing_wiring_preparation_root_v1 seals actual engine/worker/caller nineteen-phase wiring and independent stdlib raw parser. No modules imported or trace/parser/model/C0 executed. Before execution fix inherited abnormal caller cleanup in a new copy (raw retention and post-reap complete failure elapsed), validate independent parser on registered engineering trace boundaries, seal complete new actual request replay checker and exact limited registration/supervision/transport, then check actual RAM>=8GiB/CPU15/owner tree. Keep all telemetry/source/parameter checks in8+2; no old10/100/220 or completed QA rerun, sample warmup or threshold/time relaxation. Formal controls/equal-time iteration/strong dynamic/science/fullrequest remain missing; family2closed/family3notstarted; pickupterminal64/pendingempty/no81.'
    changes=[]
    for x in prior:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(phase_timing_wiring_preparation=summary,next=next_action)
            w['reproduction']['phase_timing_wiring_preparation']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [分阶段接线与独立解析源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    shutil.copy2(Path.home()/'.codex/automations/maskco/automation.toml',O/'AUTOMATION_BEFORE_WIRING_PREPARATION.toml')
    addition='\n2026-10-12配送分阶段计时接线与独立字节解析源码准备：delivery/phase_timing_wiring_preparation_root_v1封存新engine/worker/caller19阶段和不导入writer/model的stdlib parser。固定原key[12,3]/25masked、同initial头seed20261012/0更新/8+2，纯当前输入及head/消费者/过滤/物理原字节，初始化AST相同不预热样本；新参数段显式计算两份完整inventory，依赖参数资格仍计时内发布前。两writer各独立buffer，正常wait/reap后原worker trace及caller trace归档清理计外层elapsed，JAX真实同步拟接线；phase19含等待不是纯kill、最终trace写/归档关闭及部分marker为外层间隙，组合NPZ写/枚举hash未内部拆分。独立parser拟schema/序号/保留padding/尾部/时钟/有理数区间并集，open不算完成、close不等资格。全部仅AST/字节，0trace/parser/model/C0/solver/旧QA运行，无服务器/GPU/共享MD/最终种子。静态发现继承caller异常finally缺原buffer可靠保存及清理后完整失败钟，须新副本修正；未执行故不称运行bug。下一先修正异常源、另登记parser工程边界验证、完整新请求重放/准确限定登记监督传输及实际资源本人树，execution_authorized=False，不盲跑。不得重跑旧10/100/220或完成QA、不放宽8秒/样本预热/按成本换头；正式已训同初始化CE/同任务随机/各控制/等耗时迭代/强动态/科学完整依赖/全请求仍缺，第2族FAIL封闭、第3族未启；取货终局64/callback80/pending空无81。新报告配送_分阶段计时实际接线与独立字节解析源码准备_2026-10-12.md、reproduction仅新PHASE_TIMING_WIRING_PREPARATION_BOUNDARY_20261012.json及16入口WORK_STATUS先存旧字节更新，旧FAIL/ledger/13800/31/183/C1保持。稳定可归因核心在线增益未建立，maskco ACTIVE30分钟，准备/常规未变静默。'
    create(O/'AUTOMATION_ADDITION.json',dict(addition=addition))
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(AST_review_checks=len(checks),all_expected_static_conditions=True,
        known_static_blockers_preserved=True,entries=16,report_sha256=sha(report),boundary_sha256=sha(boundary),
        identity_sha256=sha(backup/'UPDATE_IDENTITY.json'),no_workload_import_or_execution=True))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    print('Static source review recorded; execution blocked, old bytes and16 backups preserved.')

if __name__=='__main__':main()
