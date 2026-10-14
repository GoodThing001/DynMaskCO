"""AST/byte preparation only. Never import or execute the prepared trace."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=R/'delivery/phase_timing_preparation_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    code=O/'bounded_phase_trace_v1.py';ast.parse(code.read_text(encoding='utf-8'))
    design=read(O/'PHASE_DESIGN.json');assert design['execution_authorized'] is False
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),kind='source_preparation_not_experiment',
             package_files={p.name:sha(p) for p in [code,O/'PHASE_DESIGN.json']},
             preparation_tool_sha256=sha(__file__),basis_sha256=sha(R/'delivery/timed_request_timing_boundary_diagnosis_root_v1/DIAGNOSIS.json'),
             AST_parsed=True,module_imported=False,trace_executed=False,model_executed=False,execution_authorized=False,scientific_family3_started=False)
    create(O/'PREPARATION_REGISTRATION.json',reg)
    prior=read(R/'current_entries_before_timed_boundary_diagnosis_20261012/UPDATE_IDENTITY.json')['files']
    assert len(prior)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in prior)
    backup=R/'current_entries_before_phase_timing_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in prior:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    report=E/'02_配送线/文档/结果与进度/配送_分阶段计时与有界原始遥测源码准备_2026-10-12.md'
    prose='''# 分阶段计时与有界原始遥测源码准备

依据已闭合离线计时边界诊断，新增独占phase_timing_preparation_root_v1，只封存计时库源码与19阶段设计。AST解析/逐字身份完成，没有导入或运行该库、模型、C0、solver、旧请求或旧QA；不新增科学族或数据。

计时库拟使用调用端事前分配的固定缓冲：64字节header、每记录64字节、最多4096记录/262208字节，单writer。每同步调用保存请求序号、phase、step、span开始、操作开始/结束与open/success/exception状态，容量不足显式失败，不悄悄截断。调用端完整计时仍计入所有记录写、操作、最终读取/保存与取消回收。操作段不包含最后trace写，不能把这些操作时长之和冒充完整请求；嵌套段重叠也不得求和。

buffer拟在核实际本人writer已退出回收后读取；kill可能留下open/残写/撕裂，保留原buffer、拒绝非法记录，不据标签算完成或重试追入口。未证明并发读取、原子发布、任意native内部指令点或父崩溃持久化。close仅遥测标记，不等科学依赖闭包/模型资格/候选完成或提交成功。

19阶段把输入投影、原incumbent与候选C0、同步encoder/decoder/head、当前候选及锚点、regret消费者、NPZ/JSON保存、SHA、依赖与参数检查、发布、父选择和回收区分。NPZ序列化+写、依赖枚举+hash暂为组合段，不能声称内部已分离；未来JAX段必须实际同步。原8秒执行+2秒回收及日初无样本前向/C0预热保持，全依赖与参数资格核验仍须在请求计时内，不能挪到窗外宣称速度资格。

尚缺实际engine/worker/caller接线、独立parser与算术/源身份核查器、精确执行登记、真实解释器/依赖/监督/派发下载与运行前资源本人树核查；execution_authorized=False，不盲跑库或旧草案。正式已训同初始化CE/同任务随机、各控制、等耗时MaskCO迭代、独立强动态/科学完整闭包与全请求仍待；第2族FAIL封闭，第3族未启动。取货已权威终局64/callback80/pending为空，无callback81。

准备不是运行资格或增益。所有旧生产/FAIL/183/C1/ledger/13800字节保持，没有服务器/GPU/共享MD上传、最终20260927/28使用。稳定可归因MaskCO核心在线增益仍未建立。
'''
    create(report,prose)
    boundary=R/'reproduction/PHASE_TIMING_PREPARATION_BOUNDARY_20261012.json'
    summary=dict(directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),source_prepared=True,AST_parsed=True,trace_executed=False,model_executed=False,execution_authorized=False,formal_controls_qualified=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,family3_started=False,stable_online_gain=False)
    create(boundary,summary)
    note='\n> 2026-10-12 分阶段计时源码准备：新phase_timing_preparation_root_v1封存固定单writer原始buffer遥测库与19阶段设计，最多4096×64+64=262208字节；只有AST/字节，0库运行/模型/C0/旧QA。保留8+2秒、全部trace成本/原依赖参数资格检查计时，open/残写拒绝、不把close当资格。实际接线、独立parser/核查、准确登记及资源监督传输未闭合，execution_authorized=False，不盲跑；正式控制/等耗时迭代/强动态/全科学/预算仍缺，第2族封闭、第3族未启；取货终局64/pending空无81，稳定可归因在线增益未建立。\n'
    next_action='Prepared bounded_phase_trace_v1.py and19-phase contract only, AST/bytes sealed; no trace/model/C0 execution. Next wire exact new engine/worker/caller spans with real sync and total elapsed including telemetry and mandatory end dependency/parameter checks; prepare independent parser/arithmetic/source checker and exact exclusive execution/transport registration before any bounded new engineering workload. No old10/100/220 or completedQA rerun, no8s relaxation/sample warmup. Formal trained controls/equal-time iteration/strong dynamic/science/fullrequest still missing; family2closed/family3notstarted; pickupterminal64/pendingempty/no81.'
    changes=[]
    for x in prior:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(phase_timing_preparation=summary,next=next_action)
            w['reproduction']['phase_timing_preparation']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [分阶段计时原始遥测源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    shutil.copy2(Path.home()/'.codex/automations/maskco/automation.toml',O/'AUTOMATION_BEFORE_PHASE_PREPARATION.toml')
    addition='\n2026-10-12配送分阶段计时与有界原始遥测源码准备：delivery/phase_timing_preparation_root_v1封存bounded_phase_trace_v1.py及PHASE_DESIGN19阶段，固定单writer header64/record64/最多4096即262208字节。只AST/字节，不导入/运行遥测库或模型/C0/solver/旧QA，不升级执行预算科学。记录拟含序号phase step实际span/op起止与open/success/exception，溢出失败；真实kill可能open/残写，保留buffer，实际writer身份退出/reap后读，尚无并发/原子/native内部保证。全部trace操作和收集仍计完整请求，操作段不含最后记录写、嵌套不得求和冒充完整时长；close不是科学闭包/候选/发布资格。19阶段仍NPZ序列化写/依赖枚举hash为组合，JAX须实际同步；原8+2和全源参数放行核验计时不挪窗外、不预热同样本。下一实际engine/worker/caller接线、独立parser/算术/源身份核查、准确登记/真实资源本人树/监督传输未齐，execution_authorized=False，不盲跑。本源码准备不重跑旧10/100/220或任何完成QA，第2族FAIL封闭、第3族未启。正式已训同初始化CE/同任务随机/各控制/等耗时MaskCO迭代/独立强动态/科学全依赖与全请求仍缺；取货终局64/callback80/pending空无81。新报告配送_分阶段计时与有界原始遥测源码准备_2026-10-12.md和16入口WORK_STATUS先存旧字节更新/reproduction新PHASE_TIMING_PREPARATION_BOUNDARY_20261012.json；旧FAIL/ledger/13800/31/183/C1不变，无服务器/GPU/共享MD/最终种子。稳定可归因MaskCO核心在线增益未建立，maskco ACTIVE30分钟，准备及常规未变静默。'
    create(O/'AUTOMATION_ADDITION.json',{'addition':addition})
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(entries=16,report_sha256=sha(report),boundary_sha256=sha(boundary),identity_sha256=sha(backup/'UPDATE_IDENTITY.json'),prepared_source_unchanged=sha(code)==reg['package_files'][code.name],execution_authorized=False))
    print('AST and bytes sealed; source remains unexecuted. 16 entry backups preserved.')
if __name__=='__main__':main()
