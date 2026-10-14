"""Archive new offline diagnosis and preserve all prior current-entry bytes."""
import hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
R=ROOT/'research/review_3080_20261010'
O=R/'delivery/timed_request_timing_boundary_diagnosis_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    d=read(O/'DIAGNOSIS.json');reg=read(O/'PREREGISTRATION.json')
    assert read(O/'RUN_EXIT.json')['exit_code']==0
    assert all(sha(ROOT/p)==h for p,h in reg['inputs'].items())
    prior=read(R/'current_entries_before_timed_actual_request_execution_20261012/UPDATE_IDENTITY.json')['files']
    assert len(prior)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in prior)
    backup=R/'current_entries_before_timed_boundary_diagnosis_20261012'
    backup.mkdir(exist_ok=False)
    for x in prior:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    report=E/'02_配送线/文档/结果与进度/配送_已存计时与证据采集耗时边界诊断_2026-10-12.md'
    text='''# 已存计时与证据采集耗时边界诊断

本次事前登记后只解析原十请求的保存时钟、下载清单及四个冻结源码的AST；没有导入模型、运行C0/求解器、重放前向或重新请求。以binary64时钟对应的精确有理数相减、保存每段hex与有理数，全部请求的分段和精确等于原首尾差。原生产、独立18647/24FAIL、220窄核验和所有旧QA均未重跑或改写。

两正常请求“特征完成→encoder完成”分别1.0547138000838459与1.0484482999891043秒，该段还包括原incumbent C0及保存；“encoder完成→首评分完成”分别0.6643542000092566与0.6546405996195972秒，包含候选枚举、锚点、decode/head、同步及NPZ保存。这些不是单独encoder或head速度。首评分完成后至8秒截止还分别有6.141295399982482与6.156070400029421秒；缺逐步计算、序列化、写入、hash时钟，不能从最终16插入或下载时间分离模型/I/O/JIT耗时或预测删去I/O后的完整链。

两个实际故障标记到延迟入口7.797150900121778与7.610975299961865秒，是标记发布、目录/包/stdlib/native枚举及hash、差分和JSON留存的组合。闭包同时包括实际module/native文件及额外已登记包/源/stdlib树，8896不是实际导入模块数量；1027是该负载module映射数量。只有这两个聚合区间被实际观测，不能把它们称hash单项成本或全部超时唯一原因。

原下载清单显示两正常request子目录保存10744502与10851477字节（包括随机残写），这是工件体积而非I/O耗时。冻结完整链程序还在stage6后、发布前进行完整依赖与参数身份核验；本十请求没有到达该段，实际耗时未知。禁止扣除故障仪器耗时、外推步数或把资格检查移出计时而声称原8秒通过。

下一有限工程须先封存分阶段实际起止时钟、计算/序列化/保存/hash/依赖检查的计时范围与有界证据格式，保留实际所有工作和原8+2秒合同；完整源码参数科学闭包仍为放行前必要门。当前只有诊断及设计约束，未执行新请求或准备可派发模型程序，execution_authorized=False。正式已训同初始化CE、同任务已训随机、各控制、等耗时MaskCO迭代、独立强动态与全科学依赖/完整请求仍待；第2族FAIL封闭，第3族未启动。

取货保持实际权威终局64/callback80/pending为空，无callback81。31必要门/183历史/C1与旧FAIL、ledger和13800字节缺口保持。没有服务器/GPU/共享MD上传或最终种子使用。稳定可归因MaskCO核心在线增益仍未建立。

读操作局限：此前本地Win32_Process查询Access Denied，不能据此断言当前无进程；一次只读路径拼写pretrain_initial错误触发FileNotFoundError后按实际pretrained_initial修正读取。没有进程终止、实验恢复或原工件改变。
'''
    create(report,text)
    boundary=R/'reproduction/TIMED_BOUNDARY_DIAGNOSIS_EVIDENCE_20261012.json'
    summary=dict(directory=str(O.relative_to(ROOT)),diagnosis_sha256=sha(O/'DIAGNOSIS.json'),registration_sha256=sha(O/'PREREGISTRATION.json'),requests=10,offline_only=True,model_executed=False,old_QA_repeated=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,family3_started=False,stable_online_gain=False)
    create(boundary,summary)
    next_action='Offline timing-boundary diagnosis completed from all ten archived receipts and frozen AST only; exact clock partitions and evidence byte accounting do not isolate model/IO/JIT or establish counterfactual speedup. Next prospectively prepare phase timing and bounded evidence design with all actual work and mandatory eligibility closure accounted, then independent bounded execution registration; do not rerun old10/100/220 QA, relax8s or warm same sample. Formal trainedsameinitCE/tasktrainedrandom/controls/equal-time iteration/independentstrongdynamic/fullscience/fullrequest remain missing before family3. Pickup terminal64/pendingempty/no81; stable attributable online gain not established.'
    note='\n> 2026-10-12 新离线计时边界诊断：仅原十请求保存时钟/manifest与四冻结源码AST，精确有理数分段；特征→encoder及encoder→首评分是含C0/候选/保存的组合，缺逐步模型/I/O/JIT隔离钟。故障标记→延迟7.797150900121778/7.610975299961865秒含枚举/hash/差分/JSON，8896为加载与注册联合树而非实际导入模块数量；完整链发布前依赖/参数核验尚未实际到达，不推断删I/O后速度。下一先封存实际阶段钟/有界证据与全部计时范围，不挪资格检查冒充通过、不重跑旧10/100/220或旧QA。新请求execution_authorized=False，正式控制/科学闭包/完整预算/第3族未升级；取货终局64/pending空无81，稳定可归因在线增益未建立。\n'
    changes=[]
    for x in prior:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat()
            w['delivery'].update(timed_boundary_diagnosis=summary,next=next_action)
            w['reproduction']['timed_boundary_diagnosis']=str(boundary.relative_to(ROOT))
            new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [已存计时与证据采集边界诊断](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8')
        changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    shutil.copy2(Path.home()/'.codex/automations/maskco/automation.toml',O/'AUTOMATION_BEFORE_TIMING_DIAGNOSIS.toml')
    addition='\n2026-10-12新增配送离线计时边界诊断：delivery/timed_request_timing_boundary_diagnosis_root_v1事前登记后只解析原十请求时钟、manifest及四冻结源码AST，binary64精确有理数分段，无模型/C0/solver/原QA重跑。两正常特征至encoder1.0547138000838459/1.0484482999891043秒及encoder至首评分.6643542000092566/.6546405996195972秒均含其它计算与保存；首评分至截止6.141295399982482/6.156070400029421秒缺逐步计算/IO/JIT分离钟。实际故障标记至延迟7.797150900121778/7.610975299961865秒为枚举hash差分JSON组合，8896为实际加载与注册额外树联合文件，不是导入模块数。完整链stage6后发布前全依赖/参数检查未实际到达，不能推断去IO反事实速度或挪出计时升级资格。下一先另事前封存分阶段钟/有界证据与完整计时设计，不重跑旧10/100/220及任何完成QA、不放宽8秒/预热旧样本；新execution_authorized=False，第2族FAIL封闭、第3族未启动。正式已训同初始化CE/同任务随机/各控制/等耗时迭代/独立强动态/科学全依赖/完整预算仍缺。取货权威终局64/callback80/pending空无81。新报告配送_已存计时与证据采集耗时边界诊断_2026-10-12.md及16入口WORK_STATUS先存旧字节更新，reproduction仅新TIMED_BOUNDARY_DIAGNOSIS_EVIDENCE_20261012.json；原FAIL/ledger/13800/31/183/C1保持，无服务器/GPU/共享MD/最终种子使用。早先本地进程只读Access Denied不作无进程断言；路径只读拼写失误原观察保存。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立，maskco ACTIVE30分钟；诊断后准备/常规未变静默。'
    create(O/'AUTOMATION_ADDITION.json',{'addition':addition})
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(entries=16,report_sha256=sha(report),boundary_sha256=sha(boundary),identity_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),all_registered_inputs_unchanged=all(sha(ROOT/p)==h for p,h in reg['inputs'].items())))
    print('Recorded offline diagnosis; 16 prior entry bytes preserved.')
if __name__=='__main__':main()
