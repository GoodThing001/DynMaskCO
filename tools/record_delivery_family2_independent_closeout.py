"""Archive family2 registered failure with successful raw replay, no model gain claim."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';OUT=D/'multistep_insertable_family2_root_v1';A=D/'multistep_insertable_family2_data_audit_root_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    inspect=sorted(A.glob('inspection_*'))[-1];audit=read(inspect/'INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json');reg=read(OUT/'RESEARCH_REGISTRATION.json');prod=read(OUT/'downloaded/RESULT.json')
    assert audit['checks_count']==96248 and audit['failed']==36 and audit['raw_replay_failure_names']==[]
    assert audit['counts']['states']==256 and audit['counts']['candidates']==24758 and len(audit['registered_barrier_failures'])==36
    assert prod['failed']==35 and read(OUT/'downloaded/RUN_EXIT.json')['exit_code']==1 and read(inspect/'RUN_EXIT.json')['exit_code']==1
    assert not audit['scientific_teacher_data_qualified'] and not audit['actual_training_qualified'] and not audit['training_execution_enabled']
    start,end=read(inspect/'AUDIT_SOURCE_START.json'),read(inspect/'AUDIT_SOURCE_END.json');assert start['files']==end['files'] and not end['changed'] and not end['new_files']
    assert sha(A/'audit.py')==read(A/'AUDIT_REGISTRATION.json')['program_sha256']==sha(ROOT/'tools/audit_delivery_multistep_family2_data.py')
    for n,h in read(A/'AUDIT_REGISTRATION.json')['inputs'].items():assert sha(OUT/'downloaded'/n)==h,n
    assert sha(OUT/'RESEARCH_REGISTRATION.json')=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176'
    old=D/'multistep_insertable_family2_data_audit_root_v1';assert 'NoneType' in read(old/'downloaded/AUDIT_FAILURE.json')['traceback'] and read(old/'downloaded/RUN_EXIT.json')['exit_code']==1
    mechanism=audit['posthoc_descriptive_mismatch_diagnosis'];assert mechanism=={'exact_route_false':34,'same_adjacency_different_route':4,'different_adjacency':30,'oracle_better_fullC0_than_teacher':3,'oracle_equal_fullC0_to_teacher':0,'oracle_worse_fullC0_than_teacher':30,'oracle_incomplete':1,'H_incomplete':1}
    report=E/'02_配送线/文档/结果与进度/配送_第2族多步教师正式屏障失败与机制诊断_2026-10-11.md'
    text='''# 配送第2族多步教师：正式屏障失败与独立机制诊断（2026-10-11）

第2族不放行训练。64新实例的256多步状态与24,758教师候选全部保留，9,819生产检查35FAIL、原退出1；新进程独立完成96,248检查，其中36项注册屏障失败，原始数据/完整C0/动作/轨迹/实际身份核查失败为0。独立退出1是保留屏障失败的结果，不是将原FAIL改写为PASS。未启动三臂128步、DEV或确认，没有稳定可归因MaskCO在线正向。

## 登记、全部原件与资格

注册SHA `562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176`，第2族/每线12族上限，在新数据前固定。科学范围与数字见[前声明](../协议与审计/配送_多步可插入教师第2族_科学预声明_2026-10-11.md)。原目录research/review_3080_20261010/delivery/multistep_insertable_family2_root_v1；服务器独占路径省略delivery。完整STATES超过1GiB，12项原文件以有界readv下载，清单绑定精确SHA，没有重复生成教师或解。生产总895.376秒；其中256请求均在合作式10秒内，教师3秒搜索。它不等全请求可抢占部署预算资格。

35个生产失败为34固定二值理想输入不能精确重现教师路线，以及(59,2)一项H插入提议未完成。此尝试与剩余订单保存，原程序采用原计划回退并继续下一状态。没有修改策略、删状态、重采或把不完整尝试补成成功。本项是插入提议失败，不是最终原计划漏服务；独立核对回退状态与后续轨迹成立。完整C0教师目标全部hard；256目标偏好标签均正，仅证明此受限教师搜索相对各H状态找到低成本目标，不证明模型学会或在线收益。

独立审计按实际开始请求重建64固定数据种子、四步掩码、每步H驱动状态，逐一验证24,758完整候选、未掩码原车与严格顺序、原保留边、完整C0成本与严格择优、所有插入事件、报告/剩余订单及回退、最终教师与理想输入原件。共享原C0物理评价器，不等另一物理实现。原生产源/输入在审计前后再次读取实际字节；审计自身8,790文件开始结束一致，无变更/新增。raw_replay_failure_names=[]，全部原始正确性核查通过。

96,248总项中的36失败为原生产屏障FAIL一项、34理想目标精确重现条件和一项H完整提议条件。不能从总体“有36失败”误写原件不可复算，也不能从原件全部可复算误写科学数据已合格。依据冻结的全部目标精确恢复等条件，scientific_teacher_data_qualified=False、训练仍关闭。原规则不放宽。

## 具体机制与可陈述范围

34个不精确状态中，4个原教师与二值探针路线的无向邻接相同、完整C0成本不同；其余30个邻接不同。33个完整探针解中3个成本更低、30个更高，另1个探针不完整。所有这些完整成本差已被独立重新计算，原始逐状态值保存在审计summaries与本地只读事后算术记录。该分解属于事后描述，不是新门或因果消融。

四个同邻接/异成本路线对直接说明：对于这些固定输入，单一无向邻接不是完整冷链成本的充分表示。原CVRP距离目标对某些方向变化可以不敏感，冷链完整成本仍依赖执行语义；当前目标将方向/车槽信息折叠，不能从同一目标邻接区分这些路线。若两个候选邻接完全相同，当前以邻接差乘log概率的偏好分数对它们恒为0，CE邻接目标也相同。这是当前监督目标的具体信息缺口，不是所有模型、输入或取货策略不可能改善的证明，也不能在未分解原路线前把四项全部归因某一种方向变化。

另外，三项探针实际比受3秒限时的教师更低成本，说明“精确复现教师”与“成本更好”不同；教师不是已证明的最优解。尽管本族精确重现屏障前瞻固定得保守，当前不能根据看到的三项改善删掉或改写它。有限二值探针不精确也不证明任意连续logits不可达。无论是否发生这些情况，本族没有模型训练和开发部署测量，不能当作MaskCO效用负结论或正向证据。

下一研究候选是匹配实际插入决策的方向/车辆/位置动作监督，或明确保留方向/车槽语义的目标。须先在已观察工件上核查哪种执行语义造成别名、验证生产解码器动作与标签；然后另立数据前登记的有限新族和独立控制，不能直接改本族教师、guide、训练权重、掩码、预算或门槛后重跑。此为候选假设，不是新科学族已启动。

MaskCO作者说明其参考解掩码恢复和迭代重构范式；这种范式不替本项目解决有方向的冷链执行目标。此处仅使用实际可读取的[作者官方实现说明](https://github.com/Thinklab-SJTU/MaskCO)，OpenReview全文本次仍被验证页拦截。策略改变状态分布的问题由[DAgger原论文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)支持；本族只用H轨迹，不是DAgger，也不借用其性能保证。数值设置为项目选择，物理参数仍pilot而非已实测校准。

## 审计自身缺陷与恢复

初版独立审计v1完整读入大文件，处理(59,2)时把合法保留的None报告当成完整报告索引，真实TypeError退出1，186.347秒。源/注册/原失败/退出全部保留，未改在跑树。核本人监督及worker实际已退出后，在v2新目录绑定相同12项科学输入SHA，改成逐状态流式读取并显式核不完整提议、剩余订单和回退；205.692秒完成上述96,248项。没有重跑教师、求解器或训练，没有更改原FAIL或科学条件。全部实际恢复输入上传及原科学输入再次哈希通过。

## 三线后续边界

取货已观察154联合提交/76拒单沿用；仍缺真正现场公开提取、可中止父传输、正式学习与同初始化CE、独立强动态接拒及全日10秒事务，未宣称新收益。负结果只追加本族正式屏障失败与目标别名范围，原183文件、C1联合FAIL、CI跨零非等价、源码/训练/完整复现缺口不改。第1族DEV16失败及原I3/D2不重跑。共享MD只维护本地，没有上传服务器或公开发布。
'''
    create(report,text)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':sha(OUT/'RESEARCH_REGISTRATION.json'),'independent_audit_sha256':sha(inspect/'INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json'),'checks':96248,'registered_barrier_failures':36,'raw_replay_failures':0,'independent_source_files':8790,'original_producer_failures':35,'original_run_exit':1,'data_qualified_under_registered_rule':False,'target_adjacency_alias_pairs':4,'posthoc_mismatch_counts':mechanism,'not_global_impossibility':True,'no_scientific_training_or_DEV':True,'stable_online_gain':False,'shared_MD_uploaded':False}
    create(OUT/'FAMILY2_INDEPENDENT_CLOSEOUT.json',receipt);create(R/'reproduction/FAMILY2_INDEPENDENT_EVIDENCE_BOUNDARY_20261011.json',receipt)
    manifest=read(R/'current_entries_before_family2_collection_complete_20261011/UPDATE_IDENTITY.json');backup=R/'current_entries_before_family2_independent_closeout_20261011';backup.mkdir(exist_ok=False);files=[]
    intro='\n> 2026-10-11配送第2族独立结案：256状态/24,758候选完成96,248项复算，原始数据/轨迹/完整C0/身份失败0，36项原注册屏障失败仍保留；三臂训练/DEV未启动。4组同无向邻接但异完整成本，属于当前目标信息缺口，不是全局不可改善。初版审计None报告异常保存，v2同原字节恢复完成。稳定MaskCO在线正向仍无，取货现场缺口/负结果183封存沿用。\n'
    for entry in manifest['files']:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);oldtext=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(oldtext);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(current_science_phase='family2_registered_barrier_FAIL_independently_diagnosed_closed',current_science_independent_data_qualified=False,current_family2_raw_replay_qualified=True,current_family2_independent_checks=96248,current_family2_independent_registered_failures=36,current_family2_raw_replay_failures=0,current_family2_independent_source_files=8790,current_family2_independent_identity=sha(inspect/'INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json'),current_family2_source_closed=True,current_family2_target_alias_pairs=4,current_family2_diagnosis=mechanism,current_family2_independent_audit_directory=str(A.relative_to(ROOT)),current_family2_v1_audit_exception_preserved=True,current_science_training_enabled=False,current_science_development_enabled=False,current_science_confirmation_enabled=False,next='Family2 closed under original registered barrier FAIL; no repeat collection/audit/search/train/DEV. Use already observed evidence to distinguish direction/vehicle/action alias and prepare executable action target engineering; new finite family only after prospective registration; pickup full-live request gaps continue.')
            s['reproduction']['family2_independent_boundary_receipt']='research/review_3080_20261010/reproduction/FAMILY2_INDEPENDENT_EVIDENCE_BOUNDARY_20261011.json';newtext=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            newtext=intro+oldtext if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else oldtext+intro
            if f.parent==E/'02_配送线':newtext+='\n- [第2族正式屏障失败与独立机制诊断](文档/结果与进度/'+report.name+')\n'
        f.write_text(newtext,encoding='utf-8');files.append({'path':entry['path'],'before_sha256':sha(saved),'after_sha256':sha(f)})
    create(backup/'UPDATE_IDENTITY.json',{'files':files,'new_report':str(report.relative_to(E)),'shared_MD_uploaded':False,'original_scientific_FAIL_and_registration_preserved':True})
    for f in files:assert sha(E/f['path'])==f['after_sha256'] and sha(backup/f['path'])==f['before_sha256']
    print(json.dumps({'independent_checks':96248,'raw_replay_failures':0,'registered_barrier_failures':36,'target_alias_pairs':4,'entries_updated':len(files),'stable_gain':False}))
if __name__=='__main__':main()
