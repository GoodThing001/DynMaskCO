"""Publish local engineering qualification and preserve each former entry."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=R/'delivery/multistep_insertable_teacher_engineering_root_v2';a=json.loads((d/'INDEPENDENT_MULTISTEP_REPLAY.json').read_text());assert a['failed']==0 and a['checks_count']==9670
    p=R/'pickup/actual_masked_fleet_followup_root_v2/TELEMETRY_CORRECTION.json';assert json.loads(p.read_text())['full_10_second_transaction_qualified'] is False
    report=E/'02_配送线/文档/结果与进度/配送_多步插入教师工程验收与环境闭包_2026-10-11.md'
    note='''# 配送多步插入教师工程验收与环境闭包（2026-10-11）

本轮只验证两份已观察工程快照的接口与身份，未生成新科学实例、训练模型或重跑DEV16。稳定、可归因MaskCO核心在线增益仍未建立。

## 固定范围与实际执行

在900000、900001两份旧快照上，各用预先登记的四次50%订单掩码（seed=2026101141）。每一步从实际非学习distance-regret2部署解码器的上一步完整C0可行改进继续；教师标签不决定下一步状态。共保留8状态，其中6份初始路线已不同于第一个状态，以及944份教师候选。全部状态不筛掉、不重采；仅以完整C0可行且改善决定部署状态转移。父和worker实际CPU7，每请求含教师、部署解码、认证、传输总上限10秒，教师GLS搜索上限3秒；实际响应3.263–3.334秒。

教师仍是距离提议、完整C0选择的有限启发式，未证明最优。每个标签保持原mask所保留边、未掩码订单的原车及相对顺序。独立验收重新生成掩码、逐候选核边与骨架、完整物理标志与四项成本、重播择优和实际部署状态，9670项0失败。它共享既有C0物理求值器，不能称独立物理校准。

## 工程修正与保留的失败

前一轮v3模块提取解时压缩空车槽位，后续泛化可能改变车辆编号。本轮独占副本保留全部车槽，旧模块未改；本次8状态未出现未掩码骨架空车，故不能据此声称空车场景已通过运行验收。

v1完整保留：8状态、926候选；源码无变，但起始清单枚举后metadata版本查询懒加载email/parser，导致四个文件缺起始身份，独立验收9522项中2项失败。故v1环境闭包不合格。以有证据的工程缺陷恢复v2，先完成metadata/platform查询再枚举源清单；科学参数、旧快照、掩码和状态转移不变。GLS为墙钟启发式，恢复候选数不同不用于择均值。

v2实际Python3.12.14、NumPy2.5.3、ORTools9.11.4210、protobuf5.26.1；2016项源、已缓存字节码、依赖包、解释器及随附DLL开始/结束SHA全部一致，无未封存的新增加载文件。该闭包限本工程执行，不包含操作系统、未来JAX训练或服务器部署。新科学实验仍须另行登记其实际环境。

## 文献依据与适用边界

[ORTools官方路由文档](https://developers.google.com/optimization/routing/routing_tasks)支持从已有路线初始化以及有界搜索；本文的3秒、10秒、50%和四步是工程检查参数，文献不保证这些值最优。[NeuRewriter原论文](https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf)将改写动作与实际解质量变化联系，支持检查教师动作与部署动作是否一致；并不证明当前训练目标会产生增益。

本轮只覆盖非学习部署形成的8状态，未覆盖学习模型全部状态分布，也没有新增全状态理想logit可达性证明。此前两个部分掩码理想边分数探针维持有限存在性范围。下一步是完整训练/JAX/服务器环境封存、匹配多步数据和部署角色的预声明新假设，以及同初始化重建控制和已训练随机初始化控制；在新数据前完成登记，不选择checkpoint、不改旧FAIL或未过门的确认入口。

## 原件

根research/review_3080_20261010/delivery/multistep_insertable_teacher_engineering_root_v2保存ENGINEERING_REGISTRATION.json、STATES.jsonl、WORKER_SOURCE_START/END.json、RESULT.json及INDEPENDENT_MULTISTEP_REPLAY.json。v1的CLOSURE_ENUMERATION_FAILURE.json保存恢复原因。所有共享MD仅本地维护，未上传服务器。
'''
    with report.open('x',encoding='utf-8') as f:f.write(note)
    pickup=E/'01_取货线/文档/结果与进度/取货_整请求硬预算日志更正_2026-10-11.md'
    pnote='''# 取货整请求硬预算日志更正（2026-10-11）

旧actual_masked_fleet_followup_v1/hard_admission.py日志声明all_in_hard_worker=True，但实际父进程承担公开投影、序列化、完整认证、状态恢复及现场安装；末尾超时回滚不能抢占上述运算。撤回此字段作为完整10秒事务资格的证据，所有旧源与日志原字节保留。此前184项原型验收仍只证明共同K10候选评分及故障回退，不升级为正式五臂在线资格。

独占actual_masked_fleet_followup_root_v2副本仅修正日志为False并显式记录worker覆盖范围及事务不合格；通过语法检查，非日志行为未改，尚未运行或宣称完整事务恢复。TELEMETRY_CORRECTION.json绑定两版实际SHA并说明旧日志解释。

后续须将昂贵阶段纳入可中止流程并核原子安装、同初始化重建控制和完整接拒事件轨迹，在工程独立验收后另登记新鲜有限日。S3禁用、最终留出种子禁用和物理仿真标签均保持。此次纠正没有效用收益。
'''
    with pickup.open('x',encoding='utf-8') as f:f.write(pnote)
    entries=[E/x['path'] for x in json.loads((R/'current_entries_before_finite_teacher_oracle_closure_20261011/UPDATE_IDENTITY.json').read_text())['files']]
    backup=R/'current_entries_before_multistep_teacher_telemetry_20261011';backup.mkdir(exist_ok=False);manifest=[]
    section='''

## 2026-10-11 新增：多步教师工程闭合与取货预算日志更正

配送两份旧快照×四个实际非学习部署状态，共8状态/944教师候选；2016项实际环境文件开始结束一致，独立9670项0失败。v1缺失metadata懒加载文件的起始身份，8状态/926候选及失败原件完整保留，v2同参数工程恢复。此阶段无新训练、新科学日或在线收益，下一步另封存训练/服务器闭包并登记新假设。详见`02_配送线/文档/结果与进度/配送_多步插入教师工程验收与环境闭包_2026-10-11.md`。

取货旧硬预算日志的all_in_hard_worker=True不符合父公开投影/认证/安装的实际覆盖，已用独占副本更正，原字节保留；尚未通过完整10秒事务运行验收。详见`01_取货线/文档/结果与进度/取货_整请求硬预算日志更正_2026-10-11.md`。负结果31项必要门及183文件历史封存沿用结案资格，未重复整理/实验。三线均不因此得到稳定MaskCO核心在线增益；自动任务保持ACTIVE，所有共享MD仅本地更新。
'''
    summary='\n2026-10-11新增：配送8个多步旧状态/944候选、2016项环境开始结束封存，独立9670项0失败；取货全事务硬预算日志已纠正但运行资格未闭合。新训练与确认未启动，稳定MaskCO在线增益未建立。详见两线文档索引和三线交接最新节。\n'
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(multistep_engineering_states=8,multistep_teacher_candidates=944,multistep_engineering_checks=9670,multistep_engineering_failed=0,multistep_loaded_environment_files=2016,multistep_engineering_directory=str(d.relative_to(ROOT)),multistep_engineering_identity=sha(d/'INDEPENDENT_MULTISTEP_REPLAY.json'),all_model_rollout_states_qualified=False,scientific_training_closure_qualified=False,next='register complete training/server environment and new matched deployment-state scientific family before fresh data; do not rerun DEV16 or old QA');s['pickup'].update(whole_worker_telemetry_claim_withdrawn=True,whole_worker_telemetry_receipt=sha(p),whole_worker_telemetry_successor='research/review_3080_20261010/pickup/actual_masked_fleet_followup_root_v2/hard_admission.py',atomic_all_in_budget_qualified=False)
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        elif f==DOC:new=old+section
        else:new=old+summary
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    with (backup/'UPDATE_IDENTITY.json').open('x') as f:json.dump({'files':manifest,'new_reports':[str(report.relative_to(E)),str(pickup.relative_to(E))],'server_MD_uploaded':False,'old_results_modified':False},f,indent=2)
    print(json.dumps({'entries':len(manifest),'new_reports':2,'stable_online_gain':False}))
if __name__=='__main__':main()
