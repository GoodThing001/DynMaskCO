"""Correct mutable current claims after byte-preserving archive; no science edits."""
from pathlib import Path
import hashlib,json,datetime

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
OLD=E/'results/research_20261010'
NEW=E/'results/research_20261010_continuation_v2'
STAMP='2026-10-10 二十轮独立核查更正'
REPORT='docs/当前规划/决策与审计/二十轮新增工作_独立核查与继续研究_2026-10-10.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepend(p,text):
    old=p.read_text(encoding='utf-8')
    if STAMP in old:raise RuntimeError('Already corrected: '+str(p))
    p.write_text(text+'\n\n'+old,encoding='utf-8')

def main():
    # Only these three mutable current records are revised; originals verified first.
    for name in ('PROGRAM_CLOSEOUT.md','ROUND_20261010_GUIDE_DOSE.md','WORK_STATUS.json'):
        if (OLD/name).read_bytes()!=(NEW/'pre_correction_archive'/name).read_bytes():
            raise RuntimeError('Pre-correction byte archive mismatch: '+name)
    replay=json.loads((NEW/'INDEPENDENT_DESTROY_EQ8_REPLAY.json').read_text())
    if replay['all_qualification_checks_pass'] is not True:raise RuntimeError('Do not report failed replay as qualified')
    checks=sorted(NEW.glob('SERVER_CHECK_*.json'))
    server=json.loads(checks[-1].read_text())
    end=server.get('independent_actual_end_identity',{})
    if end.get('n')!=478 or end.get('mismatches')!=[] or not end.get('original_start_items_unchanged'):
        raise RuntimeError('Independent actual end identity required')
    current={'schema':'current-research-status-v2','updated':datetime.datetime.now().astimezone().isoformat(),
        'parent_program_sha256':sha(NEW/'PROGRAM.json'),'historical_rounds_preserved':20,'new_cycle_cap_per_line':12,
        'stable_online_gain_established':False,'MaskCO_attribution_established':False,
        'pickup_impossibility_proven':False,'learning_side_closed':False,'equivalence_established':False,
        'delivery':{'stage':'destroy_eq8_complete_independently_replayed_negative_signal',
                    'instances':32,'candidate_replays':768,'final_replays':96,'audit_checks':8424,
                    'actual_end_items':478,'actual_end_mismatches':0,'mean_costs':replay['mean_costs'],
                    'paired':replay['paired'],'actual_decode_calls':replay['actual_decode_calls'],'timeouts':replay['timeouts'],
                    'next':'task-specific full masked reconstruction and objective-aligned training engineering; new protocol before research data'},
        'pickup':{'stage':'actual_midday_causal_suffix_engineering_in_progress','stable_positive':False,
                  'budget_mixed_arm_corrected':{'uncond_usage':.9036071502168543,'uncond_rejected':109.8},
                  'next':'resume parity, hard10s submission barrier, future-mutation invariance and whole-fleet full-J replay before new finite online comparison'},
        'reproduction':{'stage':'headline_consistency_and_scientific_support_separated','headline_checks':29,'headline_failures':0,
                        'old_original_package_modified':False,'historical_source_training_closure_complete':False},
        'automation':'maskco-i3 ACTIVE each30min','server_shared_MD_uploaded':False}
    with (NEW/'WORK_STATUS.json').open('x',encoding='utf-8') as f:json.dump(current,f,ensure_ascii=False,indent=2,allow_nan=False)
    close='''# 三线二十轮结题陈述：独立核查更正版

> **2026-10-10 二十轮独立核查更正**。旧版完整原字节保存在[更正前封存](../research_20261010_continuation_v2/pre_correction_archive/PROGRAM_CLOSEOUT.md)。原实验数据、协议、权重、源码与负裁决不覆盖；24节实验史仍保留。下文是当前有效解释。

**稳定且可归因的 MaskCO 核心在线增益仍未建立。取货问题没有被证明无可改善策略；学习方向也没有被穷尽。** 用户已授权核查后继续研究，新有界批次登记于[PROGRAM v2](../research_20261010_continuation_v2/PROGRAM.json)，原20轮预算保留、不重新计数。

| 方向 | 已有证据 | 当前准确结论 |
|---|---|---|
| 配送 | ANY50第三32快照队列对LNS +2.0855%，描述95%CI[0.1962%,3.9125%]，此前两队列同向；第三队列原路线/完整物理轨迹缺失。 | 非学习快照基线线索。前两项含事后比较，GLS区间跨零、重用开发数据；未达到总控三全新40完整日/确认家族alpha/强动态对照和学习归因资格。 |
| 取货 | 距离枚举、参考策略拒绝计数、贪心全信息比较、外生预算扫描。 | 每项仅约束自己的动作域和运行点；服务数、距离与另一配置的目标分解不能拼成当前J上界。可行贪心是最大化目标的下界。 |
| 负结果 | 29项headline一致、8项学习引用重读；原R1/C1负结果保留。 | 数字引用一致不等于完整研究复现或科学主张资格；源码/训练/原轨迹缺口逐项保留。 |

取货预算摘要实际混用了最后rolling臂：现重新读取uncond预算利用率0.9036071502、拒单109.8。旧结果不改，用独立核查工件记录更正；平均能耗外推不能称已证上界。

ANY50优势的92.09%距离占比来自另一次12例开发子集运行，并非三个确认队列的总体机制。经典related代码为简化空间聚类移除，未完整实现文献Shaw过程；不能据其失败推出学习掩码无信号。相同10秒上限是合法部署比较，额外匹配尝试次数和解码消耗回答机制归因，不能称原上限预算造假。

新配送D-DESTROY-EQ8-v1已于14:14:27完成：三臂各256次实际解码，32新实例/768候选/96最终路线、manifest、成本和完整轨迹独立重放8424检查全过；服务器478项实际结束身份独立核对一致，三臂0超时。均值Pguided87.041、Rguided83.043、Pdiscard83.388；P相对丢弃指导−4.839%[−10.364%,0.146%]，P相对随机模型−5.508%[−11.847%,0.282%]。未建立该删除指导角色的收益；CI跨零不表示等价，也不排除新任务训练或完整掩码生成。该批不重复、不补样本、不换seed挑均值。

取货新原型在真实非零时刻保存当前cargo/thermal/承诺行程，并以同合法在线续程核完整J；首工程状态差值为零，两臂hard。它仍处工程验收，不是稳定正向。配送继续任务专属完整masked encoder+decoder和目标对齐控制的隔离实现。负结果线完善结构化当前主张与证据身份门，保持原183文件和历史负结果。

稳定确认仍要求冻结候选后使用三个全新40完整日队列，分别胜同合同强滚动与归因控制、独立重放身份/轨迹/硬可行/超时，并按预声明确认家族alpha控制多轮筛选。原取货最终20260927/28禁用。不保证阳性，不因结果不显著重跑或扩大样本。

详见[完整审计与论文依据](../../docs/当前规划/决策与审计/二十轮新增工作_独立核查与继续研究_2026-10-10.md)、[新批当前状态](../research_20261010_continuation_v2/WORK_STATUS.json)和[本批独立重放](../research_20261010_continuation_v2/INDEPENDENT_DESTROY_EQ8_REPLAY.json)。新隔离实验与本地文档继续；此前服务器共享MD同步和公开发布尚无批准。
'''
    (OLD/'PROGRAM_CLOSEOUT.md').write_text(close,encoding='utf-8')
    prepend(OLD/'ROUND_20261010_GUIDE_DOSE.md',f'> **{STAMP}**：以下24节为历史研究过程，原字节另已封存。有关“稳定已确认”“取货无策略空间”“学习侧关闭”“GLS等价”“0.095%效用上界”和92.09%总体机制的旧解释由[更正结题](PROGRAM_CLOSEOUT.md)及[独立核查](../../{REPORT})覆盖。原数字、失败、试验协议与尝试均保留；当前新批状态见[WORK_STATUS v2](../research_20261010_continuation_v2/WORK_STATUS.json)。')
    old=json.loads((OLD/'WORK_STATUS.json').read_text(encoding='utf-8'))
    new_status={'schema':'historical-batch-with-current-correction-v2','current_authoritative':current,
                'historical_status_not_current':old,'historical_bytes_archive':'../research_20261010_continuation_v2/pre_correction_archive/WORK_STATUS.json'}
    (OLD/'WORK_STATUS.json').write_text(json.dumps(new_status,ensure_ascii=False,indent=2),encoding='utf-8')
    banner=f'> **{STAMP}（当前有效）**：撤回“取货不可改善”和“学习侧已关闭”的泛化。ANY50保留为非学习快照线索，稳定MaskCO在线增益尚未建立。新D-DESTROY-EQ832实例/768候选/96路线独立重放8424检查通过、478实际结束身份一致，但两项归因门未过，0超时；不重复该批。取货真实日中后缀原型与任务专属配送masked训练工程继续，负结果支持门区分数字一致与科学资格。原20轮封存，新有界批次与30分钟跟进ACTIVE；文档仅本地。[完整更正与进度]({REPORT})。'
    prepend(E/'项目当前状态.md',banner)
    prepend(E/'00_项目导航.md',banner)
    for rel in ['docs/当前规划/结果与进度/执行进度表.md',
                'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',
                'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']:
        prepend(E/rel,f'> **{STAMP}**：新批配送删除指导归因32实例已完成且独立重放/身份通过，未建立收益；取货日中因果后缀工程和配送任务训练骨架继续，支持等级核验器在完善。原20轮不重置，稳定在线目标未建立。详见[更正审计](../决策与审计/二十轮新增工作_独立核查与继续研究_2026-10-10.md)及`results/research_20261010_continuation_v2/WORK_STATUS.json`。')
    for direction in ['01_取货线','02_配送线','03_负结果线']:
        for filename in ['README.md','实验结果索引.md','文档索引.md']:
            prepend(E/direction/filename,f'> **{STAMP}**：当前状态按[二十轮独立核查与继续研究](../{REPORT})和[新批工件](../results/research_20261010_continuation_v2/WORK_STATUS.json)维护；下方早期研究快照保留。稳定MaskCO在线正向未建立，原负裁决不改。')
    audit=E/REPORT
    with audit.open('a',encoding='utf-8') as f:f.write('''

## 本批实际交付与后续屏障

配送新批D-DESTROY-EQ8-v1于14:14:27正常完成，QA/pilot/pipeline退出均0。完整下载441文件，32新实例逐个再生；768候选与96最终路线的manifest、品质、能耗、成本及原轨迹8424独立检查全过；实际结束478源码/权重/协议项独立重算一致。三臂各256解码/0超时，均值成本Pguided87.041248、Rguided83.042754、Pdiscard83.387755，平均用时3.832/3.873/4.151秒。固定机会与实际解码调用匹配不代表消耗秒数相等。

Pguided相对Pdiscard−4.8393%[−10.3637%,0.1462%]，相对Rguided−5.5077%[−11.8467%,0.2825%]；固定2000配对bootstrap独立一致，两门未过。没有任何异常导致日志前缀丢失，现有完整批仍有效；未来版本另增加逐尝试持久化，不回写或重复本批。完整logits数组未保存，所以本地重放验证其输入/输出SHA和mask/路由持久化，但不冒称已独立重算指导排序；后续补只读模型推理身份核查。

取货有限公开初始状态工程有同C0-v2/B的可行J见证+2.26274，但为合成3单配置，不是当前I3分布或在线方法收益。真实日中原型首状态J差0，两臂hard；仍须无干预续程一致、其他车能耗完整、承诺不可改、未来信息变化不影响选路、完整10秒认证提交和同预算新开发面板。负结果支持门24项初版回归通过，当前进一步分离历史/更正主张、禁止最终seed及检查子集范围。

新增全文依据：[Ropke与Pisinger作者技术报告2004](https://di.ku.dk/forskning/Publikationer/tekniske_rapporter/tekniske-rapporter-2004/04-13.pdf)§2.1.1/算法2给出随机锚点、相关性排序、随机秩与逐步删除；§2.1.2明确包含随机删除；§2.2包含贪心与regret插入。该全文支持“不能把nearest-k空间聚类当完整Shaw、不能称LNS文献不使用随机”的更正。它是2004技术报告，不冒称出版版全文；DTU另一技术报告本轮打开失败。50%掩码、8机会、10秒与确认样本/alpha均为本项目注册设计，文献不保证其最优或有收益。
''')
    inventory={p.relative_to(E).as_posix():sha(p) for p in [OLD/'PROGRAM_CLOSEOUT.md',OLD/'ROUND_20261010_GUIDE_DOSE.md',OLD/'WORK_STATUS.json',NEW/'WORK_STATUS.json',E/'项目当前状态.md',E/'00_项目导航.md',audit]}
    with (NEW/'CURRENT_DOCUMENT_CORRECTION_SHA256.json').open('x',encoding='utf-8') as f:json.dump(inventory,f,ensure_ascii=False,indent=2)
    print(json.dumps({'current_records_corrected':True,'original_experiment_results_overwritten':False,'local_only':True}))

if __name__=='__main__':main()
