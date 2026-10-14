"""Register the newly authorized research program; never edits old protocols."""
import hashlib, json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
out=root/'results/research_20261010'
out.mkdir(parents=True,exist_ok=True)
program={
 'version':'three-line-research-20261010-v1',
 'authorization':{'date':'2026-10-10','user_instruction':'继续完成这三个方向，不断修正完善，直到有稳定的正向增益','scope':'new documented hypotheses, isolated implementations, finite registered server experiments and independent confirmations'},
 'prior_failures_immutable':['i3_online_recovery_1009_v1','delivery_pilot_1008_v2','negative_result_paper'],
 'objective':'stable attributable MaskCO online utility on pickup and delivery; accurate reproducible scientific reporting on the negative-result line',
 'not_guaranteed':'No positive outcome is promised; exhausted hypotheses and negative results remain evidence.',
 'search':{'cycles_per_line':12,'hypothesis_fixed_within_experiment':True,'development_feedback_allowed':True,'old_failed_chain_repeats':False,'new_protocol_before_data':True,'new_result_directory_each_attempt':True,'all_attempts_retained':True,'no_success_conditioned_resampling':True},
 'validation':{'old_pickup_final_seeds_forbidden':[20260927,20260928], 'selection':'Select at most one frozen candidate per registered confirmation family using development only; never tune on confirmation', 'stable_online':'Three fresh 40-complete-day cohorts, every cohort passes main and attribution gates, all hard, complete independently checked source/weight/data/contract/budget identity and raw trajectory replay.', 'pickup_main':'I3-derived new method minus cond_hist >=20 utility/day and paired CI lower>0; minus same-init reconstruction-only control mean>0 and CI lower>0. Also compare qualified same-contract strong rolling controller before stronger efficiency claims.', 'delivery_main':'Against each registered random/nonlearned and qualified strong rolling control, mean relative complete-cost reduction>=1% and paired CI lower>0; same online information, legal actions and total budget.', 'alpha_spending':'For confirmation family j=1,2,... allocate two-sided alpha_total=0.05/(j*(j+1)), split equally across pickup and delivery and Bonferroni across registered contrasts and cohorts within each line; exploratory pilot 95% CIs are descriptive only.', 'statistical_method':'Independent complete days or snapshots, paired percentile bootstrap with predeclared counts/seed; Monte Carlo tail precision checked before confirmation. No formal discovery if sample/coverage assumptions are unsupported.', 'sample_size':'First mechanism pilot is bounded exploratory n=32, no power claim. Dynamic n=40/cohort is a resource boundary; plan detectable-effect/precision before confirming, no interim-significance expansion.'},
 'delivery_first_pilot':{'id':'D-EQ8-20261010-v1','scope':'exploratory snapshot attribution only; not same-time competitiveness or online gain','dataset_seed':2026101002,'instance_ids':list(range(32)),'mask_schedule_seed':2026101041,'arm_order_seed':2026101099,'bootstrap_seed':2026101077,'bootstrap_replicates':2000,'random_model_seed':42,'arms':['pretrained','random','nonlearned'],'attempts':8,'mask_fraction':0.3,'guide_weight':1.0,'all_incomplete_attempts_keep_incumbent':True,'fixed_schedule_no_resample':True,'endpoint':8,'secondary_endpoint':1,'mean_relative_attribution_threshold':0.01,'confirmation_not_triggered_by_pilot_alone':True},
 'execution':{'current_gpu_policy':'No team GPU until fresh availability audit; CPU-only initial engineering/mechanism tasks','bounded_jobs':'tmux for >5min; explicit wall timeout, own PID tree only; never kill team processes','max_single_job_s':28800,'server_shared_md_upload':'still pending explicit separate approval, do not retry','publication':'not authorized','physical_scope':'literature-based simulation; no own physical-calibration claim or universal calibration barrier'},
 'current_stage':'hypothesis-bound parallel implementation and engineering qualification; no stable-positive claim'
}
with (out/'PROGRAM.json').open('x',encoding='utf-8') as f:json.dump(program,f,ensure_ascii=False,indent=2,allow_nan=False)
sha=hashlib.sha256((out/'PROGRAM.json').read_bytes()).hexdigest()
with (out/'PROGRAM.sha256').open('x',encoding='utf-8') as f:f.write(sha+'\n')
tree='C:/Users/mmc/.agents/skills/arbor/scripts/tree.py'
for line in ('pickup','delivery','reproduction'):
    args=['python',tree,'--run-dir',str(out/line)]
    subprocess.run(args+['observe'],check=True,capture_output=True)
    subprocess.run(args+['cycle'],check=True,capture_output=True)
    subprocess.run(args+['set-status','--node','n1','--status','running'],check=True,capture_output=True)
doc=root/'docs/当前规划/决策与审计/三线持续研究_新授权与验证准则_2026-10-10.md'
text='''# 三线持续研究：新授权与验证准则

2026-10-10。用户明确要求继续三方向研究、持续修正至有稳定正向。本条开启新的研究阶段，覆盖旧“无获准余项、暂停”在未来工作范围上的含义；不更改已结案实验、失败主门或历史论文数值。成功是研究目标，不保证实验结果。

取货既有主差−144.680、配送既有单步主门失败均保留。新方法和新数据有单独版本及预声明，所有成功与失败都进入假设树。负结果线改善复现闭包、证据质量和当前论文，并记录后续新证据；不能将原C1失败改写为正向或等价。

## 本轮实际工作

| 方向 | 第一项具体工作 | 当前证据边界 |
|---|---|---|
| 取货 | 检查模型缺失车辆/资源状态、教师与部署预算差异；实现资源观测/预算匹配诊断，再设计新训练 | 工程诊断不等于新模型或在线收益；旧权重不重训覆盖 |
| 配送 | 新P/R/H固定8次日程，32全新快照的探索性归因pilot；随后完善动态合同、优化器与强滚动控制 | 快照归因不等于在线或同墙钟竞争；已有动态事件认证器尚不是优化器 |
| 负结果 | 建立新研究证据到论文主张的拒绝式验收，独立复现新工件 | 历史训练缺件仍如实保留，不公共发布/投稿 |

完整预声明与初始种子在[PROGRAM.json](../../../results/research_20261010/PROGRAM.json)。配送首批种子2026101002、掩码2026101041、顺序2026101099、bootstrap2026101077，随机网络42；先工程验收、封存全部源及权重，再生成32例。固定8次、固定删除比例0.3、三臂同掩码和修复/认证；超时或不可完成尝试保留incumbent并全部报告。32例是资源限定的探索，不声称充分功效。不得因pilot通过直接称稳定或直接开全量。

## 稳定正向怎样确认

开发数据允许指导新假设；确认数据只验冻结后候选。每个确认家族最多从开发阶段选一个候选，不在确认结果上调参。用三组全新40完整日队列，同信息、合法动作和总预算，各组都满足主门和归因门，原始轨迹、源码/权重/合同/数据、超时与硬可行独立复核。每个正式动态协议还须冻结模型初始化与训练种子重复、基线适配、样本精度、具体数据种子和统计实现后才启动；这些尚未注册完成，不把本总纲冒充可直接运行的动态协议。

取货保留实用阈值20效用/日及对纯重建的正归因；配送采用对随机、非学习和已合格强滚动控制均至少1%平均成本降低及CI下界正。三组确认、40日和阈值是项目设计，不是论文推定常数。持续搜索采用家族alpha花费0.05/[j(j+1)]，再按方向/比较/队列分配；仅探索性的95%区间不能升级为多轮搜索的确认性结论。确认前检查bootstrap尾部模拟精度与独立样本的精度限制，拒绝把重复抽样数当样本量。

最终取货20260927/28继续隔离。确认失败封存该候选，后续改变另开假设与新确认家族，不重测同一确认集。不同任务绝对效用不能排名；迭代对单次提高不能替代学习归因。

## 文献依据与推导边界

[MaskCO正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf)支持保留解上下文、掩码恢复和迭代；P/R/H同日程控制是针对本项目D2混杂的设计推导，原论文不确定本项目8次或32例。

[Ross等，AISTATS2011](https://proceedings.mlr.press/v15/ross11a.html)解释策略改变观测分布时单一历史策略训练的局限，支持检查新策略访问状态；不证明本项目资源输入一定增益，也不自动使新系统成为DAgger。

[Agarwal等，NeurIPS2021](https://arxiv.org/abs/2108.13264)支持报告少量运行下的统计不确定性与重复差异；本项目配对完整日、独立确认和alpha花费是设计决定，不能称为论文已验证的覆盖保证。

[Baty等，动态VRPTW学习与组合优化](https://arxiv.org/abs/2304.00789)提供派车学习与组合求解分工的依据。其协议不能直接充当本项目有限多温车队、返仓复用与10秒下已合格的强动态基线；必须适配并验收。

既有三线72项文献/实际依据表、商品级动力学修订继续有效，商品/温区/外推和单位均保留来源；文献仿真不冒称真实设备校准。

## 持续执行边界

每条假设树首批12个协调周期，各实验有明确样本和最长运行时间；阶段结束依据新证据继续修正，而非为正均值无限重跑。常规计算安静；正式裁决、异常、显著方法变更和阶段完成通知。只启动验收后冻结的新任务，先核实际进程和资源。原上游、对比原仓库、冻结旧树/权重/183份论文原包不改。

新工程包/协议JSON/实验产物可按用户授权在服务器隔离执行。此前待批共享MD/历史镜像上传不重试；本地文档继续维护。公开发布、DOI、投稿不在本轮范围。
'''
with doc.open('x',encoding='utf-8') as f:f.write(text)
notice='> **2026-10-10 新授权：三线持续研究已登记**：用户要求继续修正完善直至稳定正向；新假设、新协议、新目录及独立确认允许推进，既有P/D失败保持。三线首项隔离实现进行中，尚无新在线正向。旧暂停仅表示上一有限链结案。[新授权与验证准则](docs/当前规划/决策与审计/三线持续研究_新授权与验证准则_2026-10-10.md)。服务器共享MD上传仍待此前批准。\n\n'
p=root/'项目当前状态.md';old=p.read_text(encoding='utf-8');p.write_text(notice+old,encoding='utf-8')
p=root/'00_项目导航.md';old=p.read_text(encoding='utf-8');p.write_text(notice+old,encoding='utf-8')
print(json.dumps({'program_sha256':sha,'document':str(doc),'status':'parallel_implementation_registered'},ensure_ascii=False))
