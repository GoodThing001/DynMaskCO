# 当前静态冷链物流对比方法源码与适配

目录归类：[当前7项CC_Compare](../../../CC_Compare/README.md)为活动浏览入口。本目录保留评分适配器、运行依赖及8份原始来源快照，保护历史SHA与已有载荷路径。HGS及另一PyVRP版本仅作原始证据，不算当前活动方法。[整理记录](../../results/cc_compare_organization_20261021/README.md)。

文档版本：[全项目文档索引](../../docs/文档索引与版本状态.md)。

2026-10-21当前算法入口：[训练与参数README](../../scripts/static_profit/README.md)、[对比基线与迭代](../../docs/static_delivery_research_20261021/对比基线与迭代.md)。完整净效益目标下，已实跑MASKCO任务适配、官方ALNS框架适配和自编VNS；四例验证集24份路线已独立重算并封存为[开发基线v0](../../results/static_delivery/algorithm_baselines/v0_dev04_validation/comparison.json)。[dev09阶段结果](../../docs/static_delivery_research_20261021/dev09阶段结果.md)：四模型训练完成，但总任务超时退出124，23份验证路线已重算，9项待dev10补全；不得将部分结果当完整实验。基础模型已冻结，正式训练配置和方法检查点尚未冻结。

| 方法 | 当前使用资格 |
|---|---|
| MASKCO基础版 | 原骨干的静态净效益任务适配，从头训练；保留原始来源与新增适配身份 |
| ALNS | 固定官方框架＋本项目同目标算子；不是某篇冷链论文全部细节的精确复现 |
| VNS | 本项目同目标实现；保留邻域、种子与实际源码 |
| OR-Tools / PyVRP旧Solomon运行 | 车辆数/距离目标，不能继承为当前同目标成绩 |
| HGS / RouteFinder / DeepACO / CaR | 源码已保存，尚未满足当前目标及完整约束适配与核验资格 |

以下为历史阶段记录，涉及“未训练”“尚未迁移”的表述描述当时状态，当前状态以上述入口为准。

完成更新：[dev09＋dev10完整结果](../../docs/static_delivery_research_20261021/dev09_dev10完整结果.md)已核验32份验证路线并封存为v1；保留原超时与负结果，正式方法/检查点仍未冻结。

**正式科研冻结入口**：[审查决定](../../docs/static_delivery_research_20261021/科研冻结审查_v1.md)、[严格评分适配器](adapters/publication_model_v1_strict.py)、[分析计划](../../configs/static_delivery/publication_experiment_freeze_v1.json)。严格入口保留v1合法数值，修复NaN出库被误判可行及正式评分关闭品质检查的风险。旧预检/源码SHA不改。完整目标ALNS、VNS及MaskCO对比须先适配并验证、封存方法配置，再正式测试；历史成绩不能直接继承。

当前论文协议入口：[组合模型与数据尺度v1](../../docs/static_delivery_research_20261021/组合模型与数据尺度_v1.md)、[新增评价器](adapters/publication_model_v1.py)。原数据100客户、20吨、24h，恒温275K零阶品质/80%阈值、虚拟U最大净效益及可择时出库已冻结；47例可行、9例有排除证据，独立重算与几何四折名单见[预检](../../results/static_delivery/publication_protocol_v1_preflight_20261021/README.md)。旧ALNS情景脚本尚未迁移到该目标，不能将旧成绩直接当作新协议对比。未新增神经训练。

新增[同商品价格/单位模块](adapters/direct_value_evidence.py)与[证据裁定](../../docs/static_delivery_research_20261020/direct_value_evidence/README.md)：9组报价核算、6项检查，阻止TA冒充原文指数和USD/EUR混算。完整食品价值路由配置仍未闭合；既有ALNS结果保留为假设研究。

2026-10-20。用户要求为当前静态任务新建独立源码目录。本目录与旧CC_Compare分开；旧动态适配器、旧成绩和资格不直接继承。研究目标为最大单次配送净效益。**现已实现独立情景评价器与ALNS适配，并完成72次明确假设下的搜索**，见[最新结果及边界](../../docs/static_delivery_research_20261020/net_benefit_scenario/README.md)。实际食品定价/验收及车辆物理校准仍不完整，不把情景结果称为业务利润。

- `upstream/`：7个作者/官方仓库的8份固定commit快照，保留LICENSE、源码和上游附带文件；未执行其训练/生成脚本，也未解包附带的模型/数据压缩文件。快照里的包文件包括原作者__init__.py，原样保留；本地新增适配器不建包__init__.py。
- `archives/`：原下载ZIP，本地保留。
- `SOURCE_PINS.json`：固定commit、URL、ZIP SHA、逐成员核验摘要；各仓库`SNAPSHOT_PROVENANCE.json`保留完整逐文件SHA。
- `adapters/`：本项目独立适配器与检查；不修改upstream。
- `.runtime/packages/`：pip --target隔离的运行库；版本见`RUNTIME_VERSIONS.json`。一次venv ensurepip失败的空运行目录保留，实际未使用。
- [方法适用性](方法适用性.md)、[机器分类](method_eligibility.json)：评估原17项及新增ALNS/HGS/CaR。

本轮已保存：ALNS、OR-Tools、PyVRP、HGS-CVRP、RouteFinder、DeepACO、CaR。多数为MIT，OR-Tools为Apache2.0，以各源LICENSE为准。未把所有既有大型权重再次复制进新目录。数据和论文许可与代码许可分别核定。

本次实际求解使用官方OR-Tools9.14.6206、PyVRP0.14.0 wheel，不声称已经编译本地C++源码。其源码快照固定版本另有SHA。新目录原文件不得混入旧动态补丁。

PyVRP实际运行匹配源码为`upstream/PyVRP-0.14.0/`，官方v0.14.0 commit`5d9776a954b810bdb3fe71d47b1e7de7cffe90d2`。早期按既有清单下载的`upstream/PyVRP/`项目版本为1.0.0a0，单独保留其身份，不冒称为实际执行的0.14.0源码。

从扩展根运行：

```powershell
python -B -X utf8 comparisons/static_coldchain/adapters/solomon_preliminary.py --out results/static_delivery/solomon_new_run --seconds 10
python -B -X utf8 comparisons/static_coldchain/adapters/quality_evidence_demo.py --out results/static_delivery/quality_new_run.json
python -B -X utf8 comparisons/static_coldchain/adapters/test_evidence_and_solomon.py
```

输出路径须为新路径。Solomon结果是先少车后短距离的100客户经典验证，独立双精度评价；不计算商品利润。品质程序只计算文献参数的变化量；`require_profit_inputs`在缺少价格等证据时拒绝利润计算。两者未接入既有run_rrp命令，不改变原RRP目标。

净效益适配顺序：ALNS+共同独立评价器，小例数学规划，再接MaskCO/RouteFinder/DeepACO；CaR作为复杂约束候选。相同静态数据、硬约束、目标与计算预算是资格前提。原距离模型仅事后重算净效益时，应标成迁移控制，不称同目标方法。

完整研究和[导师初步结果](../../docs/static_delivery_research_20261020/initial_results.md)另见[证据包](../../docs/static_delivery_research_20261020/README.md)。研究任务本身未提交/推送；用户随后授权统一发布，范围见[GitHub同步说明](../../../GITHUB_SYNC.md)。没有运行新神经训练或改动旧任务。

## 文献数值情景净效益

新入口`adapters/run_net_benefit_scenario.py`，配置`configs/static_delivery/net_benefit_scenario_v1.json`。`net_benefit_model.py`用于搜索；`net_benefit_audit.py`从原节点独立重算时刻、约束、质量、金额，未导入搜索成本函数。所有货币以单次配送EUR记，亦可统一除以4.50 EUR归一化；固定人员共同常数从优化排序中省略。

```powershell
python -B -X utf8 comparisons/static_coldchain/adapters/run_net_benefit_scenario.py --out results/static_delivery/net_benefit_scenario_NEW --max-vehicles 25 --warm-seconds 5 --search-seconds 5
python -B -X utf8 comparisons/static_coldchain/adapters/test_net_benefit_scenario.py
```

六类各取首个通过单客户必要条件检查的完整作者实例：C105、C205、R101、R201、RC101、RC201。C101/C201原件冲突单列，不修改或生成客户。共同OR-Tools成本初解后，原作者ALNS7.0.0框架实际从固定源码导入，扩展目录内定义两个破坏/两个修复算子，4个价值设定×3种子重新搜索。上游105成员SHA不变；`SNAPSHOT_PROVENANCE.json`的`source_executed=false`保留获取阶段身份，本轮执行事实见新结果。

当前仅有ALNS的数值情景适配；原VRPTWQ完整质量函数、作者精确算法、完整食品净效益合同和MaskCO适配均未冒称完成。报告保留原运行记录、仅设定名称的元数据更正副本和原SHA源码快照，路线及计算金额未改。
