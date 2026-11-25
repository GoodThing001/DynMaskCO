# 项目结构与阅读代码

## 代码目录承担什么职责

项目根目录为 `D:/PyCharm_/MASKCO-Main`。读代码时，按“原模型—任务适配—锁版求解—实验结果”四层查找，比按文件修改时间查找更容易建立完整理解。

| 路径 | 主要职责 | 阅读时关注的问题 |
|---|---|---|
| `MASKCO_code/` | 作者原始 MaskCO 网络与模块 | 节点表示、注意力、邻接偏置、输出头如何计算 |
| `C-VRP_Cold-chainVehicleRoutingProblem/` | 静态冷链业务模型、训练、计时与评价 | 坐标和时间怎样变成品质、费用、U与可行性 |
| `CC_Compare/` | 对比方法来源、原代码和配置登记 | 对照来自哪篇论文、哪个版本，如何区分原模型与适配器 |
| `research/paper_preparation_20261023/bottleneck_plan/` | 各项研究方法、锁版源码、分批实验和分析 | 哪个结果对应哪个方法、面板和计算预算 |
| `…/maskco_perishable_external_20261119/reserved_validation/` | ORDER_SPLIT 的锁版配置、求解器副本和历史13法结果 | 当前完整方法的确切配方与代码身份 |
| `…/recent5_20261120/` | 新增5方法及同期6法评价 | 新方法原论文、公开权重、冷链适配与比较结果 |
| `…/supplement_sync_20261120/` | 历史同期4法复核与固定车队价格敏感性数据 | 热图增量、强基线差额和计算代价 |
| `…/unified_external_20261120/` | 统一16法、62例、前瞻三来源及机制与界限分析 | 最新主结果、数据映射、原版消融、轨迹和可控费用 |
| `…/maskco_followup_dev_20261120/startup/` | 冻结模型编译复用与首个60秒交付 | 准备开销、数值一致性、服务时限 |
| `…/model_documentation_20261121/` | 本手册与新日期汇报稿源文件及输出 | 业务公式、术语、结果与实现的完整阅读入口 |

省略号 `…` 在上述表格中均指 `research/paper_preparation_20261023/bottleneck_plan`。历史目录中还保留早期取货、多车RRP和其他重组候选；这些目录便于追溯，不改变本手册的静态配送目标和主模型定义。

## 建议按什么顺序读源码

1. **先读业务参数与路线评价。** [publication_model_v1.json](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json) 保存单位、容量、温度、品质和经济参数；[publication_model_v1.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py) 给出路线时刻和效益计算。对照手册中的两客户演算，逐项确认输入到 U 的关系。
2. **再读独立复算。** [preflight_publication_model_v1.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/preflight_publication_model_v1.py) 使用独立的时间和约束计算核对路线。理解它之后，能区分“求解器声称可行”与“路线被重新验证可行”。
3. **读网络与训练目标。** [profit_neural.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py) 是冷链适配入口；[CVRPModel.py](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/CVRPModel.py)、[TSPModel.py](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/TSPModel.py) 和 `MASKCO_code/modules/` 是主干模块。输入特征、完整教师邻接、掩码邻接和输出 H 是阅读重点。
4. **读锁版搜索主循环。** [pool_search.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py) 组织外层时段、前向、算子试选和当前状态；[masked_repair.cpp](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp) 执行密集的破坏修复与路线评价。Python 负责调度，C++ 负责频繁调用的搜索操作。
5. **读跨车与分车。** [block_split.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/block_split.py) 提出路线组和访问顺序；[split_order.cpp](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/split_order.cpp) 按固定顺序求最优连续分段。
6. **阅读最新实验统计。** [unified_report.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/unified_report.py)组织16法逐案例、逐几何、逐来源配对；[three_sources.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/three_sources.py)组合荷比、XTW和RWVRP的前瞻来源均值与描述区间；[economic_bound.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/economic_bound.py)构造与冷链U同单位的指派松弛上界。
7. **历史统计入口。** [sync_report.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/sync_report.py) 把终点转换为逐案例、逐几何和逐来源统计，并生成同期四法报告。可用它核对平均方式、配对差额和描述区间的具体含义。

## 怎样从结论找到原始结果

最新入口是 [统一论文实验总览](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/论文实验总览.md)。具体终点分三个目录：[既有面板benchmark01](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/benchmark01)、[前瞻与压力prospective01](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/prospective01)、[商超retail01](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/retail01)。各目录的 `RESULTS.md` 给可读表格，`endpoints.csv` 是一个方法×案例×种子的终点，`paired.csv` 保存配对差额，`analysis.json` 保存几何簇、种子与来源统计，`independent_audit.json` 保存独立重算。

例如，核对“前瞻热图增量+12.11 U”，先读 [three_sources.json](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/three_sources.json)，再回到三个来源中ORDER_SPLIT与ORDER_SPLIT_D的同区组配对。荷比10日均值−0.07 U、新XTW8几何+35.29 U、RWVRP8城+1.09 U，三来源各取1/3得到总体。区间[−3.04,+32.74] U来自来源内整组重采样，不从显示到两位小数的表格反推。若要解释单个城市或几何的差异，应继续比较其品质损失、里程和制冷费用。

新数据的身份从 [new_panel/heldout_sources.json](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/new_panel/heldout_sources.json) 与 [retail_source/heldout_sources.json](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/retail_source/heldout_sources.json)追至作者原件。`selection.json`记录确定性选取、坐标重叠与投影误差，`heldout_admission.json`记录单客和共同初解可行性。62个案例的原始输入与方法结果应按批次读取，主XTW与旧低重叠XTW共享7例，不重复计数。

| 要解释的问题 | 直接阅读入口 |
|---|---|
| 原版MASKCO到完整方法的变化 | [原版四臂结果](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/origin01/RESULTS.md)与[机制消融说明](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/原始MASKCO与机制消融.md) |
| 随时间和搜索种子的变化 | [trajectory01](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/trajectory01/RESULTS.md)，`paired_seeds.csv`与`cold_start.csv` |
| 编译复用的首次请求代价 | [startup结果](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_followup_dev_20261120/startup/RESULTS.md) |
| 净效益距离最优可能还有多少 | [bounds结果](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/bounds/RESULTS.md)，`case_bounds.csv`与`source_gaps.csv` |
| 原方案价格变化的经济解释 | [固定路线敏感性](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/economic_sensitivity/RESULTS.md) |
| 实际城市胜负及集中程度 | [RWVRP逐城结果](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120/results/retail01/CITY_RESULTS.md) |

历史证据入口仍包括[reserve02十三法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/RESULTS.md)、[recent5六法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/results/benchmark01/RESULTS.md)、[sync01四法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/results/sync01/RESULTS.md)。同名 `benchmark01` 分属不同研究目录：recent5是历史六法，unified_external是最新十六法，引用时必须保留完整目录身份。
