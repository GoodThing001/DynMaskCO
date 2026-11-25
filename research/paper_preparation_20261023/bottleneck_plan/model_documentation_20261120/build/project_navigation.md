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
| `…/supplement_sync_20261120/` | 同期4法复核、端到端计时及同规则消融 | 热图增量、强基线差额和计算代价 |
| `…/presentation_20261120/` | 最终20页PPT、图表与逐页渲染 | 汇报中每张图对应的实验与模型机制 |

省略号 `…` 在上述表格中均指 `research/paper_preparation_20261023/bottleneck_plan`。历史目录中还保留早期取货、多车RRP和其他重组候选；这些目录便于追溯，不改变本手册的静态配送目标和主模型定义。

## 建议按什么顺序读源码

1. **先读业务参数与路线评价。** [publication_model_v1.json](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json) 保存单位、容量、温度、品质和经济参数；[publication_model_v1.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py) 给出路线时刻和效益计算。对照手册中的两客户演算，逐项确认输入到 U 的关系。
2. **再读独立复算。** [preflight_publication_model_v1.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/preflight_publication_model_v1.py) 使用独立的时间和约束计算核对路线。理解它之后，能区分“求解器声称可行”与“路线被重新验证可行”。
3. **读网络与训练目标。** [profit_neural.py](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py) 是冷链适配入口；[CVRPModel.py](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/CVRPModel.py)、[TSPModel.py](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/TSPModel.py) 和 `MASKCO_code/modules/` 是主干模块。输入特征、完整教师邻接、掩码邻接和输出 H 是阅读重点。
4. **读锁版搜索主循环。** [pool_search.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py) 组织外层时段、前向、算子试选和当前状态；[masked_repair.cpp](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp) 执行密集的破坏修复与路线评价。Python 负责调度，C++ 负责频繁调用的搜索操作。
5. **读跨车与分车。** [block_split.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/block_split.py) 提出路线组和访问顺序；[split_order.cpp](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/split_order.cpp) 按固定顺序求最优连续分段。
6. **最后读实验统计。** [sync_report.py](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/sync_report.py) 把终点转换为逐案例、逐几何和逐来源统计，并生成同期四法报告。可用它核对平均方式、配对差额和描述区间的具体含义。

## 怎样从结论找到原始结果

报告中的汇总表来自终点记录而非图像上的人工读数。同期四法的 `endpoints.csv` 是“一个方法×一个案例×一个种子”的终点表，`paired_geometries.csv` 是几何配对差额表，`sync_summary.json` 保存按来源汇总的完整数值，`independent_audit.json` 保存复算结果。它们位于 [sync01结果目录](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/results/sync01)。

例如，若想理解“热图平均提高25.17 U”，先在同一区组配对 ORDER_SPLIT 与 ORDER_SPLIT_D，然后平均同案例的3个种子，再平均同几何案例，最后使丹麦、XTW、GH三来源等权。若想解释某个几何为何改善，返回相应终点路线，比较交付品质价值、里程费用和制冷费用，而不是仅比较车辆数。

三个主要总报告分别是 [reserve02历史13法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/RESULTS.md)、[benchmark01同期6法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/results/benchmark01/RESULTS.md) 和 [sync01同期4法](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/results/sync01/RESULTS.md)。每个报告保留自己的面板、运行数和方法身份。
