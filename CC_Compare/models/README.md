# 方法源码与实现入口

| 方法 | 身份 |
|---|---|
| [ALNS](ALNS) | 官方固定ALNS源码；完整目标适配器仍在扩展profit_search中 |
| [VNS](VNS/IMPLEMENTATION.json) | 自定义移位/交换/反转/合并搜索；指向唯一现行实现，不复制算法代码 |
| [OR-Tools](OR-Tools) | 固定官方源码，经典路由控制，不是完整净效益求解器 |
| [PyVRP](PyVRP) | 固定0.14.0对应源码，不与另一开发版本混用 |
| [RouteFinder](RouteFinder) | 候选，需完整目标适配与验证 |
| [DeepACO](DeepACO) | 候选，需完整目标适配与验证 |
| [CaR](CaR) | 候选，需品质约束与完整目标适配 |
| [AttentionModel](AttentionModel) | ICLR 2019作者源码，经典注意力构造基线；需时间窗、品质与净效益适配 |
| [POMO](POMO) | NeurIPS 2020作者仓库；优先NEW_py_ver/CVRP，属于2021重构实现，需冷链适配 |
| [CaDA](CaDA) | ICML 2025作者源码；优先100客户版本，约束感知跨问题基线，仍需品质与净效益适配 |

作者源码字节不修改。权重集中于checkpoints，当前比较数据集中于data；原作者包内的小型测试fixture保持其源码结构。移出的原基准数据与无关权重见[资产排除记录](../configs/OMITTED_VENDOR_ASSETS.json)，完整原快照仍在扩展证据目录。直接运行作者脚本时需显式指定集中后的数据/权重路径，尚无净效益适配的候选不提供虚假的统一求解接口。

新增三项的完整ZIP保存在results/source_acquisition_20261021，每项SNAPSHOT_PROVENANCE.json记录固定commit、ZIP SHA、逐文件SHA及资产集中后的真实路径。随包38份权重集中于checkpoints/upstream_extra_20261021；本次三个ZIP没有单独数据资产，后续若获取作者原任务数据，集中位置为data/upstream_extra_20261021。README中链接的外部HuggingFace资产未假称已下载，原任务数据不进入冻结47例实验。
