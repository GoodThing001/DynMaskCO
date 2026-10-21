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

作者源码字节不修改。权重集中于checkpoints，当前比较数据集中于data；原作者包内的小型测试fixture保持其源码结构。移出的原基准数据与无关权重见[资产排除记录](../configs/OMITTED_VENDOR_ASSETS.json)，完整原快照仍在扩展证据目录。直接运行作者脚本时需显式指定集中后的数据/权重路径，尚无净效益适配的候选不提供虚假的统一求解接口。
