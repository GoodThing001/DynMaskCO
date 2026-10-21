# 静态冷链对比方法工作区

当前仅保留用户确认的7项：ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。一级子目录按职责分类，各方法位于models内。

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 56份作者100客户原实例的同字节镜像、47例纳入身份与几何划分 |
| [models](models/README.md) | 6套固定作者源码及1个自定义VNS实现入口 |
| [scripts](scripts/README.md) | 统一经典基线运行入口、目录与SHA核验入口 |
| [configs](configs/README.md) | 已有模型/分析配置镜像、方法清单与固定源码来源 |
| [docs](docs/README.md) | 保留范围、方法资格与目录规则 |
| [results](results/README.md) | 原始实验结果入口；新运行使用独立子目录 |
| [checkpoints](checkpoints/README.md) | 与当前候选有关的已有预训练权重、原路径和SHA |

ALNS/VNS已有同目标开发适配；OR-Tools/PyVRP仍属距离路由控制；其余三项为待适配候选。所有方法尚不能据此宣称正式优胜。当前MASKCO在[主算法目录](../C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/README.md)，不重复复制。

原有20项适用性审查是历史候选池，当前活动名单以[methods.json](configs/methods.json)为准。旧动态方法、适配器、历史结果和MAPT内嵌Git仓库已整体移出活动目录；原字节和未跟踪文件均可恢复，见[迁移报告](../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)。

本次只调整目录与入口；现行训练、评价、远程任务和历史manifest继续使用原冻结路径，不重写已有实验身份。作者源码按原字节保存；原包中的非当前数据和无关权重不放入活动模型树，完整来源仍保存在原证据快照。
