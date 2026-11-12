# 静态冷链对比方法工作区

用户于2026-10-21进一步要求约10项对比方法，当前清单为ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR、AttentionModel、POMO、CaDA。原始MASKCO基础任务适配另作直接参照，本项目改进模型不计入这10项。一级子目录按职责分类，各方法位于models内。

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 56份作者100客户原实例的同字节镜像、47例纳入身份与几何划分 |
| [models](models/README.md) | 9套固定作者源码及1个自定义VNS实现入口 |
| [scripts](scripts/README.md) | 统一经典基线运行入口、目录与SHA核验入口 |
| [configs](configs/README.md) | 已有模型/分析配置镜像、方法清单与固定源码来源 |
| [docs](docs/README.md) | 保留范围、方法资格与目录规则 |
| [results](results/README.md) | 原始实验结果入口；新运行使用独立子目录 |
| [checkpoints](checkpoints/README.md) | 与当前候选有关的已有预训练权重、原路径和SHA |

最新[十方法开发验证v2](results/ten_methods_20261021/run01/README.md)已交付：ALNS/VNS与六项真实作者网络适配各完成第0折四例独立验证；六种神经方法仅训练38例一轮，CaDA存在退化，尚不能称为收敛后的论文复现。OR-Tools/PyVRP仍属品质违规的距离控制。基础MASKCO/D只回放历史记录，正式测试保持关闭。当前MASKCO在[主算法目录](../C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/README.md)，不重复复制。交给执行端的任务见[执行端Codex指令](docs/执行端Codex_十方法复现任务.txt)，本地完整性检查运行`python -B CC_Compare/scripts/verify_layout.py`。

原有20项适用性审查是历史候选池，当前活动名单以[methods.json](configs/methods.json)为准。旧动态方法、适配器、历史结果和MAPT内嵌Git仓库已整体移出活动目录；原字节和未跟踪文件均可恢复，见[迁移报告](../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)。

本次只调整目录与入口；现行训练、评价、远程任务和历史manifest继续使用原冻结路径，不重写已有实验身份。作者源码按原字节保存；原包中的非当前数据和无关权重不放入活动模型树，完整来源仍保存在原证据快照。
