# MaskCO 静态多车冷链优化工作区

当前主线：静态、单仓、多车、全部客户必须服务，以品质折算交付价值减距离费和制冷费的总净效益最大化。科研模型与分析计划已冻结，算法仍在训练/验证开发阶段；正式测试资格与最新实验状态以[项目当前状态](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md)为准。

先从[工作区导航](工作区导航.md)找文件，再按[文档版本索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)判断资料所属阶段。

| 目录 | 内容与使用方式 |
|---|---|
| [C-VRP_Cold-chainVehicleRoutingProblem](C-VRP_Cold-chainVehicleRoutingProblem/README.md) | 本项目扩展代码、数据、配置、文档、结果与归档；日常开发入口 |
| [MASKCO_code](MASKCO_code) | 原作者MASKCO骨干，保持只读 |
| [CC_Compare](CC_Compare/README.md) | 已确认7项对比方法，按data/models/scripts等职责分类；作者源码保持只读 |
| research、results、tools | 根目录原位保留的受保护历史材料、用户原结果与安全检查工具；不在这里新增算法实验 |
| .vscode及其他点目录 | 编辑器、连接与代理配置，保留原位置 |

常用入口：

- [算法方法、训练策略与参数](C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/README.md)
- [当前模型与算法阶段](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_research_20261021/算法阶段_v0.md)
- [结果分类与证据入口](C-VRP_Cold-chainVehicleRoutingProblem/results/README.md)
- [工具分类与运行路径](C-VRP_Cold-chainVehicleRoutingProblem/tools/README.md)
- [历史阶段与恢复材料](C-VRP_Cold-chainVehicleRoutingProblem/archive/README.md)
- [本次目录整理记录](C-VRP_Cold-chainVehicleRoutingProblem/results/workspace_organization_20261021/README.md)

旧RRP v3、动态取货及负结果保留各自目标、参数、权重和原始结论；它们的结果不充当当前净效益算法成绩。工程约束见[AGENTS.md](AGENTS.md)，发布排除范围见[GITHUB_SYNC.md](GITHUB_SYNC.md)。
