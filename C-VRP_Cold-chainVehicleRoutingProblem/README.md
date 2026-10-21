# 静态多车冷链净效益MASKCO扩展

当前研究目标是全部车辆合计的品质折算交付价值减距离和制冷费用。模型、数据与分析计划已封存；方法选择、训练与验证继续开发，正式测试资格以[项目当前状态](项目当前状态.md)为准。

从[项目导航](00_项目导航.md)进入具体模块，研究记录的版本身份见[全项目文档索引](docs/文档索引与版本状态.md)。

| 分类 | 目录与职责 |
|---|---|
| 运行代码 | [scripts](scripts/README.md)：当前static_profit、旧RRP static_delivery、测试及project_paths路径契约 |
| 配套工具 | [tools](tools/README.md)：运行辅助、冻结工具、资料获取/分析/核验与维护 |
| 配置 | [configs](configs/README.md)：当前冻结模型、分析计划、算法候选与历史参数 |
| 数据 | [data](data/README.md)：作者100客户数据、原Solomon、旧RRP与发表案例 |
| 对比方法 | [CC_Compare](../CC_Compare/README.md)：已确认7项，按data/models/scripts等职责分类 |
| 来源与适配证据 | [comparisons/static_coldchain](comparisons/static_coldchain/README.md)：独立适配器和原始快照，保持已有路径 |
| 研究文档 | [docs版本索引](docs/文档索引与版本状态.md)：模型、方法、论文证据和各阶段边界 |
| 输出结果 | [results](results/README.md)：实验开发、协议预检、冻结审计、历史结果与目录整理记录 |
| 历史保全 | [archive](archive/README.md)：取货/负结果、旧冻结包、原件备份与恢复说明 |
| 原位保护 | research、scripts/simulation等残留：受保护历史材料，保持原位置，不作为当前运行入口 |

当前命令入口为[scripts/static_profit/run.py](scripts/static_profit/run.py)，训练策略见[算法README](scripts/static_profit/README.md)。旧RRP入口[scripts/static_delivery/run_rrp.py](scripts/static_delivery/run_rrp.py)保留原v3/33托盘口径。

目录整理只改变允许迁移的工具和中间文件位置，不改变模型、原数据、实验数值或历史结论。[整理清单与核验](results/workspace_organization_20261021/README.md)。
