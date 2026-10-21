# 对比方法目录整理记录

日期：2026-10-21。用户已明确确认7项保留范围，并要求CC_Compare的下一层按职责分类。

```text
CC_Compare/
├─ data/          原100客户数据镜像与几何划分
├─ models/        ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR
├─ scripts/       统一经典基线运行与核验入口
├─ configs/       方法资格、源码commit与配置镜像
├─ docs/          范围和科学边界
├─ results/       原结果入口与新运行
└─ checkpoints/   9份相关权重与SHA
```

旧目录29,506份文件、约7.75GiB已完整移入[恢复树](../../archive/cc_compare_reorganization_20261021/README.md)。活动树准备时5,360份、约244.63MiB。原件仍在本地，因此体积变化指活动目录，不表示释放了同等磁盘空间。

非选定方法和旧动态适配退出活动树。MAPT内嵌Git历史及原未跟踪文件也完整保留；没有永久删除原件，没有使用git clean/reset、自动暂存、提交或推送。现行算法、评分、原始数据、冻结文件、服务器运行和历史manifest保持原路径与字节。

官方源码按固定来源复制；PyVRP采用实际运行对应的0.14.0。相关权重集中于checkpoints：RouteFinder 4份、DeepACO CVRP100 2份、CaR 100客户CVRP/CVRPBLTW 3份。无关数据与权重不进入活动模型树，完整原快照仍保留。VNS指向唯一现行实现，不另复制算法。

7项保留不等于7项已完成完整目标比较：ALNS/VNS是同目标开发适配，OR-Tools/PyVRP是距离路由控制，另外3项为待适配候选。原20项审查只作历史候选池，不按效果改写结论。整理中检测到研究进展文档并行更新，已保留其最新内容再追加目录说明。

核验结果：通过。机器回执为[verification.json](verification.json)。

- 29,506件旧树原件全量SHA一致，MAPT内嵌Git的HEAD和未跟踪状态一致。
- 5,340份复制文件逐件SHA一致；活动目录恰为7类、models内恰为7项。
- 72项科研封存、两个旧恢复ZIP、173份原始资料与19份算法文件核验通过。
- 统一入口CLI帮助加载通过；ALNS/VNS在训练例r101上各0.05秒/最多1轮的功能检查完成，两份完整路线再次独立审计通过，不能据此作方法效果结论。[入口回执](entrypoint_verification.json)。
- Windows长路径已通过扩展路径处理，最长301字符的原件可读取并核验；不改系统设置。
本地[迁移清单](migration_manifest.json)和[原树SHA清单](legacy_inventory.json)不作为GitHub可下载大件发布。

入口：[CC_Compare/README](../../../CC_Compare/README.md)。工作区根目录运行`python -B CC_Compare/scripts/verify_layout.py`，加--full-archive可核验全部旧树原件。运行入口为`python -B CC_Compare/scripts/run_classical.py --help`，继续使用原训练/验证角色门槛。
