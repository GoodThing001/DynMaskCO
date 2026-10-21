# 科研复核与冻结

2026-10-21，模型、数据及分析合同通过复核；没有运行新的方法比较或神经训练。

- [科研审查与冻结决定](../../../docs/static_delivery_research_20261021/科研冻结审查_v1.md)
- [审查摘要](audit_summary.json)：94条合法路线独立LP最优排程、39条品质不可行、九例原路由不可行、141组经济倍率固定路线LP、47例等价目标恒等式及评分边界检查。
- [独立LP记录](independent_lp_checks.json)、[经济权重排程记录](economic_weight_schedule_checks.json)：只验证排程/评分，不是权重敏感性算法效果实验。
- [封存清单](FREEZE_LOCK.json)、[清单SHA](FREEZE_LOCK.sha256)：包含原始客户文件、原文、模型、分析计划、严格入口及审查代码。范围不包括尚不存在的方法实现和训练检查点。

在扩展根目录运行`python -B tools/audit_and_seal_publication_protocol_v1.py`可验证封存。已有封存时不会重写；文件改变即失败，需另开版本而非自动盖章。
