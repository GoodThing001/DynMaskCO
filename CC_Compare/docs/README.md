# 方法范围与目录约定

2026-10-21用户明确选择保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。其余旧方法仅保留在独立历史恢复树，移出当前活动范围；这不是按测试结果挑选或否定这些方法的学术价值。

当前模型是静态多车品质净效益，旧动态取货/接拒协议及成绩不直接复用。7项保留不等于7项已完成同目标适配：ALNS/VNS为开发基线，OR-Tools/PyVRP为距离策略控制，三项神经候选待适配与训练。

新增数据放data、模型放models、运行入口放scripts、配置放configs、说明放docs、实跑输出放results、权重放checkpoints。第三方原代码不修改，原许可证随源码保留。

[完整整理报告](../../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)记录原路径、现路径、备份、核验和已知边界。原历史manifest不改写，恢复时需恢复原工作区相对位置。
