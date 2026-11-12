# 方法范围与目录约定

现行范围已按用户随后“约十个对比方法并下载开源代码”的要求扩至10项，新增AttentionModel（ICLR 2019）、POMO（NeurIPS 2020）、CaDA（ICML 2025）。9套作者源码加1个自编VNS，基础MASKCO另列参照。来源与SHA见configs/SOURCE_PINS.json，执行任务见[给执行端Codex的完整指令](执行端Codex_十方法复现任务.txt)。旧七项整理说明及v1验证Excel保留原阶段身份；新增方法仍待冷链目标与约束适配。

2026-10-21用户明确选择保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。其余旧方法仅保留在独立历史恢复树，移出当前活动范围；这不是按测试结果挑选或否定这些方法的学术价值。

当前模型是静态多车品质净效益，旧动态取货/接拒协议及成绩不直接复用。7项保留不等于7项已完成同目标适配：ALNS/VNS为开发基线，OR-Tools/PyVRP为距离策略控制，三项神经候选待适配与训练。

本轮对比方法调研与验证总表见[Excel总表](../results/validation_baselines_20261021/MASKCO_对比方法_数据集与指标总表_v1.xlsx)及[逐例可复算结果](../results/validation_baselines_20261021/comparison.json)。主对比确定为同目标基础MASKCO、ALNS和VNS，本项目D模块仍为改进候选；RouteFinder、DeepACO、CaR保留后续适配资格，不把原CVRP/TSP等任务的论文成绩写入冷链净效益表。主数据仍为原56例中统一纳入的47例、四组几何，主指标为净效益U，同时报告等价损失总成本和硬约束可行率。

2026-10-21首轮仅使用第0折c205—c208、种子1000、主条件。ALNS/VNS各4例60秒单CPU新运行；基础MASKCO与D复用dev11注册的严格60秒结果并逐历史可行解重算。四例平均净效益分别为基础276311.62、D候选276309.08、ALNS276228.98、VNS276028.98 U。D仍低于基础2.55 U，未建立改进优势。批次与共享负载不同，只作开发参考，正式测试未启用。

OR-Tools 9.8.3296与PyVRP 0.11.3的距离控制各4例均因品质违规不能获得本问题有效分数。PyVRP原接口失败4次、隔离0.14.0安装因Python版本不符失败的证据均保留，兼容入口另在run03完成；不宣称运行的是活动0.14.0源码commit构建。共20次原生尝试、8份复用神经结果、1560个历史可行解事件已审计，72项冻结不变。

新增数据放data、模型放models、运行入口放scripts、配置放configs、说明放docs、实跑输出放results、权重放checkpoints。第三方原代码不修改，原许可证随源码保留。

[完整整理报告](../../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)记录原路径、现路径、备份、核验和已知边界。原历史manifest不改写，恢复时需恢复原工作区相对位置。
