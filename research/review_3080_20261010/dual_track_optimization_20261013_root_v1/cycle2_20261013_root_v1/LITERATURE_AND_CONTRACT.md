# 双线第二轮：精确调度与增量场景补救资格

2026-10-13。承接第一轮实际观察：配送带中间输出的编译图不能代表旧部署图；取货独立接拒请求的已观察场景补救阶段触及固定预算。两线分别登记新候选并做工程语义资格验收，未经结果核验不安装。工程循环2/12与每线最多12科学族分开记账。

## 新增论文与阅读范围

Niels A. Wouda、Leon Lan、Wouter Kool，2024，**PyVRP: a high-performance VRP solver package**，INFORMS Journal on Computing 36(4):943–955，DOI [10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)，arXiv [2403.13795v2](https://arxiv.org/html/2403.13795v2)，[出版论文作者机构PDF](https://research.vu.nl/ws/portalfiles/portal/388474973/wouda-et-al-2024-pyvrp-a-high-performance-vrp-solver-package.pdf)，[期刊研究软件快照](https://github.com/INFORMSJoC/2023.0055/blob/main/README.md)。实际读取HTML摘要、引言、§2.3数据约定、§4.3.2路线操作与§5.3扩展机制；打开机构PDF确认来源，未逐页阅读全文、未运行或安装其软件。

论文把性能关键部分放在C++，路线邻域可利用缓存；SWAP*结合时间窗缓存与已知劣移动早停。扩展新约束需同步数据、成本增量和缓存。对本项目的可检验启发是：先核验路线摘要/缓存/保守筛查是否保留实际动作和分支评价，再考虑加速；不能只凭距离差公式一致推断冷链评价等价。

本轮取货使用的是项目既存 `incr_eval` 候选，**不属于PyVRP实现或论文复现**，该候选也不是从论文收益推导而来。原论文主要为静态CVRP/VRPTW距离目标，其默认整数距离/时长不能替代本项目实际浮点账本、订单品质、温区能耗、动态揭示和不可撤销承诺。没有采用论文的迭代时限或基准收益。若本轮动作、分支值或完整C0不精确一致，则候选不安装；下一步才另登记只做保守不可能筛查、仍走原精确排序的方案。

沿用参考：Russell W. Bent、Pascal Van Hentenryck，2004，**Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers**，Operations Research 52(6):977–987，DOI [10.1287/opre.1040.0124](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124)。本轮再次查阅出版摘要和书目，作者网页工具不可访问，未新增全文阅读资格。本项目接受/拒绝共同K10效用比较仍是借鉴场景思想，不能称原consensus distinguished-plan算法复现。第一轮其余四篇学习/搜索参考与实际阅读范围保留在[第一轮论文记录](双线共同迭代_论文依据与本轮工程证据_2026-10-13.md)，没有借这些文献启动科学数据或选择模型。

工程参考沿用[JAX官方FAQ浮点计算说明](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)：它只解释检查独立算术边界的必要性，不证明本项目编译pass原因。配送新候选保持原操作边界并逐位核验，不以跨库诊断容差取代严格评分门。

## 放行与证据边界

本轮仅新工程候选，不训练、不进入第3族、不增加科学样本或效应估计，不重跑已完成QA。取货K10/冷链合同/接拒核算/四次补救上限和8秒执行+2秒回收不变；已完成前缀和实际退出保留。配送仅新合成头接口输入，不能把局部相等、冷编译时间或组件成功升为全MaskCO链速度与在线收益。原FAIL不覆盖，完整依赖、正式同初始化已训CE/同任务已训随机、独立强动态、等耗时掩码重构迭代及多新鲜日校正确认继续作为后续科学屏障。
