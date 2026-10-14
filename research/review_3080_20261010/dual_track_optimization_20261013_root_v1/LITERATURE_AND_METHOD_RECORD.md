# 取货与配送共同迭代：论文依据与方法记录

记录时间：2026-10-13。用户授权两线共同推进、每轮增加实质优化，并将参考论文写入记录。下列是有界方法调研，不是系统综述。原论文的实验收益不转移为本项目收益；工程假设树不增加科学族计数，也不授权第3族开始。

| 文献与原始来源 | 实际阅读范围 | 对当前缺口的启发与采用状态 | 本项目必须补齐的条件 |
|---|---|---|---|
| André Hottung、Kevin Tierney，2020，**Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem**，ECAI 2020，DOI 10.3233/FAIA200124。[会议原文](https://ecai2020.eu/papers/786_paper.pdf)，[arXiv 1911.09539](https://arxiv.org/abs/1911.09539)。 | 原文摘要、引言、§3破坏/修复流程及训练描述；不是全文复现。 | 配送/取货的 MaskCO 应在不完整当前解上学习修复，并在迭代中评价结果。不能仅训练完整无向邻接后假定插入动作一致。现有有方向动作组件可承接修复接口，但尚无正式已训控制。 | 固定承诺与未掩码顺序、当前状态可行插入、全C0、学习与非学习修复独立消融、完整等耗时迭代。论文静态容量约束不能替代冷链和现场事件。 |
| André Hottung、Paula Wong-Chung、Kevin Tierney，2025，**Neural Deconstruction Search for Vehicle Routing Problems**，本记录引用 arXiv:2501.03715v1 预印本；没有核定另一个出版版本。[原文](https://arxiv.org/html/2501.03715v1)。 | §3.1–3.3的移除策略、训练、greedy insertion与网络职责。 | 候选新方法：让 MaskCO 学习移除客户与再插入顺序，保留确定性可行修复；减少把所有车型/位置逐次交给重头评分的职责。当前只是待工程验证候选，未训练、未选择该方法、未登记科学新族。 | 需要与现有修复职责分开消融；不能依据旧教师成本选择掩码；保持同初始化已训CE和同任务随机控制，固定最终步与新鲜数据前登记。 |
| Stéphane Ross、Geoffrey Gordon、J. Andrew Bagnell，2011，**A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**，AISTATS，PMLR 15:627–635。[出版页面](https://proceedings.mlr.press/v15/ross11a.html)，[原文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)。 | 摘要、§3和Algorithm 3.1，以及保证依赖条件；没有复现其任务。 | 候选新方法：在实际学习策略产生的状态上获取动作监督并聚合，检验H轨迹训练与学习部署分布错位。当前H多步教师不属于DAgger；本轮不生成该类科学数据。 | 先锁定动作表示、实际训练/部署源、有限轮次和监督预算；不沿用论文的“validation选最佳”去挑本项目checkpoint，也不把其理论保证解释为本冷链收益保证。 |
| Russell W. Bent、Pascal Van Hentenryck，2004，**Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers**，Operations Research 52(6):977–987，DOI 10.1287/opre.1040.0124。[出版社](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124)。 | 出版社摘要及完整书目身份；索引中的Brown作者PDF打开返回404，未声称读到全文。 | 取货采用其“显式未来场景与当前在线决策”思路，新增独立接受/拒绝分支滚动接拒组件。我们的配对SAA效用选择不等于原MSA的consensus distinguished-plan算法；不得标成逐项论文复现。 | 同一无条件历史场景、独立接拒入口、双方真实滚动补救、不可见未来隔离、承诺不变、全C0。旧strong_rolling仅路线模式，不能占独立强动态接拒角色。 |
| André Hottung、Yeong-Dae Kwon、Kevin Tierney，2022，**Efficient Active Search for Combinatorial Optimization Problems**，ICLR 2022；arXiv首版2021。[会议原文](https://openreview.net/pdf?id=nO5caZwFwYu)，[arXiv](https://arxiv.org/abs/2106.05126)。 | arXiv摘要与会议原文标题/出版身份；没有全文算法复现。 | 候选新方法：只更新较小参数子集进行实例适应，作为未来等耗时搜索研究选项。当前10秒部署与正式训练屏障尚缺，暂不执行在线更新。 | 在线梯度/优化器/保存/取消均计时；实例适应与离线学习必须分别归因，不能让部分参数更新绕过同初始化控制或数据前登记。 |

工程参考：[JAX官方FAQ：jit changes the exact numerics of outputs](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)。实际读取该节：XLA可能改写算术求值，导致不同浮点结果。它支持检查中间算术的理由，不定位本项目具体编译pass，也不授权放宽旧逐位门。

## 本轮固定工程任务

- 配送n1：仅首个已保存 `pretrained_initial/step0` 输入新增逐算子观测，区分原eager、带观测的编译图以及旧无观测编译输出；不重跑原34候选或backbone/C0。观测本身可能改变图，须明确证据限度；在新源中预初始化完整依赖后再封存起始身份。
- 取货n2：在原输入顺序首个合格已封存公共状态上，实现具有自己接拒决策的接受/拒绝滚动补救比较。不能以旧route-mode共享接拒入口冒充独立控制，不重复旧无模型完整日。
- 两组件均事前工程登记、实际执行、独立检查后才记录资格；不在论文记录中先写PASS。产物与失败记录位于各自新独占目录，最终结果另追加，原文件不覆盖。

## 预先确定的科学与停止边界

本轮不进入科学第3族、不训练、不触碰最终取货20260927/28、不重跑旧DEV/确认。第2族FAIL封闭；每线最多12个新科学族。只有正式学习控制、独立强动态接拒、完整冷链/硬约束、等耗时掩码重构迭代、科学全依赖与完整请求预算满足，才另登记新鲜研究数据与校正确认。

持续目标是稳定、可归因核心在线正向增益，不能保证数值正向。预登记投入边界或无法消除的外部阻塞到达时如实报告；不以无限换种子、调头、挑checkpoint或重复旧QA寻找正均值。
