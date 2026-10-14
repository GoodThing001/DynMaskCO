# 完整 MaskCO 有界训练准备：snapshot 新假设

仅保持项目原 delivery snapshot C0 合同；与真实动态配送、旧 D2 或 D-DESTROY-EQ8 不同。当前 JSON 为 DRAFT，不含正式研究种子，不得生成 64 正式实例或启动 128 正式步。工程种子 202610110071、实例 900000/900001 已预先注册；工程仅 8 个教师状态，两个实际完整模型各 2 步 batch4。工程模型永久不可复用作科学候选。

## 教师与重建目标

每实例 50 客户，共同五条初始路线；固定 4 个 50% 客户 mask，数据生成、准备、单候选 distance-regret2 修复、完整 C0 认证与提交共享总 10 秒，各 mask 上限 2.5 秒。使用纯距离 proposal 不等于 C0 最优 teacher。保留候选原路线、掩码、插入过程、硬资格、完整交付/温度品质能耗事件和所有负/小/不可行/超时，不筛例、不换例。原初始 hard 失败留档并阻断，不在新实例补齐。

新科学设计明确：

`CE_target = argmin{C0(incumbent), C0(candidate)}`，仅限预算内完整硬可行的 pair；平局取 incumbent，失败/超时取已认证 incumbent。两臂完全相同重建目标。偏好项仍比较 candidate 与 incumbent 的原顺序，`y = sign(C0(incumbent)-C0(candidate))`；差值绝对值≤1e-8 或失败则偏好零。负样本不会被重建 CE 推向较差候选，也不会被删除。超时后重认证不使该候选转成成功教师。

## 固定完整训练

正式提案 64×4=256 状态，batch4×固定128步=512曝光，每状态恰两遍，注册 RNG 逐遍 permutation。CE-only 与 CE+0.2 preference 两臂同原 CVRP 全 encoder/decoder、同10D零扩展初始参数、同状态/掩码/日程/SGD(无momentum,1e-5)。只有配对目标系数为0或0.2。逐步实际键、教师状态 SHA、loss、全234叶有限梯度留档；固定最后一步保存重载，不选checkpoint。128、batch4、学习率和0.2均为新的项目选择，无文献最优保证。

route preference score 为客户两条正边 log-probability 均值，logistic loss 使用完整 C0 pair 的正负标签。证据支持函数形式不等于已证部署效用：

- [MaskCO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf)：支持 partial adjacency 条件完整掩码重构和 encoder/decoder 联合训练。本项目实际 upstream CVRP 用 customer-row 正边 CE；不把通用 BCE 表述当作代码。
- [RankNet, Burges et al. 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)：支持配对 logistic 排序损失，不能保证 route-score 对成本排序或在线收益。
- [Mercier et al. 2017](https://doi.org/10.1111/1541-4337.12269)：支持区分温度管理与商品品质。本工程保留旧固定 rate C0 仅复核，不能宣称温度依赖动力学或车辆商品物理校准。
- [Pillac et al. 2013](https://publications.polymtl.ca/13267/)：信息演化是动态路由核心。snapshot 无货物生命周期/因果滚动部署资格，训练成功不补足这些证据。

## 尚未启动的部署与确认门

真正科学数据必须在首次生成之前登记：新64 TRAIN、新16开发诊断、新3×40确认、掩码/日程/臂顺序/bootstrap种子和实际源码/原权重/合同身份。最终20260927/28永远禁用，不复用已看DEV。开发16仅诊断，不用于checkpoint或超参选择。

确认提案：全部臂同总10秒，含特征、真实推理、搜索、认证和提交；新模型两目标/原暖启动/随机10D未训练/ANY50/GLS，路线、manifest、候选、回退和超时全存。三个新40队列各自对ANY50和GLS相对改善均值≥1%、固定2000次配对bootstrap95%CI下界>0；另对CE-only均值与CI下界>0作目标归因。不得将三队列混合掩盖某队列失败。随机未训练不隔离预训练效应；如要该主张，还缺同训练日程随机初始化完整模型控制。GLS若仍以距离提案只能称具体强距离搜索候选，不能称C0全目标最优。

启动屏障是独立教师验收、实际10D全模型/目标验收、全部部署臂总10秒集成验收，加新的正式注册身份。当前训练器只有完整训练实现，不含训练后10D anytime inference adapter及强对照新完整轨迹留档链；这些仍须集成验证。因此没有自动科学启动权限，也不承诺正向增益。
