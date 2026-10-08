# CO-enriched-ML A-v1 适配评估（2026-09-29，P1 评估项）

## 1. 方法与资产

- 论文：arXiv:2304.00789（Baty et al., Transportation Science 2024）；仓库
  `CC_Compare/CO-enriched-ML/`（EURO Meets NeurIPS 2022 挑战协议）。
- 方法 = **ML-CO 管线**：特征工程（FeatureComputer）→ 学习策略预测「本 epoch 派车集合」
  → PC-HGS（C++，需 cmake-20 编译）路由已派集合 → 下一 epoch。
- 权重：仓库自带 1 个（NN iteration-29，挑战实例/其特征口径）；A-v1 需重训。

## 2. 语义映射（挑战协议 → A-v1）

| 维度 | CO-enriched-ML | A-v1 | 桥接 |
|---|---|---|---|
| 动态性 | epoch 波次派车（固定时刻批量决策） | 揭示即承诺（连续） | 波次→逐揭示决策需改造 |
| 拒单 | **无接受/拒绝**——未派请求顺延到后续 epoch | accept/reject 不可撤销 | 需显式加入 reject 动作 |
| 路由 | PC-HGS 每 epoch 重路由（多趟回 depot） | 单趟 pickup-to-depot + 在途接单 | 语义不同 |
| 目标 | 距离/惩罚 | 收入−燃油−拒绝损失 | 目标需换 |
| 冷链 | 无 | 预算/温区 | 外层 C0 |

## 3. 结论

- **可桥接但工作量高**：其「ML 预测派车 + CO 路由」范式与 A-v1 的「场景前瞻 + 共享下游」
  同构，但其波次/顺延语义与「揭示即承诺」冲突，需 (a) 改造策略动作为 accept/reject、
  (b) 按 A-v1 特征/目标重训、(c) PC-HGS 编译适配（C++20）。列为「需训练候选」，不阻塞当前批次。
- 文献价值：作为「预测+优化」动态 VRPTW 的定位引用（相关工作表）。
- 建议：P1 评估完成；实施排在 P2 之后（若需要「预测+优化」学习型对照再立项）。
