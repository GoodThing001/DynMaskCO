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

## 4. 复核更正与实施结果（2026-10-13，用户批准「追加 CO-enriched-ML 适配（改造 + 重训）」）

本节结论以 [`a1_accept_v1/RESULTS.md`](a1_accept_v1/RESULTS.md) 为准（含全部命令与原始数字）：

1. **PC-HGS 可编译**（上文「需 cmake-20 编译适配」的可行性疑问已解决）：cmake 3.31.6 +
   g++ 15.2.0（C++20），源外构建 → `a1_accept_v1/build_pchgs/PCHGS`（529,960 B，BUILD_EXIT=0）。
   真实障碍是 CPM 自举下载、三个依赖 clone 超时、nlohmann_json 自带 doctest 在 glibc≥2.34
   编译失败，均已绕过且**未修改第三方原文件**。→ **不需要 OR-Tools/PyVRP 替代 CO 层**。
2. **桥接已实现**（`a1_accept_v1/coenrich_{features,model,provider}.py`）：官方 FeatureComputer
   逐行复用、官方 MLP 架构 NumPy 复刻（对账 max|Δ|=4.8e-07）、PC-HGS 经官方 JSON 接口作
   优先级生成器；TW/温区/能耗预算交共享 `certify_plan`。烟测全绿（覆盖性/确定性/偏好改变
   排序 58–59 之 60/未来信息不变性/计时 0.02 s·2.1 s）。
3. **已按 A-v1 目标重训**：训练天 seed 20260920（≠ 开发 20260926）、8 天 1567 样本、
   与官方同架构、3000 步 4.4 s CPU；val acc 0.26→**0.8375**（多数类 0.5565），
   与官方权重分数相关 r=−0.22 → **确实改变决策**。训练目标为替代口径（参考臂接单标签回归，
   非原文结构化学习损失），已在 RESULTS.md §3 明确标注。
4. **尚未接线**：F1 正式批次运行期间不改 `ordering_providers.py`；接线只需注册表加一行，
   命令见 `a1_accept_v1/README.md` §5。
