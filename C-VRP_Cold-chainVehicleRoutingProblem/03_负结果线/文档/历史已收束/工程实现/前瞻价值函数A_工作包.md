# 前瞻价值函数 A · 工作包（最小原型）

日期：2026-09-22。性质：新研究方向的最小原型计划，不是已批准的大规模实验。前提：G/GV/L 局部残差已被 D1–D3 + 因果头腔诊断证伪（myopic 头腔不进入终局、myopic 顺序决策有害、顺序 oracle 头腔 ≈100% 未来信息）。经用户裁定继续走 A。

> **执行回传（2026-09-22）：A 已判死。** 第 1 步可预测性探针（采样版，252 条）显示终局增益 `terminal_gain` 从可见状态不可预测（线性探针 out-of-sample Spearman ≈ −0.04，R² 为负；myopic_gain 相关仅 +0.135）。第 2/3 步不再执行。见[探针结果](../结果与进度/价值可预测性探针结果_2026-09-22.md)。


## 1. 学习职责（为什么是它）

顺序 oracle 的 −0.21/−0.27 头腔来自**未来揭示**（clairvoyant 终局 rollout），可见状态的 myopic 评分反相关、有害。要因果捕获它，唯一可能的是学一个**期望终局价值**：

```
V_θ(s, P_partial) ≈ E[ terminal J_vis | s, P_partial ]
```

期望对**已知的 reveal/tw 分布**取（不读具体未来订单，但利用其分布结构）。用它做：
- **defer/serve 时机**：当前订单现在服务 vs 留待以后（预留车辆给未来紧时间窗订单）；
- **候选打分**：给距离搜索/重优化的候选按其期望终局价值排序。

这仍是 MaskCO 原生（部分计划条件 + 车队条件 + 掩码重构），对准唯一被证明有头腔的信号；但先验已很低（要胜过 JF1-H，且从可见状态学到 myopic 反相关的信号）。

## 2. 第 1 步（先做）：终局增益可预测性回归探针

**目标**：回答"`selected_delta`（clairvoyant oracle 每决策的终局增益）能否从可见状态特征预测"。预测不了 → A 判死，不花 GPU。

**做法**：复用 clairvoyant 顺序 oracle（`sequential_oracle_plan` + `make_oracle_hook`），每决策额外记录：
- `selected_delta`（终局增益，监督目标）；
- 可见状态特征：`myopic_delta`（myopic J_vis 增益）+ 结构化可见特征（客户 tw/demand/temp_class/reveal、车队 idle/committed/load、增量距离、n_visible/n_deferred/clock）。

**判读**：
- 若线性探针 / 小 MLP 的 out-of-sample Spearman 显著 > 0（且 > 随机基线）→ 有可学习信号，进入第 2 步；
- 若 ≈ 0（myopic_delta 反相关且结构化特征也无增量）→ A 判死，转 B（直面硬约束）。

**预算**：8 实例、无训练、纯 NumPy + sklearn（或手写线性），1 天内。

## 3. 第 2 步（探针通过后）：最小 V_θ + 接口验收

- V_θ：可见状态 + 部分计划 → 标量（复用小 MLP，或 MaskCO 编码器 + 价值头）；输入一致性验收（train/online/diag 三路张量逐字段一致）。
- 目标：回归 `terminal J_vis`（或 `selected_delta`），监督来自 clairvoyant oracle 的 per-decision 记录。
- 验收：探针特征 vs V_θ 输入对齐；确定性 + 近并列动作不翻转。

## 4. 第 3 步（验收通过后）：有限 pilot

- V_θ 驱动的 defer/serve/预留 + 候选打分，vs D（距离搜索）/ R（regret-2）在同一公开信息、动作范围、墙钟预算下。
- 预注册立项门（建议仍"相对 D 平均 J 降 1%"），G/V/L 式对照（V_θ vs 同信息轻量网络）以证 MaskCO 必要性。

## 5. 停止边界

- 探针无信号 → A 收束，转 B。
- pilot 无增量 → A 收束；届时证据链（残差 + 价值函数均无因果头腔）将强烈支持"当前合同/规模对 MaskCO 无正贡献信号"，需在 (i) 放宽硬约束 / (ii) 负结果论文 / (iii) 换任务 三者中拍板。
