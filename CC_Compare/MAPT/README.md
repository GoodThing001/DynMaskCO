# MAPT — Multi-Agent Pointer Transformer（AAAI 2026）

> **仅论文数据对比**，无公开源码。
> 用途：作为 DynMaskCO 主对比表「动态随机请求在线决策」赛道的论文锚点，记录可从论文提取、用于对比的数据与指标。

---

## 1. 论文信息

| 项 | 内容 |
|------|------|
| 标题 | Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems |
| 作者 | Zengyu Zou, Jingyuan Wang（通讯）, Yixuan Huang, Junjie Wu（北京航空航天大学） |
| 会议 | **AAAI 2026**（Proceedings of the Fortieth AAAI Conference） |
| 论文（arXiv） | https://arxiv.org/abs/2511.17435 （v2, 17 Dec 2025） |
| 论文（AAAI 官网） | https://ojs.aaai.org/index.php/AAAI/article/view/38700 |
| DOI | `10.1609/aaai.v40i19.38700` |
| 代码地址（论文内声明） | https://github.com/Beihang-BIGSCITY/MAPT — ⚠️ **截至 2026-08 不可公开访问**（无开源） |
| 本地 PDF | `C-VRP_Cold-chainVehicleRoutingProblem/docs/论文参考/对比方法/对比方法/2511.17435v2.pdf` |

> 建议在投稿前再 check 一次该 GitHub 是否转为公开；若公开则升级为「可复现对比」，否则维持「仅论文数据」。

---

## 2. 解决的问题（MVDPDPSR）

MAPT 解决 **Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests（多车动态取送货 + 随机请求）**：

- **取送货（PDP）**：每个请求有 `from`（取件站）和 `to`（送件站），车辆须先取后送。
- **多车**：K 辆车协同，车辆状态 `⟨容量 cap, 剩余空间 space, 当前目的地, 剩余行程 dist⟩`。
- **动态随机请求**：每个请求有 `appearance time`（揭示时间），到达后才可见 —— 与 DynMaskCO 的 `reveal_time` / EDoD **机制直接对齐**。
- **目标**：`maximize 完成请求的总利润(profit) − 行驶成本(cost × 距离)`，非纯距离最小化。
- **决策范式**：离散时间片 MDP，每步做「请求分配 + 车辆下一站选择」的联合动作（中心化，非多智能体独立解码）。

### 与 DynMaskCO 的差异（对齐分析，关键）

| 维度 | MAPT | DynMaskCO（本方法） |
|------|------|------|
| 问题 | 取送货 PDP（先取后送） | 纯送货（单程，depot 出发） |
| 动态性 | 请求按 `appearance time` 揭示 ✅ | 订单按 `reveal_time` 揭示（EDoD）✅ |
| 时间约束 | **无时间窗**（只有揭示时间） | **硬时间窗** TW |
| 目标 | 利润最大化（profit − cost） | 纯距离最小化 + 品质/能耗多资源 |
| 冷链 | ❌ 无 | ✅ 温度分级 + 品质衰减 + 制冷能耗 |
| 求解 | PPO + Transformer(enc 6/dec 2, h=128) + Pointer | masked-generation + K-beam 资源状态解码 |

> **结论**：两者**唯一机制级对齐点 = 动态揭示 + non-anticipatory 在线决策**；问题结构（PDP vs 送货）、目标（利润 vs 距离）、约束（无 TW vs 硬 TW）均不同。因此按 `docs/ALL/基线对比.md` §5 对齐协议，MAPT 只能做**「同问题定位」引用 + 定性对比**，**不能做数值直接对比**（除非把双方都压到同一目标标量 J 下重算，但 PDP 结构差异使这一步不可行）。

---

## 3. 数据集（8 个，可引用）

### 3.1 合成数据（5 个）

| 场景 | 站点 I | 请求 M | 车辆 K | 时间片 T | cost/单位距离 | 车辆容量 | profit/请求 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| synth-S | 20 | 110 | 5 | 58 | 0 | 3 | 距离 |
| synth-S-cost | 20 | 110 | 5 | 58 | 0.3 | 3 | 距离 |
| synth-L | 50 | 550 | 15 | 128 | 0 | 3 | 距离 |
| synth-L-cost | 50 | 550 | 15 | 128 | 0.3 | 3 | 距离 |
| synth-XL | 300 | 550 | 50 | 128 | 0 | 3 | 距离 |

- 站点间距离：synth-S 采样 `Uniform{0..10}`、synth-L `Uniform{0..30}`、synth-XL `Uniform{0..20}`，再跑最短路。
- 请求 OD 均匀采样；`appearance time` 采样 `Uniform{1..T}`；profit = OD 距离。

### 3.2 真实数据（DHRD，3 个城市）

| 场景 | 城市 | 站点 I | 请求 M | 车辆 K | 时间片 T | 车辆容量 | profit/请求 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| dhrd-tpe | 台北 | 36 | 800 | 20 | 48 | 6 | 1 |
| dhrd-sg | 新加坡 | 36 | 700 | 15 | 48 | 6 | 1 |
| dhrd-se | 斯德哥尔摩 | 36 | 200 | 3 | 48 | 6 | 1 |

- DHRD 数据集（Assylbekov et al. 2023，ACM RecSys'23）：外卖配送请求，按 5-char geohash 划站点；距离 = 最少穿越区域数；76 天训练/验证 + 14 天测试。

> 这 8 个数据集与我们的 Solomon R1/C1/RC1（50 节点）**实例集不同**，规模也不同（请求数 110–800 vs 我们 50 节点），**不能直接对表**。

---

## 4. 评估指标

| 指标 | 符号 | 含义 | 与我们对齐? |
|------|------|------|:---:|
| 目标值 | Obj ↑ | 总利润 − 行驶成本 | ❌（利润最大化 ≠ 距离最小化） |
| 请求完成率 | Comp ↑ | 完成请求 / 总请求 | ⚠️（≈ 我们的服务率，但口径不同） |
| 推理时间 | Time ↓ | 单实例平均决策时间 | ✅（可对齐 wall-clock budget） |

---

## 5. 基线方法（论文内的对比对象）

| 类别 | 方法 | 说明 |
|------|------|------|
| Rolling-Horizon | **OR-Tools（CP-SAT）** | 滚动时域 + CP-SAT 精确解 |
| Rolling-Horizon | **SA / GA** | 滚动时域 + 模拟退火 / 遗传算法 |
| Static（全知） | **MAPDP\* / PARCO\*** | MARL 静态方法（clairvoyant，不现实） |
| 规则 | **Nearest** | 贪心最近邻 |
| MDP 逐步 | **MAPDP** | 静态求解器每步解一步（可动态用） |

---

## 6. 论文核心结果（可引用数字）

### 6.1 主表（Table 2）— MAPT vs 基线

MAPT 行（Obj ↑ / Comp ↑）：

| 场景 | MAPT Obj | MAPT Comp | 最强学习基线 MAPDP | 强 OR 基线 OR-Tools |
|------|:---:|:---:|:---:|:---:|
| synth-S | **275.2** | **0.83** | 157.5 / 0.41 | 182.1 / 0.54 |
| synth-S-cost | **192.1** | **0.84** | 70.0 / 0.42 | 117.4 / 0.52 |
| synth-L | **1875.1** | **0.80** | 958.1 / 0.37 | N/A |
| synth-L-cost | **1348.6** | **0.81** | 385.2 / 0.37 | N/A |
| dhrd-tpe | **697.2** | **0.87** | 135.4 / 0.17 | N/A |
| dhrd-sg | **376.1** | **0.71** | 64.9 / 0.18 | N/A |
| dhrd-se | **78.9** | **0.64** | 19.3 / 0.16 | 55.3 / 0.45 |
| synth-XL | **3227.5** | **0.65** | 2543.1 / 0.50 | N/A |

> 「N/A」= 该算法在可接受时间窗内跑不出结果。

### 6.2 推理时间（Table 4，秒/实例，合成数据）

| 场景 | OR-Tools | SA | GA | Nearest | MAPDP | MAPT |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| synth-S | 401.7 | 23.7 | 16.7 | 0.059 | 0.79 | **1.42** |
| synth-L | N/A | 166.7 | 103.1 | 0.117 | 31.98 | **8.30** |
| synth-XL | N/A | 613.9 | 354.8 | 0.148 | 33.22 | **27.43** |

> MAPT 比 OR-Tools/SA/GA 快 1–2 个数量级，但比 Nearest（贪心）慢。

### 6.3 消融（Table 3，证明三组件有效）

| 变体 | 结论 |
|------|------|
| w/o Relation-Aware Attention | 略降（所有数据集） |
| w/o AutoRegressive Decoding | 显著降（synth-S 275.2→217.2） |
| w/o Informative Priors | 显著降（synth-S 275.2→175.2） |

### 6.4 泛化（Table 8，与我们的「可迁移表示」直接相关）★

行 = 训练集，列 = 评估集（Obj ↑ / Comp ↑）：

| 训练 \ 评估 | synth-S | dhrd-se | synth-XL |
|------|:---:|:---:|:---:|
| **synth-S** | 275.2 / 0.83 | **74.4 / 0.61**（跨分布） | **2470.2 / 0.50**（跨规模） |
| dhrd-se | 117.4 / 0.36 | 78.9 / 0.64 | 1480.0 / 0.30 |
| synth-XL | 109.8 / 0.34 | 52.6 / 0.42 | 3227.5 / 0.65 |

> **关键对比点**：MAPT 的跨分布（synth-S→dhrd-se）与跨规模（synth-S→synth-XL）zero-shot 泛化均有**显著退化**（Comp 0.83→0.61、0.83→0.50），且跨分布反向（dhrd-se→synth-S）退化严重（0.64→0.36）。
> 而 DynMaskCO 的 zero-shot 迁移损失 **仅 0.2%（R1→C1）/ 0.6%（R1→RC1）**（`docs/实验记录/result_table.md`）。这是论文「可迁移表示」贡献的相对优势证据——但需注意两者任务不同，只能作为定性对照，不能数值并列。

---

## 7. 可用于对比的论文数据清单（摘要）

| 数据 | 出处 | 用法 |
|------|------|------|
| 动态揭示机制（appearance time） | §2.1 Definition 3 | 概念对齐：EDoD 等价物 |
| non-anticipatory MDP 逐步决策 | §2.2 / Appendix B | 与 Rolling-Horizon 对照，佐证我们 online 范式 |
| 8 数据集规模统计 | Table 5 / §3 | 引用问题规模（I/M/K/T/cap） |
| 主表 MAPT 胜率 | Table 2 | Related Work 引用「MAPT 是动态 PDP 的 SOTA」 |
| 推理时间对比 | Table 4 | wall-clock budget 对齐（同为秒级，OR 工具分钟级） |
| 泛化退化 | Table 8 | 支持「可迁移表示」相对优势（定性） |

---

## 8. 诚实注记

1. **不开源**：论文列了 GitHub 地址但不可公开访问，无法复现；只能论文数据对比。
2. **任务不同**：取送货（PDP）+ 利润最大化 ≠ 我们的送货 + 纯距离 + 硬 TW + 冷链多资源，**不做数值并列**。
3. **实例集不同**：DHRD/合成 vs Solomon，规模（请求 110–800）也不同。
4. **唯一可正面较劲的点**：动态揭示 + 在线决策范式，以及「跨分布/跨规模 zero-shot 泛化」——MAPT 泛化退化明显，DynMaskCO 迁移损失极小，但这是**定性对照**而非同口径数值对比。

> 引用（BibTeX）见论文 arXiv 页；AAAI 官方条目 `10.1609/aaai.v40i19.38700`。
