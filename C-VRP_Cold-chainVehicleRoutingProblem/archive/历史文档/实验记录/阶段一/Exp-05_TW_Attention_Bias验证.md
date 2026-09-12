# 实验 Exp-05: 时间窗 Attention Bias 验证

- **日期**：2026-07-29 ✅ 完成
- **目标**：验证将 TW 兼容性矩阵作为 decoder attention soft bias 注入的效果
- **假设**：TW attention bias 可提升可行解率和收敛速度

## 设计思路

在 decoder attention 中注入 TW 兼容性作为 soft bias：
- 对节点对 (i, j)：若 `t_start[i] + service[i] + travel(i,j) <= t_end[j]` → 兼容，bias = 0
- 不兼容 → bias = -penalty（-5.0），降低 attention 权重但不完全禁止（soft constraint）
- depot 始终与所有节点兼容
- 实现：预计算 `tw_attn_bias` 矩阵 (batch, nodes, nodes)，在 `_decode_step` 中合并到 `adjmat` attention bias

## 实验设计

| 条件 | TW filter | TW attention bias | 说明 |
|-------|-----------|-------------------|------|
| A (无 bias) | ✅ | ❌ | 基线：仅候选边过滤 |
| B (有 bias) | ✅ | ✅ (penalty=5.0) | 候选边过滤 + decoder attention 软约束 |

对照组：Exp-02 (零-shot CVRP checkpoint)

## 配置

| 参数 | 值 |
|---|---|
| 模型 | CVRPTWModel (softcap_fn, 256-dim) |
| Checkpoint | exp04_5d/step5000.ckpt (5000 steps CVRPTW 训练) |
| 数据集 | cvrptw50_r1_test.npz (128 instances) |
| batch_size | 8 |
| runs | 8 |
| cycles | 40 |
| keep_rate | 0.3 |
| two_opt_steps | 4 |
| tw_attn_penalty | 5.0 |
| enable_tw_filter | True |

## 结果

| 指标 | Exp-02 (零-shot) | 条件 A (无 bias) | 条件 B (有 bias) | Δ (B-A) |
|------|------------------|------------------|------------------|---------|
| mean cost | inf ❌ | **9.996** ✅ | **11.104** ✅ | +11.1% |
| TW feas rate | 0.0% | 0.0% | 0.0% | 0 |
| Avg TW viol/inst | 23.5 | 9.0 | **7.4** | **-17.8%** |
| Gap vs greedy | inf | -25.7% | -17.5% | — |
| Inference time | 5s | 4s | 5s | +25% |

## 分析

### 1. 5D 训练效果显著 ✅

零-shot → 5000 步训练后：
- Cost: inf → ~10（容量约束从全违反到全满足）
- TW viol: 23.5 → 9.0（TW 违规减少 62%）

仅 5000 步 mask-and-reconstruct 训练，模型已学会基本的容量约束和部分 TW 约束。

### 2. TW attention bias 有效但代价明显 ⚠️

| 效果 | 方向 | 幅度 |
|------|------|------|
| TW viol 减少 | ✅ 正面 | -17.8% (9.0→7.4) |
| Cost 增加 | ❌ 代价 | +11.1% (10.0→11.1) |
| 推理耗时增加 | ❌ 代价 | +25% (4s→5s) |

TW bias（penalty=-5.0）成功迫使模型避开 TW 不兼容边 → 违规减少。但代价是模型被迫选择更长的绕路路径 → cost 上升。这是 **cost vs feasibility 的经典 trade-off**。

### 3. TW 可行率仍为 0% ❌

**根因分析**（按影响排序）：

| 因素 | 影响级别 |
|------|----------|
| C++ 2-opt 不检查 TW | 🔴 最大 |
| 仅训练 5000 步 | 🟡 次要 |
| 贪心参考解质量低 | 🟡 次要 |
| loss 函数无 TW 惩罚 | 🟡 次要 |

### 4. Gap 为负值的原因

模型生成的解 cost 低于贪心参考解（-25.7%），但 TW 可行率为 0%。模型在优化**纯距离目标**——因为 loss（边预测交叉熵）不含 TW 惩罚，C++ 局部搜索也不检查 TW。

## 结论

- **假设部分成立** ✅：TW attention bias 减少 TW 违规 18%，但可行率仍为 0%
- **核心瓶颈** 🔴：C++ 局部搜索算子（2-opt、insertion）不检查 TW 约束
- **优先级调整**：Exp-07（TW-aware 局部搜索）应先于 Exp-06

## 下一步

→ **Exp-07**（优先级提升）：带 TW 检查的局部搜索算子
→ **Exp-06**（延后）：约束感知解码掩码
