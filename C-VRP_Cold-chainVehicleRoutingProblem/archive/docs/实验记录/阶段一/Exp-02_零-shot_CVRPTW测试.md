# 实验 Exp-02: 零-shot CVRPTW 测试

- **日期**：2026-07-29 ✅ 完成
- **目标**：用原始 CVRP checkpoint 直接推理 CVRPTW 实例（忽略时间窗约束），测量零-shot 性能底线
- **假设**：模型会生成路径，但时间窗违规率会很高（>50%），可行解率低（<20%）

## 实验设计

### 数据

- 使用 `generate_cvrptw_data.py` 生成 50-node Solomon R1 实例（因服务器上 TSPTW 数据缺失）
- 规模：128 instances
- 数据路径：`C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_r1_test.npz`

### 测试矩阵

| 条件 | checkpoint | 数据 | 期望 |
|---|---|---|---|
| A | cvrp100.ckpt (原始) | CVRPTW-50 R1 | 基线：无 TW 训练，测零-shot |
| B | 无（随机初始化） | CVRPTW-50 | 下界：无训练（因 A 已全 inf，跳过） |

## 配置

| 参数 | 值 |
|---|---|
| 模型 | CVRPTWModel (从 CVRP-100 ckpt 加载，init_proj 随机初始化) |
| 数据集 | CVRPTW-50 R1 (128 instances) |
| 训练步数 | N/A (零-shot) |
| 批次大小 | 8 |
| runs | 8 |
| keep_rate | 0.3 |
| cycles | 40 |
| sampling_steps | 2 |
| two_opt_steps | 4 |
| augment_level | 0 |
| gumbel_scale_factor | 0 |
| enable_tw_filter | True |
| tw_max | 24.00 (auto-detected) |

## 结果

| 指标 | 值 | 备注 |
|---|---|---|
| Mean cost | **inf** ❌ | 所有解违反容量约束 |
| TW feasibility rate | **0.0%** ❌ | 无一实例满足时间窗 |
| Avg TW viol/inst | **23.5** | 每实例约一半节点违反 TW（共51节点） |
| Opt cost (greedy ref) | 13.46 | 贪心构造参考解 |

### 关键发现

1. **容量约束全违反**：`cvrp_eval_cost` 对所有 128 个实例返回 inf。CVRP checkpoint 的 encoder/decoder 权重期望 3D 输入特征 (x, y, demand) 的分布，但 CVRPTWModel 的 init_proj 层被随机初始化（Linear(5→256) vs 原始 Linear(3→256)），导致 encoder 输出无意义的嵌入 → decoder 预测随机边 → C++ insertion 构造出违反容量约束的路径。

2. **时间窗约束全违反**：即使 `--enable_tw_filter` 在候选边阶段过滤了 TW 不可行的边，模型预测的边质量太差，C++ 2-opt 后仍然无法满足 TW。平均每实例 23.5 次 TW 违反（50 个 customer + depot）。

3. **TW filter 效果有限**：TW 过滤仅在候选边选择阶段生效，后续的 C++ 局部搜索（2-opt、insertion）不检查 TW，可能将原本可行的边变成不可行。

## 结论

- **假设成立且结果比预期更差**：零-shot 从 CVRP checkpoint 迁移到 CVRPTW 完全不可行（0% TW feasibility, 100% capacity infeasibility）
- **根因**：`init_proj` 层随机初始化 + encoder/decoder 对 5D 特征分布的不适配
- **启示**：MaskCO 的 mask-and-reconstruct 范式**不能零-shot 泛化**到新增输入维度的约束场景，必须进行 CVRPTW 专用训练

## 踩坑记录

1. **checkpoint 部分加载失败**："Not enough leaves to unflatten the graph" — 因为 CVRPTWModel 比 CVRPModel 多出 `tw_bias` 参数和更大的 `init_proj`，Flax NNX 无法完全合并。模型使用随机初始化的新层，旧层权重复用。
2. **`best_sols` 形状不匹配**：CVRP 2-opt 会改变 route 的 padding 长度（52→53），导致不同 cycle 的 `sols` 形状不一致。已修复：改为对最终轮 sols 评估 TW 可行性。
3. **`tw_max` 默认值问题**：修复前 `--tw_max` 默认 1.0，修复后自动从数据计算（24.00），确保与训练时一致。

## 下一步

→ **Exp-03**：生成 CVRPTW 训练数据（1280+ instances）  
→ **Exp-04**：5D 输入消融实验（CVRPTW 训练 vs CVRP 训练对比 loss 收敛）
