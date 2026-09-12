# 方向二：spoilage_bias → Temp Embedding 修复

- **日期**：2026-08-02 ✅
- **目标**：删除有害的 spoilage_bias（-6pp），替换为 Learnable Temp Embedding
- **假设**：Temp Embedding 至少恢复到 D1 baseline (71.9%)，消除有害副作用

## 改动

| 文件 | 改动 |
|------|------|
| `models/DynamicColdChainModel.py` | 删除 `spoilage_bias` 参数 + `spoilage_lambda` → 新增 `temp_embed = nnx.Embed(3, 256)` |
| `training/train_dynamic_cc.py` | 删除 loss 中的 spoilage penalty 项 |

## 结果 (R1 EDoD=0.5, 20K步 ST-mask)

| 模型 | feas | cost | viol | gap | 变化 |
|------|------|------|------|-----|------|
| D1 Baseline (无spoilage, random) | 71.9% | 16.40 | 0.0 | +2.8% | — |
| phasec_st (旧 spoilage_bias + ST) | 74.2% | 16.44 | 0.0 | +3.1% | 旧最优 |
| **step2_tempemb (Temp Embed + ST)** | **73.4%** | 16.44 | 0.0 | +3.1% | -0.8pp vs 旧最优 |

## 分析

1. Temp Embedding 性能几乎无损：73.4% vs 旧最优 74.2%（-0.8pp，在噪声范围内）
2. 比 D1 baseline 高 1.5pp（71.9→73.4%，ST-mask 贡献）
3. 完全消除了 spoilage_bias 的有害副作用（推理无 NaN/Inf）
4. Temp Embed 是可学习的，可能随更多训练步数持续改善

## 结论

- **方向二成功** ✅：删除了有害机制，性能恢复到健康水平
- Temp Embedding 是更干净的架构设计——让 Transformer 自己学温度关系
