# Phase C: Event-Driven Masking + Spoilage Bias

- **日期**：2026-08-02 ✅ 完成
- **目标**：DynamicColdChainModel (7D + spoilage_bias) + 三种掩码策略对比

## 技术实现

| 文件 | 内容 |
|------|------|
| `models/DynamicColdChainModel.py` | 7D 模型 + spoilage_bias (Arrhenius 腐败注意力偏置) |
| `training/train_dynamic_cc.py` | 训练入口, 支持 `--masking_mode` (random/spatio_temporal/targeted) |
| `decoding/cvrptw.py` | 增加 DynamicColdChainModel 加载支持 |

## 三种掩码策略

| 模式 | keep_prob | 设计意图 |
|------|-----------|---------|
| random | 1.0x (默认) | 与 MaskCO 原论文一致 |
| spatio_temporal | 0.8x | 模拟动态场景：更多掩码 → 更少已知信息 |
| targeted | 0.6x | 冷链特化：更多掩码 → 迫使学习约束 |

## 结果 (R1 EDoD=0.5, 50K步)

| 模型 | masking | feas | cost | viol | gap |
|------|---------|------|------|------|-----|
| Phase B ColdChain | random (baseline) | 64.1% | 16.44 | 0.0 | +3.1% |
| DynamicColdChain | random + spoilage | 65.6% | 16.40 | 0.0 | +2.8% |
| DynamicColdChain | **spatio_temporal** | **74.2%** | 16.44 | 0.0 | +3.1% |
| DynamicColdChain | targeted | 70.3% | 16.45 | 0.0 | +3.1% |

## 结论

1. **Spatio-Temporal masking 最优**：feas +10pp vs baseline (64→74%)
2. **spoilage_bias 效果温和**：+1.5pp feas (64→65.6%)
3. **低 keep_prob 是有效数据增强**：模拟动态场景的信息不足 → 模型学习到更鲁棒的表征
4. **gap 稳定 ~3%**：所有模型 mask-reconstruct 持续改进
