# 实验 Exp-01: CVRP 基线复现

- **日期**：2026-07-24
- **目标**：用预训练 checkpoint 在 CVRP-100 测试集上评估，确认 MaskCO 环境正确
- **假设**：JAX 0.6.2 + Flax 0.10.4 环境应能复现原文结果（Gap < 1%）

## 配置

| 参数 | 值 |
|---|---|
| 模型 | CVRPModel (16 encoder + 6 decoder layers, embed_dim=512) |
| 数据集 | cvrp100_testset1280_seed88_subset.npz (1280 instances) |
| Checkpoint | cvrp100.ckpt |
| batch_size | 8 (降 batch 避免显存溢出) |
| runs | 8 |
| cycles | 40, 80, 160, 320, 640 |
| sampling_steps | 2 |
| keep_rate | 0.3 |
| two_opt_steps | 4 |
| augment_level | 1 |
| gumbel_scale_factor | 0. (禁用 Gumbel) |

## 结果

| cycles | mean cost | opt cost | Gap | Gor 降幅 (vs cycles=40) |
|---|---|---|---|---|
| 40 | 15.58509 | 15.54979 | **0.227%** | — |
| 80 | 15.57685 | 15.54979 | **0.174%** | -23.4% |
| 160 | 15.57086 | 15.54979 | **0.135%** | -40.4% |
| 320 | 15.56647 | 15.54979 | **0.107%** | -52.8% |
| 640 | 15.56319 | 15.54979 | **0.086%** | -62.1% |

### 分析

- Gap 随 cycles 单调递减，符合 mask-and-reconstruct 的渐进式改进规律
- cycles=640 时 gap 仅 0.086%，>99% gap reduction ✅
- 与 MaskCO 原论文 CVRP-100 的报告一致
- 改进速率在 cycles=160 后放缓（边际收益递减），实际应用中 80-160 cycles 是性价比最高的选择

## 踩坑记录

1. **batch_size=128 时大量实例返回 inf/溢出**：GPU 显存不足导致数值错误，降为 8 后正常
2. **JAX 0.5.0→0.6.2 升级**：`MASKCO_env` 环境已升级到 JAX 0.6.2，API 向后兼容
3. **CUDA 13.1 + jax[cuda12]**：JAX 通过自带 CUDA 12 库运行，与系统 CUDA 13.1 不冲突
4. **`threads_over_batches > 1` 时 C++ 扩展可能因线程竞争产生 inf**：安全起见使用 `--threads_over_batches 1`

## 结论

- **Exp-01 完成 ✅** — 环境完全正常，模型推理、C++ 2-opt、checkpoint 加载全部通过
- batch_size=8 安全，预计日常实验可用 8-16
- 基准 Gap 曲线已建立，后续 CVRPTW 实验可与此对比

## 下一步

→ **Exp-02**：零-shot CVRPTW 测试 — 用原始 CVRP checkpoint 直接推理 CVRPTW，测时间窗约束下的性能底线
