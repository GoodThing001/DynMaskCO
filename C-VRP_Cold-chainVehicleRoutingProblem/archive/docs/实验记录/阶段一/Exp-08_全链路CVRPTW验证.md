# 实验 Exp-08: 全链路 CVRPTW 训练 + 推理验证

- **日期**：2026-07-29 ✅ 完成
- **目标**：完整 200K steps CVRPTW 训练 + 推理，验证 mask-reconstruct 全链路

## 训练

| 参数 | 值 |
|------|-----|
| 模型 | CVRPTWModel (softcap_fn, 256-dim, 5D input) |
| 数据 | cvrptw50_r1_train.npz (1280 instances) |
| 步数 | 200,000 |
| batch_size | 64 |
| lr | 1e-3 |
| 耗时 | ~1.5h |

**Loss 收敛**：2.82 → 0.69（前 5000 步快速下降，后续平稳 plateau）

## 推理结果

与 5K 步训练完全一致（feas=2.3%），**训练步数不是瓶颈**。

## 突破

后续 Phase 2 的 EDD 修复 + TW-aware 2-opt 将 feas 从 2.3% 提升至 49.2%。

→ 详见 `00_Phase总结.md`
