# Phase 总结

> ⚠️ **历史记录（Phase A-D，2026-08-03）**：阶段一（7/22-8/03）的 Phase 总结。后续工作见总执行方案与 [`实验结果总表.md`](../../../../docs/实验记录/实验结果总表.md)。本文档的「负 gap」结论已被 P0-1 修正。
>
> 最后更新：2026-08-03 | 全部完成

## 项目周期：2026-07-22 → 2026-08-03（13天）

---

## Phase A：静态验证 ✅ (7/22-7/31)

| 实验 | 关键结果 |
|------|----------|
| Exp-01 | CVRP gap 0.086% |
| Exp-02 | 零-shot inf/0% → 必须重新训练 |
| Exp-03 | R1/C1/RC1 + ColdChain 数据就绪 |
| Exp-04 | 5D 收敛快 30% |
| Exp-05 | TW bias viol -18%, 与 2opt 冲突 |
| Exp-07 | 首次 TW 可行 2.3% |
| Exp-08 | 200K 步 ≠ 瓶颈 |
| Exp-09 | 混合训练 C1 7%→62.5% |
| Exp-10 | ColdChain 6D feas 72.7% |
| Step1 | TW penalty ❌ 无效 |
| Step2 | LKH3 ❌ 窄TW不可行 |
| Step3 | 100-node feas 38% ✅ |
| Step4 | C++ 30,000x 加速 ✅ |

### 方法论演进 (R1)

| 方案 | feas | viol | cost | gap |
|------|------|------|------|-----|
| 零-shot | 0% | 23.5 | inf | inf |
| TW filter | 0% | 9.0 | 10.0 | -25.7% |
| +preserving 2opt | 2.3% | 9.7 | 12.4 | -7.6% |
| EDD repair | 0% | 2.1 | 11.1 | -17.3% |
| EDD+TW 2opt (Python) | 49.2% | 0.3 | 14.3 | +6.4% |
| EDD+TW 2opt (C++) | 50.0% | 0.1 | 14.3 | +5.9% |

---

## Phase B：动态数据 ✅ (8/1-8/2)

9 组数据 (R1/C1/RC1 × EDoD 0.2/0.5/0.8), ColdChainDataloader 7D 自动检测

### EDoD 敏感性

| Type | EDoD=0.2 | EDoD=0.5 | EDoD=0.8 | avg |
|------|----------|----------|----------|-----|
| R1 | 71.1% | 64.1% | 72.7% | 69.3% |
| C1 | 85.2% | 84.4% | 81.2% | 83.6% |
| RC1 | 65.6% | 66.4% | 72.7% | 68.2% |

---

## Phase C：Event-Driven Masking ✅ (8/2)

| masking | feas | gap |
|---------|------|-----|
| random | 65.6% | +2.8% |
| **spatio_temporal** | **74.2%** | +3.1% |
| targeted | 70.3% | +3.1% |

---

## Phase D：消融 ✅ (8/2)

| Variant | feas |
|---------|------|
| Baseline | 71.9% |
| +spoilage_bias | 65.6% ❌ |
| +ST-mask | 74.2% |
| +Temp Embedding | 73.4% |

---

## 阶段二：四方向优化 ✅ (8/2-8/3)

| 方向 | 方法 | feas | gap | 结论 |
|------|------|------|-----|------|
| 二 | Temp Embedding | 73.4% | +3.1% | ✅ 删除有害 bias |
| 一 R1 | Self-Training R1 | 91.4% | -4.9% | 🔥 +18pp |
| 一 R2 | **Self-Training R2** | **93.8%** | **-7.9%** | 🔥🔥 最优 |
| 三 | Predictive 0.65x | 78.9% | -8.4% | 不如 ST 0.8x |
| 四 | EDD GPU (JAX) | — | — | ✅ 12.3x faster than C++ |

### EDoD 全矩阵最终

| Type | EDoD=0.2 | EDoD=0.5 | EDoD=0.8 |
|------|----------|----------|----------|
| R1 | 73.4% g=+3.2% | **93.8% g=-7.9%** | 71.1% g=+3.1% |
| C1 | 85.9% g=+0.1% | 79.7% g=-0.1% | 78.9% g=+0.2% |
| RC1 | 64.8% g=+4.0% | 70.3% g=+4.0% | 75.0% g=+4.3% |

---

### 工业落地验证 (8/3)

| 测试模式 | feas |
|---------|------|
| clean (欧式) | 48.4% |
| asymmetric | 55.5% |
| perturb | 39.8% |
| **both (非对称+扰动)** | **76.6%** 🔥 |

vs clean baseline: 93.8% → 76.6% (-17.2pp, 合理的工业落地代价)

---

## 最优配置

**模型**: DynamicColdChainModel + Temp Embedding  
**训练**: ST-mask (0.8x) + Self-Training Round2 伪标签  
**解码**: C++ EDD repair + TW-aware 2opt + TW filter  
**结果**: R1 EDoD=0.5: feas 93.8%, gap -7.9%, viol 0.0
