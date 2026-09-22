# MaskCO 直接目标训练（干净协议）冻结结果

日期：2026-09-22

## 结论（一句话）

在 mask 池统一为 `decision_pool_from_vehicles` 口径、训练数据完全合规之后，**直接目标微调（M-trained）显著优于未微调的 M-pre，但仍显著差于 regret-2（R）**；因此不进入完整闭环，**结束「MaskCO 直接目标训练」分支**。

> 这不是「MaskCO 不可学习」的结论。准确表述是：**直接目标微调有效吸收了冷链修复信号（相对 M-pre 有稳定改善），但尚未超过廉价的 regret-2 参照，当前部署价值与方法增益不足。**

## 背景：本次协议修复

第一次直接目标训练存在两个已实证缺陷（见 `决策与审计/直接目标训练后续决策_状态身份与固定状态诊断.md`）：

1. 采集器保存可变对象引用 → 训练输入与奖励错位；
2. mask 池用「所有可见未服务客户」，越界选了 non-replan 冻结车 tail 客户（审计：256 状态中 10.5% 越界）。

本次修复后重新训练：

- **统一 mask 池入口** `mask_candidate_pool` + `validate_mask_scope`（`scripts/simulation/dynmaskco_cc_context.py`），训练/部署/回放/固定状态评估共用；
- 状态决策时刻深拷贝冻结 + 每实例 4 状态均衡采样；
- 全方法完整计划认证（exact-once / 逐车顺序 / 不可改与 protected 不变）；
- R 失败保留 P0（部署语义），不静默删除。

## 冻结身份

| 项 | 值 |
|---|---|
| 训练 checkpoint SHA256 | `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5` |
| 训练数据 SHA256 | `c70258c4fde86e5ebfb08c17e4f8ccacac5f1b18319dc180eaa048e8ae771d01` |
| seed / 步数 / 采样 | 42 / 1000 / 8×4 |
| s_TRAIN / final_loss | 0.1334 / −0.0357 |
| 状态覆盖 | 256 = 64 实例 × 4，`out_of_pool = 0` |

完整源码哈希见 `training_identity.json`（含 12 个关键源文件 SHA256 + 训练 summary + state_manifest）。

## 固定状态结果（干净 checkpoint，实例级配对 bootstrap CI）

部署语义：失败用 J0 兜底，不静默删除。三种方法失败率与认证失败率均为 0。

**TRAIN（16 实例 / 32 状态）**

| 方法 | 部署均值 J_vis |
|---|---|
| R (regret-2) | **1.0708** |
| M-trained | 1.1039 |
| M-pre | 1.1562 |

| 配对 | Δ [95% CI] |
|---|---|
| M-trained − M-pre | **−0.0524** [−0.0697, −0.0352] ✅ 显著为负 |
| M-trained − R | **+0.0331** [+0.0204, +0.0456] ❌ 显著为正 |

M-trained vs R 胜/平/负：**7 / 2 / 23**。

**CAL（16 实例 / 32 状态）**

| 方法 | 部署均值 J_vis |
|---|---|
| R (regret-2) | **1.1519** |
| M-trained | 1.1811 |
| M-pre | 1.2379 |

| 配对 | Δ [95% CI] |
|---|---|
| M-trained − M-pre | **−0.0568** [−0.0746, −0.0387] ✅ 显著为负 |
| M-trained − R | **+0.0292** [+0.0177, +0.0417] ❌ 显著为正 |

M-trained vs R 胜/平/负：**5 / 6 / 21**。

原始 JSON：`fsq_clean_train.summary.json` / `fsq_clean_train.analysis.json`、`fsq_clean_cal.summary.json` / `fsq_clean_cal.analysis.json`、`mask_audit_summary.json`。

## 决策

对照闭环准入三条标准：

| 标准 | 判定 |
|---|---|
| ① M-trained 相对 M-pre 稳定配对改善 | ✅ 满足 |
| ② 与 R 差距明显缩小 | ❌ 不满足（仍 +0.029~0.033，显著） |
| ③ 在某类明确状态稳定超过 R 且能解释机制 | ❌ 不满足（仅 7/32、5/32 胜） |

**决定：不进入完整闭环，结束「直接目标训练」分支。** 此分支正式冻结，不再补 seed、不跑完整闭环、不继续调参。下一阶段应重新定义学习模块相对 R 的职责。

## 复现命令（服务器，`C-VRP_Cold-chainVehicleRoutingProblem/` 根）

```bash
# 1. 干净训练（seed42 / 1000 步 / 8×4，网络/奖励/lr 不变）
python scripts/training/train_mpre_reinforce.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --num-steps 1000 --batch-states 8 --K 4 --seed 42 \
    --out results/m0_scale/mpre_reinforce_s42_clean

# 2. mask 范围审计（应 out_of_pool=0）
python scripts/evaluation/run_mask_scope_audit.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --max-instances 64 --max-per-instance 4 --out results/m0_scale/mask_audit_train_v2

# 3. 固定状态评估（TRAIN / CAL）
python scripts/evaluation/run_fixed_state_quality.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --split train --out results/m0_scale/fsq_clean_train
# CAL：--data dcc_50_r1_edod05_cal_teacher.npz --split cal --out results/m0_scale/fsq_clean_cal

# 4. 配对 CI / 胜平负分析
python scripts/evaluation/analyze_fixed_state.py \
    --per-state results/m0_scale/fsq_clean_train/per_state.json --out analysis.json

# 5. 训练身份记录
python scripts/evaluation/record_training_identity.py \
    --out-dir results/m0_scale/mpre_reinforce_s42_clean \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz --root .
```

## 服务器产物路径（大文件不入 git）

- checkpoint：`results/m0_scale/mpre_reinforce_s42_clean/model.ckpt`（89.6MB，不入库）
- 固定状态：`results/m0_scale/fsq_clean_train/`、`results/m0_scale/fsq_clean_cal/`（`per_state.json` ~160KB，不入库）
- mask 审计：`results/m0_scale/mask_audit_train_v2/`
- 旧缺陷 checkpoint（对照，已废弃）：`results/m0_scale/mpre_reinforce_s42_defective/`、`results/m0_scale/mpre_reinforce_s42_corrected/`
