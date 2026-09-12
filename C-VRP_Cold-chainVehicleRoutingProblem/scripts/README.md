# scripts/ — 代码索引

> 最后更新：2026-09-03（P0-M、C0 完成）
>
> ⚠️ **当前不再处于 R1.5 阶段**。论文主线、阶段和下一动作以 [`项目当前状态.md`](../项目当前状态.md) 与 [`代码实现蓝图.md`](../docs/当前规划/工程实现/代码实现蓝图.md) 为准。P0-M 与 C0 已完成；C0 参数仍为 pilot，因此冷链数值仍不能作为论文性能结论。当前执行位置请查看状态入口。

---

## 当前已完成的 P0 基础设施

| 文件 | 状态与用途 |
|---|---|
| `simulation/strict_online_env.py` | strict-online event engine 底座 |
| `simulation/recourse_snapshot.py` | P0-S `RecourseSnapshotV2` |
| `simulation/action_contract.py` | P0-A full-fleet Action Contract v1 |
| `simulation/counterfactual_teacher.py` | P0-U distance-only common-continuation teacher |
| `project_paths.py` | P0-M 工作区/上游/扩展统一路径解析 |
| `coldchain/coldchain_contract.py` | C0 版本化单位、热学、品质与目标合同 |
| `coldchain/coldchain_state.py` | C0 唯一 pickup-to-depot 热状态与 cargo manifest 转移 |
| `coldchain/coldchain_observation.py` | C0 strict-online 冷链特征屏蔽 |
| `evaluation/coldchain_evaluator.py` | C0 消费 execution trace 的权威 D/Q/E 评价 |
| `tests/test_recourse_snapshot.py` | P0-S 回归 |
| `tests/test_action_contract.py` | P0-A 回归 |
| `tests/test_counterfactual_teacher.py` | P0-U 回归 |
| `tests/test_project_layout.py` | P0-M 7 类路径/入口/syntax 检查 |
| `tests/test_coldchain_contract.py` | C0 18 项合同/状态/trace/snapshot/teacher/旧口径隔离检查 |
| `tests/test_coldchain_sensitivity.py` | C0 7 项 pilot 参数方向敏感性检查 |

P0-M 产物位于 `results/p0m/`；C0 产物位于 `results/c0/coldchain_contract_tests.json` 与 `coldchain_sensitivity_tests.json`。旧 `thermal_state.py` 仅作为统一静态回放桥接并拒绝旧单段 API；`cvrptw.py`、`rolling_horizon.py` 与数据中的旧代理字段均标为非权威 `legacy`。

## R1.5 strict-online 历史基线入口

### 训练

| 脚本 | 用途 |
|------|------|
| `run_typed_retrain.sh` | **训练入口**：`smoke`(2K) / `single`(seed42×50K) / `full`(5-seed×50K)，`RUN_TAG=r1_5_baseline` 版本化 |
| `training/train_dynamic_cc.py` | 训练主体（online-seq K=5、timestep=keep、checkpoint fail-hard） |

### 评估（strict-online）

| 脚本 | 用途 |
|------|------|
| `decoding/run_r1_5_model_utility.py` | **H0-H3 harness**：一次跑 EDD/NN/模型/shuffle，输出 summary / instance_level / event_audit / paired_h2_h3 四张 CSV + run_config.json |
| `decoding/run_online_decode.py` | 单 method runner（`--method edd/nn/model/model_shuffle`） |
| `decoding/run_r1_eval.py` | R1 45-cell 矩阵评估（5 seed × 3 type × 3 EDoD，fail-hard + instance CSV） |
| `baselines/ortools_rolling_horizon.py` | **OR-Tools-RH**（shared env，经典 baseline reference） |

### 核心模块

| 文件 | 功能 |
|------|------|
| `simulation/strict_online_env.py` | **共享事件引擎**：物理事件(Reveal/ServiceCompletion/Return) ≠ 重规划触发器(Initial/Reveal/PlanInvalidation/Exhaustion)，plan persistence，execution trace |
| `decoding/online_decode.py` | `StrictOnlineDecoder`（env 薄封装）+ `MaskCOReplanner`（模型 replanner，logit_mode=real/shuffle） |
| `decoding/resource_beam.py` | Resource Beam 搜索器（tw_margin=0，single-route lookahead 修复） |
| `evaluation/authoritative_evaluator.py` | 权威评估器：`evaluate_solution` + `evaluate_execution_trace`（trace 绝对时间） |
| `models/DynamicColdChainModel.py` | DCC 7D 模型（type_embed 5 类 + 可见性门控 + 边特征） |
| `models/cvrptw_utils.py` | `coord_normalize_visible`（仅可见节点归一化）+ causal adjacency helpers |

### 数据

| 脚本 | 用途 |
|------|------|
| `data/regenerate_baseline.py` | **D1 数据重生成**（release-feasible 修复版，archive 旧数据 + manifest） |
| `data/generate_coldchain_data.py` | DCC 生成器（release-feasible 约束：r+τ+s ≤ b） |
| `data/ColdChainDataloader.py` | 加载器（yield 未掩码特征，掩码移到训练器） |

### 测试（13 个 blocking tests，全 PASS）

| 脚本 | 用途 |
|------|------|
| `tests/test_online_protocol.py` | **#4-#18**：exact-once / prefix / dynamic-visibility / fleet / parallel / late-dispatch / committed-reservation / ready-wait / plan-persistence / reveal-recourse / stale-tail-dup / OR-wait-parity |
| `tests/test_release_feasibility.py` | **#1**：release feasibility（r+τ ≤ b） |
| `tests/test_future_perturbation.py` | **#2**：future perturbation 端到端（掩码 + 不掩码 + edge-feature） |
| `tests/test_future_cardinality.py` | **#3**：future cardinality invariance |
| `tests/test_target_label_leakage.py` | **#3b**：target label leakage |
| `tests/test_data_qc.py` | **#5**：数据 QC |
| `tests/test_no_future_leakage.py` | **P0-2**：dataloader 掩码 / 梯度隔离 / 注意力偏置 |

### 工具

| 脚本 | 用途 |
|------|------|
| `regenerate_archive.py` | 生成 `archive/代码快照/完整代码归档.txt`（源码快照） |

---

## ⚠️ 过时脚本（保留原位的历史入口）

> 以下脚本基于**旧协议**（offline 单发 decode / Phase 0 / v6 方法优化 / Phase A），
> 在 R1.5 strict-online 协议下已失效或 superseded。当前不移动这些文件；若以后归档，必须先核对所有脚本与历史文档引用，避免与 P0-M/C0 代码改动混在同一批次。

### 顶层旧批量脚本

| 脚本 | 原用途 | 失效原因 |
|------|------|---------|
| `run_method_freeze.sh` | 旧权威入口（offline 单发 decode） | complete=0%，被 strict-online 替代 |
| `phase0_freeze_baseline.sh` / `_adapted.sh` | Phase 0 冻结（15.45） | 旧协议数字作废 |
| `run_core_ablation.sh` | 核心演化消融 | 旧协议（需 strict-online 重做） |
| `run_c2_sensitivity.sh` / `run_cross_scale.sh` | 宽 TW / 跨规模 | 旧协议 |
| `run_lambda_q_sweep.sh` / `run_constraint_ablation.sh` / `run_coldchain_thermal_eval.sh` | 品质/约束/热物理 | 旧协议 |
| `run_transfer.sh` | zero-shot 迁移 | 旧协议 |
| `run_baseline_audit.sh` / `run_p0_verification.sh` | 基线/P0 审计 | 被 tests/ 替代 |
| `apply_phase_a_fix.sh` / `fix_phase_a_modules.sh` / `run_phase_a1_*.sh` / `test_phase_a_*.sh` | Phase A 边状态 | Phase A 暂停 |
| `migrate_data_paths.sh` | 数据路径迁移 | 一次性，已完成 |

### 旧核心模块

| 文件 | 原用途 | 失效原因 |
|------|------|---------|
| `decoding/cvrptw.py` | 旧 offline 单发解码器 | complete=0%，被 `online_decode.py` 替代 |
| `decoding/maskco_dynamic.py` / `resource_mask.py` / `thermal_state.py` | Phase 2/3 旧模块 | 被 strict-online env + resource_beam 替代 |
| `simulation/rolling_horizon.py` / `run_dynamic_sim.py` | 旧仿真（固定步长） | 被 `strict_online_env.py`（事件驱动）替代 |
| `baselines/solomon_audit.py` / `ortools_weighted_completion.py` / `euro_adapter.py` | 旧基线辅助 | 被 `ortools_rolling_horizon.py`（shared env）替代 |
| `training/train_cvrptw.py` / `train_coldchain.py` / `auto_train.py` | 旧训练入口 | 被 `train_dynamic_cc.py` 替代 |
| `data/generate_cvrptw_data.py` / `generate_self_training_labels.py` / `CVRPTWDataloader.py` | 旧数据工具 | 被 DCC 数据链替代 |
| `analysis/*`（constraint_ablation / edge_influence / mechanism_analysis / toy_instance_test / route_changing / recourse_regret / run_edod_matrix） | Week 4 可解释性 | 旧协议数字，需 strict-online 重做 |

---

## 目录结构

```
scripts/
├── README.md                      ← 本文档
├── models/                        # 模型定义（DynamicColdChainModel + cvrptw_utils）
├── data/                          # 数据生成 + 加载（generate_coldchain_data + ColdChainDataloader + regenerate_baseline）
├── training/                      # 训练（train_dynamic_cc）
├── decoding/                      # strict-online 解码（online_decode + run_online_decode + run_r1_5_model_utility + run_r1_eval + resource_beam）
├── simulation/                    # 共享事件引擎（strict_online_env）
├── evaluation/                    # 权威评估器（authoritative_evaluator）
├── baselines/                     # OR-Tools-RH（ortools_rolling_horizon）
├── tests/                         # 13 blocking tests
├── lib/                           # C++ 扩展（离线 2-opt 用，strict-online 主路径已不用）
│
├── run_typed_retrain.sh           # ★ 训练入口
└── regenerate_archive.py          # ★ 代码归档生成
```

---

## 后续结构处理规则

1. P0-M 路径兼容与 C0 旧物理口径隔离已完成；后续不顺带整理无关历史脚本。
2. 历史脚本如需归档，先生成引用清单，再单独迁移并修正文档链接。
3. `lib/` 是否仍被复现入口使用必须以静态引用和 smoke 结果判定，不能仅凭旧说明归档。
