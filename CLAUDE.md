# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MaskCO is the official ICLR 2026 implementation of a Neural Combinatorial Optimization (NCO) framework using masked generation. Three NP-hard problems: **TSP**, **CVRP**, **MIS**.

The `C-VRP_Cold-chainVehicleRoutingProblem/` subdirectory extends MaskCO from standard CVRP to **CVRPTW** (Time Windows), **Cold-Chain** (temperature), and **Dynamic Cold Chain (DCC)** (reveal_time + EDoD). This is an isolated workspace — **never modify original MaskCO source files**.

## ⚠️ 当前状态（2026-09-03）：论文主线锁定为 MaskCO → 动态冷链；当前 C0 冷链闭环

> **唯一状态入口**：`C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md`。本文件保留工程规则和历史背景；若研究进度或权威路径冲突，以 `项目当前状态.md` 为准。

> **v4 运营契约**：主实验是 dynamic cold-chain pickup-to-depot，不是 depot-to-customer 配送。车辆空载出发，`service_finish` 取货入舱并开始品质计时，`return_arrival` 卸货并关闭；同构多温舱、共享总容量、单次行程、无 reload。后续 C0 实现必须包含订单级 cargo manifest。

> **本文件以下大部分是 2026-08-19 冻结的旧叙事。2026-08-26~09-01 做了 P0 协议审计 → R1.7 归因 → B0 → JF1 → JF2（exact-vehicle No-Go）→ HFR-M0（Gate A FAIL/F3），旧数字全部作废。**

**核心证据链（R1 EDoD=0.5，strict-online）**：

| 方法 | `distance_cost` | complete |
|------|------|----------|
| OR-Tools-RH（OR-joint） | 23.04 | 100% |
| **JF1-H（heuristic joint）** | **24.50** | 100% |
| JF1-R（real logits joint） | 26.84 | 97.7% |
| model guarded (H2-G) | 27.10 | 100% |
| NN (H1) | 27.66 | 100% |
| model raw (H2) | 28.20 | 100% |
| JF1-S（shuffle joint） | 29.10 | 96.1% |
| model_shuffle (H3) | 30.19 | 100% |

**演进链**：exact-vehicle CE（JF2-M0-v2）Gate A FAIL（25.91 vs 24.50）→ 2×2 oracle 证明 partition×sequencing 交互（interaction **−1.7430**，P0-R 修复后复核 2026-09-02，CI [−2.387, −1.104]；B0：G_fleet 17.891% / G_seq −1.227% 亦确认）→ **HFR-M0（Hierarchical Fleet–Route Masked Reconstruction）完整实现（Step 1-5），Gate A FAIL（F3）**。

**HFR-M0 训练端 Sanity Gate 全 PASS**：G 学到 partition 信号（AUROC 0.727 / AUPRC 0.667）；A_joint 优于 A_base（L_A 13.57→8.92）；real G >> shuffle G（8.92 vs 29.56，耦合 active）。

**HFR-M0 Gate A（VAL128，7 变体消融，JF1-H=24.50）**：

| 变体 | partition | insert | cost | Δ |
|------|-----------|--------|------|---|
| g_only | G | min-travel | **24.97** | +0.47 |
| min_travel（骨架） | travel | travel | 25.92 | +1.38 |
| a_only | travel | A_joint | 26.63 | +2.12 |
| full | G | A_joint | 29.95 | +5.41 |
| group_only | G | ΔA+logσ(G) | 30.19 | +5.69 |
| group_logit | G | 纯 logσ(G) | 30.07 | +5.52 |

**方向定论（F3 坐实）**：
> HFR 训练信号可学但未转化为部署效用，说明旧 structural target 与可执行动作错位；这不证明所有 route signal 无用，也不否定 MaskCO。v4 主线固定为 **DynMaskCO-CC：基于 MaskCO 的动态冷链效用对齐掩码重构**。Cost-aware preference 是内部监督机制；当前先完成 C0 冷链状态/单位/trace evaluator，再做 O0-D/O0-CC，最终 M1 必须保留 event mask、masked reconstruction 与 iterative refinement。

**权威入口**：
- `C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md` — 当前状态与导航
- `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/执行进度表.md` — 当前阶段、Gate 与下一动作
- `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/` — 论文主控、执行契约与评价口径

**关键脚本（HFR-M0）**：
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/expert/build_teacher_routes.py` — HFR Step 1：OR-joint 完整 routes → G/A 监督
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/hierarchical_fleet_route.py` — HFR Step 2：GroupingHead + RouteResidualAdapter
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_hfr_m0.py` — HFR Step 3：frozen base + 训练
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/hfr_replanner.py` — HFR Step 4：FleetSlot + joint GA score
- `C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_hfr_gate_a.py` — HFR Step 5：Gate A eval（partition/insert 消融）
- （JF2 遗留：`joint_fleet.py` / `fleet_features.py` / `fleet_assignment_head.py` / `jf2.py` / `train_fleet_head.py` / `run_jf2_gate_a.py`）

**R1 诊断历史（2026-08-28，已被取代）**：旧 strict-online 曾发现「模型输出与 EDD 贪心逐位相同」，根因 R1-R4——经 P0 协议修复后模型 active（61% divergence）；R1.7 证明 learned preference 有独立价值；B0 证明 fleet allocation 主导；JF1 证明 joint topology −11.4% 收益；JF2/HFR-M0 证明 exact-vehicle 与 route 结构监督都无 online utility（F3）。

## 当前定位（2026-08-19 最终冻结）

**核心研究问题**：研究掩码生成范式如何从静态组合优化扩展到动态冷链车辆路径问题中的因果、可行性保持在线优化。

*"DynMaskCO investigates how masked generation can be extended from static combinatorial optimization to causal, feasibility-preserving online optimization for dynamic cold-chain vehicle routing."*

**三层贡献**（按原始课题排序）：

1. **从 Static MaskCO 到 Causal Dynamic Masked Generation**（核心）—— Visibility-Gated Attention（P0-2）+ Visible-Only Normalization（P0-3a）+ Event Masking（D1）+ Frozen Prefix（D2）+ Online Seq（Phase 3c）。核心消融（3 类型一致）：Model-only 0% feas + ~20 viol → +Resource Beam −90% viol → +EDD 100%。
2. **动态硬约束下的 Feasibility-Preserving Reconstruction** —— Hybrid NCO 架构（learned masked generation + resource-feasible search），45/45 = 100% 可行。
3. **Cold-Chain 多资源约束扩展与机制验证** —— Toy 因果翻转（weighted completion time 机制成立）+ R1 弱效应（硬 TW 主导，诚实报告）。

> ⚠️ **历史叙事（已被 2026-09-03 v4 覆盖）**：曾将冷链降为应用扩展；当前主线已重新锁定为基于 MaskCO 的动态冷链优化。

**历史入口**：`C-VRP_Cold-chainVehicleRoutingProblem/archive/docs/优化历程/v1/代码核查与修正.md`（P0 修复记录）+ `C-VRP_Cold-chainVehicleRoutingProblem/archive/docs/旧版总览/v1/完整代码归档.txt`（历史源码快照）。当前动作仍以文件顶部的唯一状态入口为准。

## 核心成果

**DynMaskCO v1 Frozen** — Phase 3c K=5 online seq + K=16 beam + TW 2opt: **100% TW feas + 0 violations, 45/45**（5 train seeds × 3 types × 3 EDoDs）。Avg cost **14.77**（全矩阵平均；R1 EDoD=0.5 单独 17.40）。Causal model（coord_normalize_visible）比 leaky model 便宜 **8.8%**。

**typed embedding + 边特征**（2026-08-18）: `temp_embed`(3)→`type_embed`(5，修复 temp_class 截断 bug) → 14.79（+0.1% 零退化）；再 + 边特征 `energy_mat` → **14.08（-4.8%）**。checkpoint 见 `ckpts/p0_fix/typed_v1{,_edge}/phase3c/`。

**核心演化消融**（2026-08-19，论文第一贡献证据）: Model-only（原始 MaskCO）feas 0% + viol 22.2 → +Resource Beam（viol 22.2→2.1，**-90%**）→ +EDD（feas 100%）→ +2opt（cost 微调）。证明「learned masked generation + resource-feasible search」hybrid 架构——模型提供路由偏好，resource-state decoding 保证硬约束。

**历史品质机制观察（C0 前口径）**：toy、R1 与 R2 数字只作为调试线索；由于热学单位、品质累计语义和执行轨迹评估器尚未闭环，不能作为当前机制结论。冷链现为论文主线，必须由 C0 在统一口径下复核。

## 诚实报告

严格 frozen-prefix non-anticipatory 对比显示 **OR-Tools Rolling Horizon 全面优于 DynMaskCO**（R1 −16.5%, C1 −37%, RC1 −21%; 100-node 仍 feasible）。「在线 SOTA」和「可扩展性」叙事已废弃。详见 `docs/调研/GPT/v5/执行方案_v5.md`。

RRNCO（ICLR 2026 learned 基线）对比同样**交叉而非一边倒**：RRNCO clairvoyant（R1 11.88）远强于 DynMaskCO 底层（~17），naive 特征屏蔽下随 EDoD 崩坏 +60%（15.38→24.67），DynMaskCO 仅 +5%（17.12→17.95）。低动态 RRNCO 赢、高动态 DynMaskCO 赢，但这只支撑「causal 架构鲁棒性」，**不**支撑「DynMaskCO 路线更优」——给 RRNCO 正经非预知方案（rolling horizon）大概率全面优于 DynMaskCO。详见 `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/基线对比.md` §3.2。

## Setup

```bash
sh install.sh          # JAX 0.5.0, Flax 0.10.4, Triton 3.1.0, PyTorch CPU, NumPy 1.26.4
cd lib && make         # Parent C++ extension (2-opt, insertion)

# CVRPTW C++ extension (EDD repair + TW-aware 2-opt)
pip install pybind11
cd "C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib" && make
```

Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (training) or `0.5` (inference). GPU 0 is default (31GB free).

## Server Environment

- **Path**: `/home/hzeng/project/MASKCO-Main/`
- **Conda**: `/home/hzeng/envs/MASKCO_env/` (Python 3.10.19)
- **GPU**: 2× RTX 5090 (32 GB), use GPU 0 (always free). `CUDA_VISIBLE_DEVICES=0` for decoding, `--gpu_id 0` for training.
- **Workflow**: Write locally → MobaXterm SFTP upload → SSH run
- **⚠️ SSH instability**: MobaXterm SSH connections drop frequently, killing foreground jobs. For any run >5min, use `tmux` (NOT `nohup`, which doesn't reliably survive the drop):
  ```bash
  tmux new-session -d -s freeze 'bash C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh eval > /tmp/freeze_eval.log 2>&1'
  tmux ls                                    # verify running
  tail -f /tmp/freeze_eval.log               # view progress (no attach needed)
  tmux attach -t freeze                      # attach; detach with Ctrl+B then D
  ```

## Architecture

### Parent Project (MaskCO)

Encoder-decoder transformer (Flax NNX). Encoder → Decoder + timestep + partially-masked adjacency matrix as attention bias → edge predictions. Training: random mask edges in solutions, CE loss. Inference: iterative mask-and-reconstruct + C++ 2-opt.

### CVRPTW / Cold-Chain / DCC Extension

Model chain: `TSPModel → CVRPModel → CVRPTWModel → ColdChainModel → DynamicColdChainModel`

| File | Input | Key Features |
|------|-------|-------------|
| `models/CVRPTWModel.py` | 5D (x,y,demand,tw_start,tw_end) | +tw_bias, config: `softcap_fn` (256d), `softcap_fn_512` (512d) |
| `models/ColdChainModel.py` | 6D (+temp_class) | Inherits CVRPTWModel |
| `models/DynamicColdChainModel.py` | 7D (+reveal_time) | **type_embed（5 类：depot/常温/冷藏/冷冻/未来订单）**（2026-08-18 替代 temp_embed，修复 temp_class 截断 bug）+ **edge_weight**（`energy_mat` 边特征，`--use_edge_feat`）, **Visibility-Gated Attention (P0-2)**: `encode(raw_features, visible_mask)` → attn bias `-1e9` for future nodes |
| `models/cvrptw_utils.py` | — | **P0-3a**: `coord_normalize_visible(inputs, visible_mask)` — visible-only normalization statistics. Fixes the leak in parent `coord_normalize` (which used ALL nodes' mean/norm). |

| File | Purpose |
|------|---------|
| `data/generate_cvrptw_data.py` | Solomon CVRPTW generator (R1/C1/RC1/R2/C2/RC2) |
| `data/generate_coldchain_data.py` | Cold-chain + DCC generator. `--edod 0.5` for dynamic. `--temp_dist "0,0,1"` for 温区分布控制（全冷冻=温度驱动验证）. **P0-4**: +`quality_loss` (Arrhenius), +`energy_mat` (refrigeration) |
| `data/CVRPTWDataloader.py` | 5D loader, pure NumPy |
| `data/ColdChainDataloader.py` | 6D/7D/8D auto-detecting loader. **P0-2**: masks future order features (coords→0.5, others→0). **P0-4**: loads `quality_loss` as optional 8D. Yields `(features, routes, timestep, visible_mask)` |
| `training/train_cvrptw.py` | CVRPTW training. `--encoder_input_dim`, `--tw_loss_lambda` |
| `training/train_coldchain.py` | ColdChain training. **Must add encoder_input_dim override before construct_model**. Adapts to 4-tuple dataloader output |
| `training/train_dynamic_cc.py` | DCC training. `--masking_mode` (random/spatio_temporal/targeted/adaptive). `--online_seq_training --online_seq_steps 5` (Phase 3c K=5). `--seed` (default 42, added 2026-08-12). **P0-2**: passes `visible_mask` to `m.encode()`. **P0-3a**: uses `coord_normalize_visible`. `--spoilage_lambda` still accepted but not used (deprecated) |
| `training/auto_train.py` | Auto GPU-detect → smoke 50 steps → full training. Uses `/bin/bash` |
| `decoding/cvrptw.py` | **Main decoder**. Auto-detects C++ ops, feature dims, model types. **P0-2**: masks future order features + passes `visible_mask`. **Phase 2 (D1-D6)**: constraint-aware decoder, event-aware mask-reconstruct, frozen prefix, adaptive mask, anytime solver. **Phase 3**: `--enable_resource_decoder --beam_width 16` (K-beam). CLI flags: `--enable_event_mask`, `--enable_frozen_prefix`, `--enable_constraint_decoder`, `--enable_adaptive_mask`, `--enable_anytime` |
| `decoding/resource_beam.py` | **Phase 3a core** — K-beam resource-state decoder: `BeamState` tracks route/arrival/load/visited; `_can_go()` checks TW+capacity+round-trip+2-step lookahead. **品质感知 score**: `--enable_quality --lambda_q` → beam score -= λ_q × weighted completion time（∑K_i·t_i 边际贡献，2026-08-19 修正方向）。支持 `dist_mat` 手动传入（toy 验证用）。 |
| `decoding/resource_mask.py` | Per-edge TW+capacity feasibility mask (v1→v3). Superseded by resource_beam for decoding. |
| `decoding/thermal_state.py` | **Phase 3d** — endogenous thermal physics: Newton cooling + refrigeration + door shock + Arrhenius decay per-beam. **COP 温度函数 (v5)**: `compute_cop(ΔT)`（非常数，电制冷拖车论文）. |
| `decoding/learnable_mask.py` | **Phase 3b** — REINFORCE mask policy (Gumbel-Top-K, MaskPolicyTrainer). Honest ablation. |
| `decoding/coldchain.py` | ColdChain/DynamicColdChain decoding wrapper |
| `decoding/maskco_dynamic.py` | **Phase 2 core module** — D1-D6 unified implementation: `dynamic_mask_reconstruct()`, `compute_affected_segments()`, `compute_adaptive_keep_rate()`, `AnytimeScheduler`, `SequentialDynamicSampler`, `apply_constraint_mask()` |
| `baselines/solomon_audit.py` | **P0-2** — OR-Tools + PyVRP 三层 Solomon sanity check. Proves OR-Tools adapter correct (J*=9.1 on DCC). |
| `baselines/ortools_rolling_horizon.py` | **P0-2 强动态基线** — OR-Tools frozen-prefix rolling-horizon（non-anticipatory）。关键发现：全面优于 DynMaskCO（16-37%） |
| `baselines/euro_adapter.py` | **P0-4** — DynMaskCO → EURO Meets NeurIPS 2022 D-VRPTW competition protocol adapter. |
| `simulation/run_dynamic_sim.py` | Rolling-horizon dynamic simulation. **P0-2**: replanner masks future orders + `encode_fn` accepts `visible_mask` |
| `simulation/rolling_horizon.py` | Event-driven simulator: Order states (unknown→known→assigned→in_transit→completed), clock-based revelation |
| `analysis/recourse_regret.py` | **P0-3** — Conditional Recourse Regret 计算（clairvoyant J_e*，待接入 J_e^π） |
| `tests/test_no_future_leakage.py` | **P0-2 verification**: 3 tests (dataloader masking, gradient isolation, attention bias math) |
| `tests/test_future_cardinality.py` | **P0-3a** — future identity invariance + cardinality leakage |
| `tests/test_target_label_leakage.py` | **P0-3b** — target adjacency future-edge leakage (vis-only Δ=0) |
| `tests/test_data_qc.py` | **P0-5** — data QC (NaN/范围/重复/TW sanity) |
| `scripts/run_method_freeze.sh` | **P0-5 canonical** — 训练 + 评估一键复现（`smoke`/`phase3c`/`full`/`eval`/`env`） |
| `scripts/run_phase_e.sh` | One-click pipeline: data gen → train → eval → summary. `smoke` (1280/500) or `full` (12800/50000) |
| `lib/` | C++ extension: `cvrptw_ops.{hpp,cpp}` + `cvrptw_bindings.cpp`. EDD (0.3ms) + TW 2opt (0.27ms). **P0-4**: +`cvrptw_quality_cost`, +`cvrptw_quality_two_opt` |

Full file index: [`scripts/README.md`](C-VRP_Cold-chainVehicleRoutingProblem/scripts/README.md)

### Data Format (.npz)

```
coords:       (N, nodes, 2)  float32
demands:      (N, nodes)     int32
tw_start/end: (N, nodes)     float32
service_time: (N, nodes)     float32
temp_class:   (N, nodes)     int32     — ColdChain/DCC [0,1,2]
reveal_time:  (N, nodes)     float32   — DCC only (0=known, >0=future)
visible_mask: (N, nodes)     float32   — 1=visible at t=0, 0=future (P0-2)
quality_loss: (N, nodes)     float32   — Arrhenius quality decay (P0-4)
energy_mat:   (N, nodes, nodes) float32 — refrigeration energy cost (P0-4)
routes:       (N, pad_len)   int32     — depot=0 separators
opt_costs:    (N,)           float32
```

## CC_Compare（对比方法复现工作区）

复现「同赛道 learned 方法」并改成 DCC-VRP 协议做公平对比（对应 `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/基线对比.md` §2.3）。**一个方法一个文件夹**，位于根目录 `CC_Compare/`，是独立于 `C-VRP_Cold-chainVehicleRoutingProblem/` 的第二个工作区。

| 文件夹 | 方法 | 出处 | 状态 |
|--------|------|------|------|
| `CaDA/` | Constraint-Aware Dual-Attention | ICML 2025 | ✅ 可适配（VRPTW 变体） |
| `RRNCO/` | Real-World NCO | ICLR 2026 | ✅ 已跑（clairvoyant + non-anticip） |
| `PIP-constraint/` | 复杂约束 Lagrangian | NeurIPS 2024 | ⛔ 不可适配（单车辆 TSPTW，缺容量+多车辆） |
| `MAPT/` | 多车动态取送货 | AAAI 2026 | ❌ 无源码 |

**适配约定**（每个方法在 `dcc_vrp/` 下放一套不改原始代码的脚本）：
- `dcc_data.py` — DCC .npz → 该方法输入（时间尺度对齐 + demand 归一 + 可见性）。
- `dcc_env.py` — 环境门控（non-anticipatory：`--mask_future`）。
- `run_dcc.py` — 评估 harness（报告 cost=纯距离 / TW Feas / Cap Feas / complete / gap）。
- `README.md` — checkpoint 下载 + 运行命令。

**关键规则**：只改 `dcc_vrp/` 适配脚本，**绝不改原始仓库代码**（如 `RRNCO/rrnco/`、`CaDA/50/`）。non-anticipatory 语义 = 未来节点特征屏蔽（coords→0.5，与 DynMaskCO P0-2 一致）+ env 用 `_true_*` 真值做可行性/奖励（serve 全部 50，不早停），等价 DynMaskCO 的「causal encoder + full-service decoder」。

**RRNCO 结果**（已跑完，见 `基线对比.md` §3.2）：clairvoyant R1 11.88 / C1 6.93 / RC1 10.24（EDoD 不变）；non-anticipatory（naive 特征屏蔽）随 EDoD 崩坏 +60%（R1 15.38→24.67），与 DynMaskCO（+5%）**交叉**——低动态 RRNCO 赢、高动态 DynMaskCO 赢。⚠️ 这是「鲁棒性」对比不是「更优」对比：RRNCO 底层模型远强（clairvoyant 11.88 vs DynMaskCO ~17），给它正经 rolling-horizon 非预知方案大概率全面优于 DynMaskCO（与 OR-Tools-RH 同一逻辑）。

环境：`cc_compare` conda env（RTX 5090 需 torch≥2.7）。建环境 `bash CC_Compare/setup_envs.sh cc`（CaDA+RRNCO 共用）。CaDA 原始 pin 旧 torchrl 0.1.1，需 `_torchrl_compat.py` 补丁。

## Key Commands

### Batch Experiment Scripts

```bash
# P0 Protocol Audit (run these first)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_p0_verification.sh"        # P0-2+P0-3a+P0-3b 一键审计
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_baseline_audit.sh"         # OR-Tools/PyVRP Solomon sanity

# DynMaskCO v1 Frozen Reproduction (THE canonical entry point)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" smoke    # 冒烟 (~10min, 1 seed 2K steps)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" phase3c  # Phase 3c only (~5h, 5 seeds)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" full     # 完整复现 (~24h, 5 seeds)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" eval     # 仅评估已有 ckpt
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" env      # 环境快照

# Legacy / supplemental
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_phase_e.sh" full           # 全流程: 数据→训练→评估
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_mixed_edod.sh" full         # 混合EDoD训练
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_baselines.sh"               # ALNS vs Greedy vs MaskCO
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_ablation.sh"                # P0-3 四层消融
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_keep_rate_sweep.sh" smoke   # 方向C: keep_rate×EDoD sweep（greedy对照）
```

### Training

```bash
# CVRPTW 50-node
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_cvrptw.py" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 5 --peak_lr 1e-3 --batch_size 64 \
    --num_steps 50000 --save_interval 5000 --data <train.npz> \
    --logdir ... --savedir ... \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None

### DCC 7D + ST-mask (fixed GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
    --num_steps 50000 --save_interval 5000 --data <dcc_train.npz> \
    --masking_mode spatio_temporal \
    --logdir ... --savedir ... \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
```

# Phase 3c K=5 online seq training (strongest model)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
    --num_steps 50000 --save_interval 5000 --data <train.npz> \
    --masking_mode spatio_temporal --online_seq_training --online_seq_steps 5 \
    --logdir ... --savedir ... \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None

# Beam decode evaluation (Phase 3, strongest decoder)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py" \
    --capacity 50 --penalty 3. --data <test.npz> --ckpt <phase3c.ckpt> \
    --keep_rate 0.3 --batch_size 8 --runs 1 --cycles 1 --sampling_steps 1 \
    --two_opt_steps 4 --seed 42 --gumbel_scale_factor 0. --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 16 --enable_tw_aware_2opt_py

# Auto-train (GPU detect + smoke + full)
python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/auto_train.py" \
    --script train_dynamic_cc.py \
    --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
    --num_steps 50000 --save_interval 5000 --data <train.npz> \
    --logdir ... --savedir ... \
    --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
```

### Evaluation (Best Configuration)

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py" \
    --capacity 50 --penalty 3. --data <test.npz> --ckpt <ckpt> \
    --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
    --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
    --threads_over_batches 1 \
    --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py

# P0-4: quality-aware 2-opt (add these flags)
    --enable_quality_2opt --lambda_q 0.1 --lambda_e 0.01
```

### Self-Training Labels (方向一)

```bash
# 用最佳模型生成伪标签
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_self_training_labels.py" \
    --data <train.npz> --ckpt <best_ckpt> --output <st_labels.npz> \
    --batch_size 8 --runs 8 --cycles 320
# Then fine-tune on st_labels.npz with train_dynamic_cc.py
```

### EDoD Matrix Evaluation

```bash
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/run_edod_matrix.py"
# Outputs 9-line summary: R1/C1/RC1 × EDoD 0.2/0.5/0.8
```

### Data Generation

```bash
# Standard CVRPTW
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_cvrptw_data.py" \
    --problem_size 50 --num_instances 1280 --type R1 --capacity 50 --output <out.npz>

# ColdChain with EDoD (0.2/0.5/0.8)
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
    --problem_size 50 --num_instances 1280 --type R1 --capacity 50 \
    --edod 0.5 --output <out.npz>
```

### P0-2 Leakage Verification

```bash
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_no_future_leakage.py" \
    --data <dcc_test.npz> --ckpt <step50000.ckpt>
# 3 tests: dataloader masking, gradient isolation, attention bias math
```

### C++ Extension

```bash
cd "C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib"
make clean && make
```

## Decoding Flags

| Flag | Effect |
|------|--------|
| `--enable_tw_filter` | Filter candidate edges by TW compatibility |
| `--enable_tw_repair_edd` | EDD sort within route segments (C++ auto-detected) |
| `--enable_tw_aware_2opt_py` | TW-constrained 2-opt (C++ auto-detected) |
| `--enable_quality` | 品质感知 beam score（weighted completion time ∑K_i·t_i）: beam score -= λ_q × 品质损耗 |
| `--quality_salable_threshold` | 不可售品质损耗阈值（Num 指标，默认 0.1 对本数据偏高，用 0.02） |
| `--save_routes` | 保存最终路线到 .npz（路线分析用，`analysis/mechanism_analysis.py` 等） |
| `--tw_margin` | Phase 3a: TW 安全裕度 (default 0.05, 0=无裕度) |
| `--lambda_q` | Quality loss weight (default 0.1) |
| `--enable_event_mask` | D1: event-driven local reconstruction (only mask affected segments) |
| `--enable_frozen_prefix` | D2: freeze executed route prefix (immutable) |
| `--enable_constraint_decoder` | D3: decoder-native TW constraint (logit=-inf for illegal edges) |
| `--enable_adaptive_mask` | D4: feature-adaptive per-node mask probability |
| `--enable_anytime` | D6: time-budget-aware cycle scheduler |
| `--time_budget_ms` | D6: total time budget in ms (default 200) |
| `--frozen_prefix_len` | D2: number of prefix positions to freeze |
| `--enable_tw_attn_bias` | ⚠️ Conflicts with preserving 2opt, don't combine |
| `--enable_tw_preserving_2opt` | ⚠️ 100% revert rate, deprecated |

## Critical Rules

1. **Never modify original MaskCO files** — all new code in `C-VRP_Cold-chainVehicleRoutingProblem/`
2. **No `__init__.py` in CVRPTW subdirectories** — imports via `sys.path.insert` to avoid shadowing parent packages
3. **`--gpu_id` must be parsed before `import jax`** — raw `sys.argv` scan sets `CUDA_VISIBLE_DEVICES`
4. **`tw_max` auto-detected** — `tw_end.max()` in both training and decoding. Must be consistent.
5. **C++ RNG seed** — incrementing per call for diversity across cycles/threads
6. **`encoder_input_dim` must be overridden on model_config before construct_model** — `train_coldchain.py` missed this initially (fixed)
7. **`auto_train.py` uses `executable='/bin/bash'`** — `/bin/sh`=dash doesn't support `source`
8. **Dataloader yields 5-tuple, UNMASKED features** — `ColdChainDataloader` yields `(features, routes, timestep, visible_mask, reveal_time)`. Features are **unmasked**（掩码移到训练器按 vis_k 逐步做，2026-08-27 起，见 §当前状态）。All consumers must unpack 5 values.
9. **Python 3.10 f-string limitation** — f-strings cannot nest same quote type. Use `%` formatting or variables for complex format strings in shell-embedded Python.
10. **Use `coord_normalize_visible`, not `coord_normalize`** — parent project's `coord_normalize` computes statistics over ALL nodes (axis=-2), leaking future coordinates. `cvrptw_utils.coord_normalize_visible()` uses only visible nodes. All training/decoding/simulation must use the visible-aware version.
11. **cost semantics unified (跨方法指标一致性)** — all methods' reported `cost` = **pure travel distance, no penalty**. Infeasibility is reported via separate TW Feas / Cap Feas metrics, never mixed into cost. `_eval_distmat_cost` (final report) is pure distance; `_eval_single_route_cost` (2-opt internal only) has capacity soft-penalty. See `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/评估口径.md` §1.0 and `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/基线对比.md`.

## Key Findings

### What Works
- **Causal Normalization (P0-3a)**: `coord_normalize_visible` → model no longer sees future nodes' coordinate statistics. Cleaner prior → cost -8.8% vs leaky
- **历史 Quality-Aware Beam 观察**：toy/R1/R2 数字来自 C0 前的多套品质口径，只作调试线索；需由唯一 cold-chain trace evaluator 复核后才能形成机制结论。
- **Transferable Representation (支持证据)**: 纯 R1 训练 → zero-shot C1/RC1 零迁移损失（C1 差 0.2%, RC1 差 0.6%）。证明 masked representation 未只记忆单一分布，支持第一贡献（不是论文主题）。
- **COP 温度函数 (v5)**: 从常数 2.5 改为 `compute_cop(ΔT)`（电制冷拖车论文：COP 随温差变化）。
- **Self-Training**: feas +20pp (73→93%), largest single lever. Generate pseudo-labels with best model → fine-tune
- **EDD repair**: viol 9→2 (-77%), highest-leverage single change
- **TW-aware 2opt (C++)**: enables mask-reconstruct iteration, 30,000x speedup
- **Spatio-Temporal masking**: feas +10pp vs random (64→74%), simulates dynamic info scarcity
- **Temperature constraint**: improves TW feas (50→73%), constraint = inductive bias
- **Mixed training (R1+C1+RC1)**: C1 feas 7→63%, essential for generalization
- **Temp Embedding**: clean replacement for harmful spoilage_bias (no performance loss)
- **Visibility-Gated Attention (P0-2)**: `visible_mask` → encoder attn bias `-1e9` for future nodes. Zero parent-project changes. Oracle→Causal gap: **76.6%→51.2% (−25.4pp)** — paper's core evidence: the cost of non-anticipatory decisions.
- **Mixed EDoD Training (方向A)**: R1+C1+RC1 × 0.2+0.5+0.8 joint training → Causal feas **74.7%** (+23.5pp vs single-EDoD 51.2%). EDoD=0.8: **28.8%→99.3%** (3×).
- **8D Quality-Aware (P0-4)**: Adding `quality_loss` as encoder feature → **77.9%** (+3.2pp vs 7D). EDoD=0.5 hardest-case: **54.9%→66.7%** (+11.8pp). Quality signal helps prioritize perishable nodes at medium dynamism.
- **Quality-Aware C++ 2-opt (P0-4)**: `cvrptw_quality_two_opt` with `total = dist + λ_q×quality_loss + λ_e×energy`. Verified λ_q sweep [0,2.0] — reference sol is locally optimal (dist unchanged), enables tradeoff selection during inference.
- **Temperature Trajectory Tracking (P0-4)**: `rolling_horizon.py` logs TTI (Time-Temp Integrator), energy (kWh-equiv), quality loss rate per time-step. static TTI=428 vs full_reopt=52 (8×), Energy 19.2 vs 2.2 (9×) — event-driven replanning dramatically reduces cold-chain cost.

### P0-2 Results: Causal Dynamic Routing (50K steps, 3 seeds) — OBSOLETE (replaced by P0-3a causal retraining)

These results used leaky `coord_normalize` (full-node statistics). See [Causal v1 Results](#p0-3a-causal-v1-results-2026-08-12) below for corrected numbers.

| Stage | Avg EDoD=0.2 | Avg EDoD=0.5 | Avg EDoD=0.8 | Overall | vs Oracle |
|-------|-------------|-------------|-------------|---------|-----------|
| Oracle (泄漏) | 76.3% | 75.0% | 78.4% | 76.6% | — |
| Single-EDoD | 57.4% | 67.4% | 28.8% | 51.2% | −25.4pp |
| **Mixed-EDoD (7D)** | **69.9%** | 54.9% | **99.3%** | 74.7% | −1.9pp |
| **8D Quality (P0-4)** | 67.5% | **66.7%** | **99.6%** | **77.9%** | **+1.3pp** |

### P0-3a Causal v1 Frozen Results (2026-08-13, 5 train seeds)

**coord_normalize_visible fix** — Phase 3c K=5 online seq + K=16 beam decoder.
5 train seeds (42/123/999/2025/2026), fixed decode seed 42, 8 runs × 40 cycles, TW 2-opt 4 steps.

| Type | EDoD=0.2 | EDoD=0.5 | EDoD=0.8 | Avg |
|------|------|------|------|:---:|
| R1 | 17.12 ± 0.06 (100%) | 17.40 ± 0.05 (100%) | 17.95 ± 0.06 (100%) | 17.49 |
| C1 | 11.51 ± 0.06 (100%) | 11.80 ± 0.06 (100%) | 12.15 ± 0.05 (100%) | 11.82 |
| RC1 | 14.67 ± 0.04 (100%) | 14.95 ± 0.06 (100%) | 15.41 ± 0.06 (100%) | 15.01 |
| **Avg** | **14.43** | **14.72** | **15.17** | **14.77** |

**45/45 = 100% feasible, 0 violations.** All std ≤ 0.06 — highly stable across training seeds.

vs OR-Tools Clairvoyant (R1 EDoD=0.5): J*=9.1 → **Clairvoyance Gap = 8.30 (91%)**.

vs old leaky model (Phase 3c beam, same decode config): 16.23 → **causal is 8.8% cheaper**.
Full-node normalization was noise, not useful signal.

**Checkpoints**: `ckpts/p0_fix/causal_v1/step50000.ckpt` (ST-mask, 814s) ·
`ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt` (K=5 online seq, ~56min)
**5-seed Phase3c**: `ckpts/p0_fix/frozen_v1/phase3c/seed{42,123,999,2025,2026}/step50000.ckpt`
**Results CSV**: `logs/p0_fix/frozen_v1/results/full_matrix.csv`

### What Doesn't Work
- **spoilage_bias (Arrhenius)**: -6pp feas, replaced by Temp Embedding
- **TW attention bias**: conflicts with preserving 2opt → numerical overflow
- **TW penalty loss**: training labels already TW-feasible, CE loss implicitly covers
- **LKH3 labels**: can't find feasible solutions for narrow-TW (R1) instances
- **Predictive masking (0.65x keep)**: too aggressive, worse than ST-mask (0.8x)
- **coord_normalize (parent project)**: computes statistics over ALL nodes → leaks future coordinate distribution into visible encodings. Fixed with `cvrptw_utils.coord_normalize_visible()`.
- **PyVRP 0.11**: solver works on 3-node but fails on 25+ node in server env (`.distance()` returns garbage). OR-Tools 9.8 used instead.
- **"Causal > Oracle" claim**: was an artifact of leaky normalize comparing against itself. Retracted.

### EDoD Sensitivity (DCC, 9 groups, Causal P0-2)
- Causal feas drops 25pp vs Oracle (76.6%→51.6%) — quantifies the cost of non-anticipatory decisions
- Non-monotonic: medium dynamic (0.5) is hardest, strong dynamic (0.8) recovers partially
- C1 most robust: clustered data provides spatial prior even with masked nodes (81.0% at EDoD=0.5)
- EDoD distribution mismatch is critical: model trained at 0.5 → cross-EDoD generalization needs mixed training

## Experiment Records

`C-VRP_Cold-chainVehicleRoutingProblem/docs/实验记录/`:
- `01_决策记录.md` — D-001 through D-031 (K-beam, PyVRP, 100-node)
- `阶段二/PhaseE_P0修复.md` — Phase A→E + Phase 2/3/4 完整实验记录
- `阶段二/00_阶段二总结.md` — 方向一~四实验记录 + EDoD全矩阵

`C-VRP_Cold-chainVehicleRoutingProblem/docs/调研/GPT/`:
- `执行方案.md` — **总执行方案（汇总 + 导航）**，详细见 v3
- `v2/第二阶段详细优化方案.md` — Phase 3/4 全部完成, 36/36 feasible, W10 通过
- `v3/执行方案_v3.md` — v3 详细执行方案（协议审计 + 理论化，已完成）
- `v3/P1_理论形式化.md` — **P1 方法形式化：3 命题 + 1 定理（含证明）**
- `v3/深度研究报告.md` — v3 深度诊断报告
- `v4/执行方案_v4.md` — v4 证据链加固（发现 OR-Tools-RH 全面优于）
- `v4/竞争力诚实评估与重新定位.md` — **诚实评估 + 重新定位**
- `v5/执行方案_v5.md` — v5 重新定位（历史）：冷链多资源 + 可迁移表示（已被 8-19 三层贡献定位取代）
- `v5/冷链参数标定清单.md` — 冷链参数标定（文献/关键词，方向三+四准备）
- `v5/模型改进计划.md` — 基于 29 篇论文的模型改进方向（5 类）

`C-VRP_Cold-chainVehicleRoutingProblem/archive/docs/旧版总览/v1/`（2026-08-21 整理为 1 导航 + 5 内容）:
- `总索引.md` — 论文写作导航（文件清单 + 核心数字速查 + 复现命令 + v6 进展）
- `评估口径.md` — 评估协议（KPI 公式 + 统计规范 + §1.0 cost 语义统一声明）
- `基线对比.md` — 跨方法指标对比（obj/gap/time vs feas/cost/clairvoyance gap）
- `实验细节.md` — 完整公式 + 模型架构 + 实验配置
- `理论形式化.md` — 3 命题 + 1 定理（非预知/前缀/可行性/anytime）
- `完整代码归档.txt` — 全部源文件当前内容归档（77 文件，2026-08-19）
> 权威结果表 → `docs/实验记录/实验结果总表.md`（不在 ALL，已移除过时的 results.json/dashboard/html）

`C-VRP_Cold-chainVehicleRoutingProblem/docs/`:
- `项目当前状态.md` — 当前状态、权威入口与下一步
- `实验记录/实验结果总表.md` — **权威结果表（旧 14.77 作废，现以 strict-online VAL128 为准）**

## 历史推进记录：修监督 → JF2 → HFR-M0（F3）→ Cost-Aware Preference

> 本节是 2026-09-01 的历史快照，其中“待办”已被 2026-09-02 完成的 P0-R/P0-S/P0-A/P0-U 取代。当前下一步只看文件顶部的唯一状态入口。

**历史状态（2026-09-01）**：R0 协议修复 ✅ → R1/R1.5/R1.7/B0/JF1 ✅ → JF2（exact-vehicle CE）No-Go ✅ → HFR-M0 Gate A FAIL。严格解释是 structural signal 可学但未建立相对 JF1-H 的 downstream utility。

**顺序**：
1. **R0/R1/R1.5-M/R1.7/B0/JF1** ✅（fleet allocation 主导 + joint topology −11.4%）
2. **JF2（exact-vehicle Fleet Head）** ✅ No-Go（M0-v2 25.91 vs 24.50 FAIL）
3. **HFR-M0（Hierarchical Fleet–Route）** ✅ 实现完成 + Gate A FAIL/F3（G 有 partition 价值、route 信号有害）
4. **历史计划：Cost-Aware probe**（v4 中降为 M0 表征探针；最终方法改为 DynMaskCO-CC masked recourse）

**待办**：
- [ ] 构造 cost-aware joint-action dataset（`(state, customer, slot, insertion_pos, continuation_cost, regret)` → pairwise preference）
- [ ] 最小 Preference-M0（frozen encoder + Utility Preference Head + pairwise logistic loss），Gate：`CostAware-M0 < JF1-H`（100% service）
- [ ] 和导师决定 Go/No-Go（2026-10-01）

**Server run (P0 protocol audit)**:
```bash
# P0-2 Baseline audit (OR-Tools + PyVRP Solomon sanity)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_baseline_audit.sh" smoke   # or full

# P0-3a Cardinality / identity leakage
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_future_cardinality.py" \
    --data "C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz" \
    --ckpt "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt"

# P0-3b Target-label leakage
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_target_label_leakage.py" \
    --data "C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_mixed_edod_train.npz" \
    --ckpt "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt"
```

**新实验脚本（2026-08-19 新增）**：
```bash
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_core_ablation.sh" r1|c1|rc1  # 核心演化消融
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_c2_sensitivity.sh"            # C2/R2 宽 TW sensitivity
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_cross_scale.sh"               # 50→100-node scale transfer
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_typed_retrain.sh" full        # typed embedding 5-seed 重训（USE_EDGE=1 边特征）
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_lambda_q_sweep.sh"            # 品质感知权重扫描
python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/toy_instance_test.py"  # ③ toy 因果验证
python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/mechanism_analysis.py" # ② 机制验证
python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/route_changing.py"     # ① route-changing
python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/baselines/ortools_weighted_completion.py"  # ④ matched-OR
```

## Phase 2: Dynamic-Aware MaskCO (2026-08-06)

Six directions in `decoding/maskco_dynamic.py` (now superseded by Phase 3 beam decoder):

| # | Direction | Status |
|:---:|------|:---:|
| D1 | Event-aware Masking | ✅ |
| D2 | Frozen Prefix | ✅ |
| D3 | Constraint-aware Decoder | ✅ (superseded by resource beam) |
| D4 | Adaptive Mask Ratio | ✅ |
| D5 | Online Seq Training | ✅ (= Phase 3c) |
| D6 | Anytime Solver | ✅ |

Paper positioning: **"Adaptive Masked Generation for Online Vehicle Routing"** — Cold Chain as application, not title.

## Phase 3: K-Beam Resource-State Decoder (2026-08-07, Complete)

Milestone: **model_only 0% → K-beam 100% feasible** (first time a neural solver hits 100% on narrow-TW). Evolution:

- **3a** K-beam resource-state decoder (each beam tracks time+load+visited; first model_only >0%)
- **3b** REINFORCE mask policy (honest ablation, zero signal)
- **3c** K=5 online seq training (only method that reduces beam cost, −3.8%)
- **3d** Endogenous thermal physics (Newton cooling + refrigeration + door shock + Arrhenius)

**100-node scale**: 9/9 feasible, sub-linear cost growth (+89% for 2× nodes). C1 maintains best cost.

⚠️ The 50-node "16.23" figure from this phase was the **leaky model** — superseded by causal v1 (14.77, see Key Findings §P0-3a).

## Phase 4: Four-Direction Polish (2026-08-08~09, Complete)

| Direction | Result |
|:---:|------|
| 1 Thermal-beam | Dist −29%, Viol −68% |
| 2 Beam+2opt | 100% feas, cost −0.3% (beam already near-optimal) |
| 3 100-node scale | 9/9 feasible (R1 39.37, C1 21.93, RC1 32.01) |
| 4 Baseline audit | ⚠️ PyVRP "0% feasible" was a server env bug (P0-2); OR-Tools solves J*=9.1 |
