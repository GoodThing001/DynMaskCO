# RRNCO-Ordering-RH-D R1 三层身份冻结计划（2026-09-26）

> 前置：R0.5 Stage B 七门全 PASS（run_id 802c3a4e…，checkpoint sha256 `b1ff3191…` 与
> SERVER_ENVIRONMENT 一致）→ 满足 `GO_R1_RRNCO_ORDERING_RH_D` 放行条件。
> 用户授权（2026-09-26「授权」）。冻结只读：不改 `../rrnco/` 上游、不改 `../../common/`、
> 不改 `../../../MASKCO_code/`。任何配置/源码变化必须升 revision 并重冻结。

## 1. 三层身份（镜像 PyVRP-RH-D / OR-Tools-RH-D 冻结模式）

| 层 | 内容 | 载体 |
|---|---|---|
| **compute** | 上游 rrnco commit、checkpoint sha256、依赖版本（torch/rl4co/torchrl/tensordict/lightning/numpy）、LICENSE sha256、R0.5 证据（run_a/run_b/B2/SERVER_ENVIRONMENT） | `identity.py`（环境检查）+ 证据 manifest |
| **control** | `BackendConfig`（t_max=4.6、num_loc=100、normalize=True、sampling_policy=visible_prob_sampling_v1、seed=0、device=cuda）、协调器规则（open-new-vehicle 优先、归一化排名、增量距离、EDD fallback）、证书容差 1e-6、以及实现这些行为的 6 个 adapter 模块源码 | `FROZEN_CONFIG.json` + SOURCE_MANIFEST（control 文件组） |
| **analysis** | 门/核验器源码（run_r0_5、b2_injection_gate、verify_r0_5、server_preflight、r0_5_snapshots、testutil）+ 证据文件 hash | SOURCE_MANIFEST（analysis 文件组）+ `R0_5_EVIDENCE_MANIFEST.json` |

身份链（单向绑定，无循环哈希）：
```
SOURCE_MANIFEST.json（per-file sha256，由 tools/gen_source_manifest.py 可重算）
  ← FROZEN_CONFIG.json（冻结配置 + 依赖 SOURCE_MANIFEST 的 sha256）
    ← FREEZE_SEAL.json（seal 清单：manifest/config/evidence/checkpoint 的 sha256 + 生成者/时间/决策引用）
```

## 2. 文件清单（logical path → 层）

- control：`subproblem.py`、`preference.py`、`coordinator.py`、`pickup_certificate.py`、
  `rrnco_backend.py`、`rrnco_guided_adapter.py`
- analysis：`run_r0_5.py`、`b2_injection_gate.py`、`verify_r0_5.py`、`server_preflight.py`、
  `r0_5_snapshots.py`、`testutil.py`
- 证据（results/）：`SERVER_ENVIRONMENT.json`、`B2_RESULT.json`、`r0_5_run_a.json`、
  `r0_5_run_b.json`、`R0_5_EVIDENCE_MANIFEST.json`
- identity：`identity.py`、`tools/gen_source_manifest.py`、`verify_freeze.py`

## 3. 冻结执行顺序

1. 服务器：`b2_injection_gate.py` → `results/B2_RESULT.json`（B2-1~B2-5，29 项）。
2. 服务器：`run_r0_5.py` 第二次（run_b）→ `results/r0_5_run_b.json`（run_id ≠ run_a，决策证据一致）。
3. 本地：`R0_5_EVIDENCE_MANIFEST.json`（results/ 内全部证据逐文件 sha256）→ `verify_r0_5.py --results` 全过。
4. 本地：`tools/gen_source_manifest.py` → `SOURCE_MANIFEST.json`（per-file sha256 + 层级分组）。
5. 写 `FROZEN_CONFIG.json`（含 control 语义 + SOURCE_MANIFEST 的 sha256）。
6. 写 `FREEZE_SEAL.json`（rev=1；manifest/config/证据/checkpoint sha256 + 决策引用）。
7. `verify_freeze.py --seal FREEZE_SEAL.json`：重算全链，逐项对账。

## 4. 与 A-v1 的关系（边界声明）

本冻结锁定的是 **旧 strict-online DCC-VRP 协议**下的 RRNCO-Ordering-RH-D 适配（SubProblem/
coordinator/公共 Bridge 管线）。A-v1 accept 协议的 RRNCO 接单适配另立工作区，不进入本 seal。
