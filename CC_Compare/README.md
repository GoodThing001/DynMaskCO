# CC_Compare — 对比方法复现工作区

> 用途：复现「同赛道学习型方法」，并针对 DynMaskCO 做 DCC-VRP 协议适配，用于公平对比。
> 约定：一个方法一个文件夹。本目录位于 MaskCO-main 根下（不改原始 MaskCO 源码，也不改动 `C-VRP_Cold-chainVehicleRoutingProblem/`）。

## 方法清单（对应 `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/实验与评估/基线对比.md` §2.3）

> 下载时间 2026-09-08。commit 冻结清单见 [`REPO_PINS.md`](REPO_PINS.md)。**主表候选**加粗。

### 主表候选（按执行优先级）

| 文件夹 | 方法 | 出处 | 开源仓库 | 代码 | 权重 |
|--------|------|------|---------|:---:|:---:|
| **`PyVRP/`** | PyVRP-RH-D（**已冻结，protocol rev1 + identity rev4**） | INFORMS JOC 2024 | [`PyVRP/PyVRP`](https://github.com/PyVRP/PyVRP) | ✅ | ✅ pyvrp==0.14.0 wheel 冻结 |
| **`CO-enriched-ML/`** | CO-Enriched ML（原生动态 VRPTW） | Transportation Science 2024 | [`tumBAIS/euro-meets-neurips-2022`](https://github.com/tumBAIS/euro-meets-neurips-2022) | ✅ | 仓库自带 1 个（NN iteration-29），其余需重训 |
| **`RRNCO/`** | Real-World NCO | ICLR 2026 | [`ai4co/real-routing-nco`](https://github.com/ai4co/real-routing-nco) | ✅ | ✅ rcvrp/rcvrptw/atsp（服务器拉回） |
| **`CaDA/`** | Constraint-Aware Dual-Attention | ICML 2025 | [`CIAM-Group/CaDA`](https://github.com/CIAM-Group/CaDA) | ✅ | ✅ HF `Goodyee/CaDA` checkpoint.zip（hf-mirror） |
| **`RouteFinder/`** | Foundation Model for VRP | TMLR 2025 / ICML 2026 J2C | [`ai4co/routefinder`](https://github.com/ai4co/routefinder) | ✅ | ✅ rf-transformer/rf-pomo/rf-moe ×50 + rf-transformer×100 |

### 次选 / 历史参考 / 备用

| 文件夹 | 方法 | 出处 | 开源仓库 | 代码 | 权重 |
|--------|------|------|---------|:---:|:---:|
| `MVMoE/` | Multi-Task MoE VRP Solver | ICML 2024 | [`RoyalSkye/Routing-MVMoE`](https://github.com/RoyalSkye/Routing-MVMoE) | ✅ | ✅ 仓库自带 `pretrained/`（n50/n100） |
| `Learning-to-Delegate/` | Subproblem Selection for VRP | NeurIPS 2021 | [`mit-wu-lab/learning-to-delegate`](https://github.com/mit-wu-lab/learning-to-delegate) | ✅ | ⚠️ 另需 10GB Dropbox zip（非优先，未下载） |
| `POMO/` | Policy Optimization with Multiple Optima | NeurIPS 2020 | [`yd-kwon/POMO`](https://github.com/yd-kwon/POMO) | ✅ | ✅ 仓库自带（CVRP100 等） |
| `DeepACO/` | Neural-Enhanced Ant Systems | NeurIPS 2023 | [`henry-yeh/DeepACO`](https://github.com/henry-yeh/DeepACO) | ✅ | 仓库自带数据（网络权重待查） |
| `AttentionModel/` | Attention, Learn to Solve Routing Problems | ICLR 2019 | [`wouterkool/attention-learn-to-route`](https://github.com/wouterkool/attention-learn-to-route) | ✅ | ✅ 仓库自带 `pretrained/`（cvrp_50/100） |
| `Omni-VRP/` | Omni-Generalizable VRP | ICML 2023 | [`RoyalSkye/Omni-VRP`](https://github.com/RoyalSkye/Omni-VRP) | ✅ | ✅ 仓库自带 `pretrained/` |
| `Sym-NCO/` | Symmetric NCO | NeurIPS 2022 | [`alstn12088/Sym-NCO`](https://github.com/alstn12088/Sym-NCO) | ✅ | 仓库自带（Sym-NCO-POMO/AM 内） |
| `SGBS/` | Simulation-Guided Beam Search | NeurIPS 2022 | [`yd-kwon/SGBS`](https://github.com/yd-kwon/SGBS) | ✅ | 待查 |
| `Learn-Improvement-Heuristics/` | Learning Improvement Heuristics | IEEE TNNLS 2022 | [`WXY1427/Learn-Improvement-Heuristics-for-Routing`](https://github.com/WXY1427/Learn-Improvement-Heuristics-for-Routing) | ✅ | 待查 |
| `PIP-constraint/` | Learning to Handle Complex Constraints | NeurIPS 2024 | [`jieyibi/PIP-constraint`](https://github.com/jieyibi/PIP-constraint) | ✅ | ⛔ 不可适配（单车辆，缺容量） |
| `MAPT/` | MA-Pointer-Transformer（多车动态取送货） | AAAI 2026 | ❌ 无公开源码 | — | 仅论文数据对比，见 [`MAPT/README.md`](MAPT/README.md) |

### 许可证缺口（待补，准入前需确认）

- `POMO/`、`Learning-to-Delegate/`、`Sym-NCO/`、`Learn-Improvement-Heuristics/` 仓库内无 LICENSE 文件，准入前需在 README/官网确认许可证。

### 体积提醒

- `Sym-NCO/` 1.4G、`Omni-VRP/` 937M、`CO-enriched-ML/` 742M（多为论文实验产物/数据）。
- 根 `.gitignore` 与 `sftp.json` 已加精确忽略规则（`checkpoints/`、`pretrained/`、`CaDA/data/`、`Omni-VRP/data/`、`evaluation/results/`、`0_test_data_set/` 等 + `*.ckpt/*.npz/*.npy/*.zip/*.tar*` 扩展名），源码正常跟踪。
- 候选未忽略大目录（如需再评估）：`CO-enriched-ML/experiments/`（189M 论文采样产物）、`CO-enriched-ML/instances/`（134M）。

## 下载与冻结状态

- **代码**：12 个仓库已 clone 到本地（2026-09-08），commit SHA / 许可证见 [`REPO_PINS.md`](REPO_PINS.md)。
- **身份**：源码树 + 权重/数据的 SHA-256 见 [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json)（生成器 [`tools/gen_asset_manifest.py`](tools/gen_asset_manifest.py)，可重跑复验）。
- **同步**：本地 → 服务器经 VSCode SFTP（`.vscode/sftp.json`；大文件/权重已加入 ignore，需上传时单独处理）。
- **HF 下载**：huggingface.co 直连不可用，走 `hf-mirror.com` 镜像。

> `MAPT` 无开源代码（仅论文数据对比，见 [`MAPT/README.md`](MAPT/README.md)，注意指标对齐）。

## DCC-VRP 适配（legacy one-shot 与正式 adapter 状态）

以下为**旧 one-shot 适配脚本**（数据桥 + 环境门控 + 评估 harness + README）——均为 legacy：

| 方法 | 适配脚本 | 正式状态 |
|------|---------|---------|
| CaDA | [`CaDA/dcc_vrp/`](CaDA/dcc_vrp/README.md) | ❌ legacy one-shot；future-as-visited 语义 + 无 reveal 循环，不符合 v4 strict-online |
| RRNCO | [`RRNCO/dcc_vrp/`](RRNCO/dcc_vrp/README.md) | ❌ legacy one-shot；一次性解码 + `_true_*` mask 泄漏 + 无 frozen prefix/冷链 trace |
| PIP-constraint | [`PIP-constraint/dcc_vrp/`](PIP-constraint/dcc_vrp/README.md) | ⛔ 无法适配（单车辆 TSPTW/TSPDL，缺容量+多车辆） |

**适配状态（2026-09-09；identity 2026-09-10 更新）**：
- **PyVRP-RH-D：已完成并正式冻结**（`PyVRP/dcc_vrp/`，protocol revision 1 行为 + identity revision 4 计算身份；DEV-PROTO 9×32 全部 Gate 通过；证据产物 `PyVRP/dcc_vrp/results/devproto_9x32_iter300/`）。identity rev2→rev3（消除循环哈希）→rev4（common `baseline_contract.artifact_hash` 排除顶层 `runtime_s`），行为不变由小样本 decision_hash parity 证明。等 DEV-GATE 完成后做服务器环境 + 跨平台 parity。
- **OR-Tools-RH-D：已正式冻结**（`OR-Tools/dcc_vrp/`，protocol revision 1 + identity manifest revision 1；OR7 预算扫描选定 `solution_limit=30`，九 cell 等权 pure distance 17.3956；冻结三件套 `SOURCE_MANIFEST.json` ← `FROZEN_CONFIG.json` ← `FREEZE_SEAL.json`）。
- 其余方法（RRNCO / CaDA / RouteFinder / CO-enriched-ML 等）：待按顺序适配——RRNCO-RH → CaDA/RouteFinder 二选一。旧 `run_dcc.py` 保留为 legacy 诊断入口，不得用于 v4 正式比较。

统一适配原则（权威文档：`C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/实验与评估/基线对比.md`）：

1. **输入**：5D 客户特征（x, y, demand, tw_start, tw_end）+ 动态揭示（`reveal_time`）。
2. **Non-anticipatory**：不读取未来客户数量、身份、可行性 mask 或目标值；未来节点特征清零。
3. **指标统一**：D/Q/E/J 由共同 C0 evaluator 重算，不信任外部方法自报值（口径见 `docs/当前规划/实验与评估/评估口径.md`）。
4. **同尺度**：方法坐标均在 `[0,1]²`，cost 与 DynMaskCO 直接可比。

### 环境 + 运行

```bash
# 1) 建环境（RTX 5090 需 torch>=2.7，见脚本内注释）
bash CC_Compare/setup_envs.sh cc               # ★推荐：CaDA+RRNCO 共用一个环境（PIP 跳过）
#   或从已有 CUDA torch 环境克隆（5090 常见做法，clone-then-prune）：
bash CC_Compare/setup_envs.sh clone <base_env>  # base = 已在 5090 跑通 CUDA torch>=2.7 的母环境
#   或分建：cada / rrnco / pip / all

# 2) 各方法 run 命令见对应 dcc_vrp/README.md（含 checkpoint 下载与评估命令）
```

> ⚠️ 已知风险：CaDA 原始 requirements pin 了 torch 2.0.1 + torchrl 0.1.1（旧 API），
> 需升级 torch>=2.7 并用 `dcc_vrp/_torchrl_compat.py` 打兼容补丁；RRNCO 用
> `normalize=False` 保证成本可比（距离矩阵与训练分布有轻微 OOD，zero-shot 局限）。
