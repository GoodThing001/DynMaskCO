# CC_Compare — 对比方法复现工作区

> 用途：复现「同赛道学习型方法」，并针对 DynMaskCO 做 DCC-VRP 协议适配，用于公平对比。
> 约定：一个方法一个文件夹。本目录位于 MaskCO-main 根下（不改原始 MaskCO 源码，也不改动 `C-VRP_Cold-chainVehicleRoutingProblem/`）。

## 方法清单（对应 `C-VRP_Cold-chainVehicleRoutingProblem/docs/ALL/基线对比.md` §2.3）

| 文件夹 | 方法 | 出处 | 开源仓库 | 可下载 |
|--------|------|------|---------|:---:|
| `CaDA/` | Constraint-Aware Dual-Attention | ICML 2025 | [`CIAM-Group/CaDA`](https://github.com/CIAM-Group/CaDA) | ✅ |
| `PIP-constraint/` | Learning to Handle Complex Constraints for VRP | NeurIPS 2024 | [`jieyibi/PIP-constraint`](https://github.com/jieyibi/PIP-constraint) | ✅ |
| `RRNCO/` | Real-World Neural Combinatorial Optimization | ICLR 2026 | [`ai4co/real-routing-nco`](https://github.com/ai4co/real-routing-nco) | ✅ |
| `MAPT/` | MA-Pointer-Transformer（多车动态取送货） | AAAI 2026 | ❌ 无公开源码（论文列 `Beihang-BIGSCITY/MAPT` 但不可访问） | 仅论文数据对比，见 [`MAPT/README.md`](MAPT/README.md) |

## 服务器下载命令

```bash
cd /home/hzeng/project/MASKCO-Main
mkdir -p CC_Compare && cd CC_Compare

git clone https://github.com/CIAM-Group/CaDA.git
git clone https://github.com/ai4co/real-routing-nco.git RRNCO
git clone https://github.com/jieyibi/PIP-constraint.git
```

> 下载后 SFTP 同步到本地对应文件夹。`MAPT` 无开源代码，跳过 clone（如需论文数据对比，用论文报告的结果，注意指标对齐）。

## DCC-VRP 适配（脚本已就绪）

每个方法在 `dcc_vrp/` 下放了一套**不改原始代码**的适配脚本（数据桥 + 环境门控 + 评估 harness + README）：

| 方法 | 适配脚本 | 状态 |
|------|---------|------|
| CaDA | [`CaDA/dcc_vrp/`](CaDA/dcc_vrp/README.md) | ✅ 可适配（VRPTW 变体，`run_dcc.py`） |
| RRNCO | [`RRNCO/dcc_vrp/`](RRNCO/dcc_vrp/README.md) | ✅ 可适配（rcvrptw，`run_dcc.py`，`normalize=False`） |
| PIP-constraint | [`PIP-constraint/dcc_vrp/`](PIP-constraint/dcc_vrp/README.md) | ⛔ 无法适配（单车辆 TSPTW/TSPDL，缺容量+多车辆） |

统一适配原则（详见 `C-VRP_Cold-chainVehicleRoutingProblem/docs/ALL/基线对比.md` §1）：

1. **输入**：5D 客户特征（x, y, demand, tw_start, tw_end）+ 动态揭示（`reveal_time`）。
2. **Non-anticipatory**：`--mask_future` 下未来节点特征清零 + 只从可见客户出发（对应 P0-2）。
3. **指标统一**：cost = 纯行驶距离（无惩罚）；TW Feas / Cap Feas 单独报告（`evaluation_contract.md` §1.0）。
4. **同尺度**：三方法坐标均在 `[0,1]²`，cost 与 DynMaskCO 直接可比。

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
