# Learning-to-Delegate（L2D）A-v1 可比性复核（2026-10-13）

> 本目录只新增探针与文档；`CC_Compare/Learning-to-Delegate/**` 官方源码**未改动**。
> 原始日志：`/tmp/l2d_probe.log`（服务器 1003，`cc_compare` env，CPU）。

## 0. 结论摘要

| 问题 | 结论 | 证据 |
|---|---|---|
| 3 个回归权重能否真实加载 | **能**（strict，0 missing / 0 unexpected / 0 shape mismatch） | §2 |
| 能否真实推理 | **能**（55223 条官方子问题记录，MAE 0.071–0.138，与标签 Spearman 0.991–0.999） | §3 |
| LKH / HGS 子求解器是否可得 | **可得**：官方 `util.py` 在 import 时自动下载并编译，**已成功产出二进制** | §1 |
| 是否进入 A-v1 40 天正式臂 | **否——维持「实施排后」**，但**理由更正**：不是「缺权重/缺子求解器」，而是**机制与规模不匹配** | §4 |

## 1. 资产与子求解器（更正旧评估的「未下载」记录）

旧记录称「`generations/`（10GB）与 `lkh3/`、`hgs/` 子求解器二进制**未下载**」。
**复核结果：该记录不准确。**

- `CC_Compare/Learning-to-Delegate/lkh3/LKH-3.0.4/`（完整 LKH-3 源码 + Makefile）与
  `hgs/HGS-CVRP/Program/`（完整 HGS 源码 + Makefile）**本地即存在**；
- 官方 `Omni-VRP/L2D/code/util.py` 在**模块导入时**就硬性要求两个可执行文件：

  ```python
  # util.py:74-76（官方原文）
  executable = get_lkh_executable()      # 缺失则 wget + tar + make
  hgs_executable = get_hgs_executable()  # 缺失则 git clone + cmake + make
  ```
  即**连只用回归头做推理都绕不开 LKH/HGS**（这是本次实测到的真实依赖，非推测）；
- 服务器上按官方路径自动获取并编译成功（探针 cwd = `a1_accept_v1/`，未污染仓库根）：

  | 二进制 | 大小 | 版本来源 |
  |---|---|---|
  | `a1_accept_v1/lkh3/LKH-3.0.8/LKH` | 1,602,184 B | 官方 `util.py` 从 akira.ruc.dk 下载 LKH-3.0.8 源码后 `make` |
  | `a1_accept_v1/hgs/HGS-CVRP/build/hgs` | 143,408 B | 官方 `util.py` `git clone` HGS-CVRP 后 cmake+make |

  → **子求解器可得且可编译**；「缺件」不再成立。

## 2. 权重加载证据

`probe_l2d_weights.py` 用官方 `supervised.SubproblemNetwork`（`fit_subproblem: true` 对应类）
逐键加载 `exps/*/models/40000.pth`：

| 权重 | ckpt 大小 | ckpt 键 | 参数张量 | missing | unexpected | shape_mismatch | load_state_dict |
|---|---|---|---|---|---|---|---|
| `regression_model_512` | 14,373,663 B | `net/opt/step` | 77 | 0 | 0 | 0 | `All keys matched successfully` |
| `regression_model_2048` | 14,374,687 B | `net/opt/step` | 77 | 0 | 0 | 0 | `All keys matched successfully` |
| `regression_maml_512` | 14,373,663 B | `net/opt/step` | 77 | 0 | 0 | 0 | `All keys matched successfully` |

（`regression_maml_512` 用官方 `supervised_maml.SubproblemNetwork` 加载。）
ckpt `step=40000`；官方 `SubproblemNetwork` = `Linear(d_node→d_hidden=128)` +
`TransformerEncoder(n_layers=6, nhead=8)` + `Linear(128→1)` + 常数基线 `mean_lkh_dist`（buffer）。

## 3. 推理证据（官方子问题数据）

数据：官方 `generations/mixed_merge/subproblem_selection_hgs/val_subproblems.npz`
（**N = 55,223** 条子问题记录，`d_node = 3`），标签 = HGS 求解后的子问题里程
（`lkh_dists`），基线 = 委托前的 `prev_dists`。

| 权重 | MAE(model) | MAE(prev_dists 基线) | MAE(常数均值) | Spearman(model) | Spearman(prev) | 确定性 |
|---|---|---|---|---|---|---|
| `regression_model_512` | **0.1128** | 0.1831 | 2.9179 | 0.9949 | 0.9944 | max\|Δ\|=0 |
| `regression_model_2048` | **0.0715** | 0.1831 | 2.9179 | 0.9986 | 0.9944 | max\|Δ\|=0 |
| `regression_maml_512` | **0.1379** | 0.1831 | 2.9179 | 0.9906 | 0.9944 | max\|Δ\|=0 |

样例（512 档）：`preds[:5]=[13.60, 13.38, 12.28, 15.05, 14.63]` vs
`labels[:5]=[13.58, 13.36, 12.23, 15.09, 14.66]`。

→ **权重真实、可加载、可推理，且确有信号**（三个都优于 `prev_dists` 基线）。
注意 `prev_dists` 本身的 Spearman 已达 0.994（因为「委托前的距离」与「委托后的距离」
高度相关），所以**回归头的边际价值主要体现为 MAE 的降低**，而不是排序相关性。

## 4. 可比性判定（维持「实施排后」，理由更正）

判定依据从「缺资产」更正为**机制 + 规模的双重不匹配**，证据如下：

1. **回归头的语义是「估 LKH/HGS 子问题里程」**：训练配置
   `dataset_dir: generations/mixed_merge/subproblem_selection_hgs`、`solver: HGS`、
   `fit_subproblem: true`，模型输出以 `mean_lkh_dist` 为常数基线（`preds = Σhead/n + mean_lkh_dist`），
   目标 `lkh_dists` = 子问题交给 HGS 后的里程。它**只在「大实例分解 → 逐子问题委托」
   的闭环里有意义**，不是通用的「下一单接/拒」打分器。
2. **规模方向相反**：官方 `generations/` 只有 `clustered_N1000/N2000`、
   `mixed_nc{3,5,7}_N{500,1000,2000}`（n ≥ 500）。A-v1 的决策池 **≤ ~90**（全天 200 单），
   且每决策 10 s 预算、揭示即承诺。官方机制的核心收益来自「在几千节点上选一个够好的
   子问题交给子求解器」，在 90 节点的池上无对应设定。
3. **无 reveal→commit 决策层**：L2D 是静态实例上的迭代构造轨迹（每步选子问题 → 子求解
   → 拼回），没有「揭示事件 → 接受/拒绝不可撤销」的语义；把它接进 A-v1 必须**自行发明**
   决策层（即「硬适配 = 发明新决策策略」，与旧判断一致）。
4. 旧评估里「L2D 需要 LKH/HGS 且不可得」这半句已由 §1 否证；但即使子求解器可得，
   上述 1–3 条仍成立。

**最终判定**：`实施排后 / 不作为 A-v1 40 天正式臂`。理由 = 机制（子问题里程回归 + 无
reveal 决策层）与规模（n≥500 vs ≤90）双重不匹配，**不是资产缺失**。论文引用时按
「大尺度委托学习」文献锚点处理，不冒充同口径对比。

若将来确需该锚点，需**预声明**：(a) 允许在 ≤90 池上使用官方回归头（明确标注为
「小尺度零样本迁移」，非官方设定）；(b) 需定义 reveal→commit 决策层并单独冻结其协议；
(c) 报告 `prev_dists` 基线以分离「回归头贡献」与「距离启发式贡献」（§3 已给出该基线口径）。
