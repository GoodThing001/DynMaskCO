# REPOSITORY_MAP — 一页看懂 DynMaskCO-CC 仓库

> 建立于 2026-10-18 · 基线 HEAD `fc51fca9d011e4df1eb90c8b321a1553c0dc217b`（`codex/o0cc-cal-phys` = `main`）
> 本文件只做**导航**。它不产生、不修改、不重新裁决任何科研结论。
> 详细审计见 [`C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/`](C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/README.md)。

## 1. 仓库是什么

**MaskCO**（ICLR 2026）的上游实现 + 一个正在做的扩展研究：把 MaskCO 的"掩码生成—重构"范式，
用到**动态冷链取货（pickup-to-depot）**的**在线因果决策**上，方法名 **DynMaskCO-CC**。
外加一套**第三方 learned-VRP 方法复现区**与一棵**研发证据树**。

```
MASKCO_code/                      上游 ICLR-2026 MaskCO 实现（TSP/CVRP/MIS）—— ❌ 永不修改
C-VRP_Cold-chainVehicleRoutingProblem/   ★ DynMaskCO-CC 主研发区（唯一可编辑的研究工作区）
CC_Compare/                       第三方 learned-VRP 方法复现（16+ 仓库）· ⚠️ 只允许改各方法的 dcc_vrp/ 适配层
research/                         研发证据树（登记 / 独立审计 / 回执）—— 证据，只追加
tools/                            交付与运维 CLI（605 文件，其中 5 个可复用）
docs/                             仅 3 个文件且被 .gitignore 忽略 —— 真正的文档在扩展区，见下
results/                          ⚠️ 未跟踪且未忽略的待交付证据（Q3 第 2 轮，13 文件）
```

## 2. 从哪里进入（按你要做的事）

| 我要… | 打开 |
|---|---|
| **知道项目现在什么状态** | [`C-VRP_…/项目当前状态.md`](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md) —— **唯一权威，读前 30 行** |
| **知道所有尝试与阶段** | [`C-VRP_…/docs/当前规划/结果与进度/执行进度表.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/结果与进度/执行进度表.md) —— **以顶部最新条目为准** |
| **知道每条决策的预声明与裁决** | [`C-VRP_…/docs/当前规划/决策与审计/`](C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/决策与审计/)（130 文件，最新一批是 2026-10-16/17 的 Q3 / G-2 / G-3 / GTMIN / C2 / M1 / L2 / L3 / C0） |
| **从代码入手** | [`C-VRP_…/scripts/`](C-VRP_Cold-chainVehicleRoutingProblem/scripts/) → 阅读顺序 `scripts/README.md` → `project_paths.py` → `coldchain/coldchain_contract.py` → `coldchain/coldchain_state.py` → `simulation/strict_online_env.py` → `simulation/jf1h_repair.py` |
| **看三条研究线的正文** | [`01_取货线/文档/`](C-VRP_Cold-chainVehicleRoutingProblem/01_取货线/) · [`02_配送线/文档/`](C-VRP_Cold-chainVehicleRoutingProblem/02_配送线/) · [`03_负结果线/文档/`](C-VRP_Cold-chainVehicleRoutingProblem/03_负结果线/) |
| **看负结果论文** | [`C-VRP_…/03_负结果线/文档/论文与投稿/论文工作稿_收口修订_2026-10-09.md`](C-VRP_Cold-chainVehicleRoutingProblem/03_负结果线/文档/论文与投稿/)（唯一当前可编辑正文）；[`negative_result_paper/`](C-VRP_Cold-chainVehicleRoutingProblem/negative_result_paper/) 是**已封存的复现材料集合** |
| **看证据树** | [`research/review_3080_20261010/`](research/review_3080_20261010/) —— `d3_g0_evidence_root_v1/` 是资格主干，`pickup/`+`delivery/` 是实验证据 |
| **交代码** | `python tools/git_deliver.py --message-file <msg> --paths <files>`（**必须给 `--paths`**） |
| **做仓库治理** | [`C-VRP_…/docs/仓库治理/README.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/README.md) |
| **看 P1 两个新工作包（草案，未授权）** | [`C-VRP_…/docs/当前规划/P1工作包/README.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/P1工作包/README.md) —— `P1-NUM-01`（数值与冷链认证可行性）+ `P1-OPP-01`（MaskCO 可预测改善空间），均标记 `DRAFT_NOT_AUTHORISED` |

## 3. 现在进展到哪（只给指针，结论以权威文件为准）

- **主线未变**：MaskCO 掩码重构 + 严格在线因果 + 冷链效用三层。
- **贯穿全部轮次的一句话**（41 份战役/轮次文档里 26 份逐字相同）：**稳定、可归因的 MaskCO 核心在线正向增益仍未建立**（`stable_online_gain=False`）。
- **最新一轮 = Q3（连续时间温度认证）**：Q3-1/3/4 **闭合（实值层，条件式，非生产）**；**Q3-2 = S3 = NO-GO**，两轮止损用尽，**Q3 资格扩张终止**。
- **永久 FAIL，不追认**：`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`、**V-10 干净 checkout FAIL**、L1 `BLOCKED`。
- **明确未授权**：训练 / GPU / 正式 rollout、真实安装、生产替换 `plan_hash`、M2、L2/L3 签字、最终测试种子 `20260927`/`20260928`（`SEALED`）。

完整台账（已完成 / 已终止正式失败 / 尚未授权）：
[`docs/仓库治理/04_实验台账.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/04_实验台账.md)。

## 4. 不能碰的地方（红线摘要）

| 红线 | 出处 |
|---|---|
| `MASKCO_code/`、`data/baseline/`、`CC_Compare/MAPT_code`（幽灵 gitlink，`160000`，无 `.gitmodules`） | 硬不变量 / `tools/git_deliver.py:37` |
| 任何 `FAIL` / `NOT_RUN` 记录：**不得删除、不得改写成 PASS** | `M1b_Q2_测试规格冻结_2026-10-16.md:36` |
| `docs/当前规划/历史已收束/` 的 42 个 5 行存根：**是兼容层，不是垃圾** | `docs/目录整理记录/2026-10-08/README.md` |
| `results/_m1a_synth`、`_m1b_q2`、`_m1b_q2fix`、`_q3_next_20261017`：下划线开头但是**正式验收证据** | 根 `.gitignore` 的否定规则 |
| `scripts/**/d3_*`、`q3_*_experimental.py`、`q3_qualification/`：**不重构、不"修红"、不重跑** | 各自 docstring（`NOT INSTALLED` / `EXPERIMENTAL_NOT_FOR_PRODUCTION`） |
| 用户写下的科研复核结论：**只能追加注释，不能改写** | `GPT项目指令_长期版_2026-10-08.md:59` |

完整清单：[`docs/仓库治理/05_冻结证据与禁止修改路径.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/05_冻结证据与禁止修改路径.md)。

## 5. 新人必须知道的四件事

1. **正式证据大多不在 git 里。** `data/` 0 个跟踪文件（2.17 GB 在盘上）、`results/` 4,295 个文件里只有 13 个跟踪、`ckpts/` **本地为空**（冻结 encoder 只在服务器）。`git clone` 之后**一个正式数字都复现不出来**——这是 `.gitignore` 的既定设计（GitHub 100 MB 上限），但必须知道。
2. **权威链埋在时间日志之下。** 状态文件是**逆序追加**的：`项目当前状态.md` 的标题在第 243 行，`执行进度表.md` 在第 49 行，`00_项目导航.md` 的真正导航在第 115 行。→ **只看每个文件的前 30 行 + 结构性小节。**
3. **存在两套 `D-0xx` 决策编号**（`docs/当前规划/决策与审计/决策记录.md` 到 D-058；`docs/实验记录/决策记录.md` 到 D-075，**同号异义**）。→ 引用 `D-0xx` 必须带文件名；09-28 之后改用**杠杆代号**（A1/A3/I3/S3-4/L1/L2/L3/Q3/M1-a/M1-b/G-1…G-3/GTMIN/C0/C2/V-10/B-1…B-9）。
4. **`negative_result_paper/` 不是独立可运行包**（22 个脚本里 21 个 import 文件夹外的模块），且已被 `03_负结果线/README.md:15` 宣告历史封存。

## 6. 一次性验证入口（无需 GPU）

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem

# 控制链 / 身份 6 件套（纯 NumPy）
python scripts/tests/test_formal_gate_contract.py
python scripts/tests/test_contract_manifest.py
python scripts/tests/test_dev_gate_control.py
python scripts/tests/test_control_chain.py
python scripts/tests/test_aggregate_cross_cell.py
python scripts/tests/test_val_control_chain.py

# 计算层回归
python scripts/tests/test_coldchain_contract.py     # C0 20/20
python scripts/tests/test_sequential_oracle.py      # 6/6

# A-v1 层
python scripts/tests/test_quality_v2.py
python scripts/tests/test_a1_evaluator.py
python scripts/tests/test_rolling_optimizer.py      # ⚠️ 已知既有失败，见下
```

**已知既有失败（2026-10-18 核实，与本轮整理无关）**：
`test_rolling_optimizer.py::test_in_transit_no_double_count` 断言 `accepted == {1, 2}`，实际 `{1}`。
前两个用例 PASS。**原样登记，未修、未改判据。**

细节：[`docs/仓库治理/08_本次修改与验证记录.md`](C-VRP_Cold-chainVehicleRoutingProblem/docs/仓库治理/08_本次修改与验证记录.md)。
