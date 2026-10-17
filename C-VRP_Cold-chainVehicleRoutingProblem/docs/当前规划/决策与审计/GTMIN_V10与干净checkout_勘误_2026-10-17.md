# GTMIN · V-10 与"干净 checkout"结论的勘误（2026-10-17，追加式）

> **性质：勘误（errata），追加式。** 本文件**先于**相关文件的发布而登记。
> 不修改任何既往报告、测试结果或既有结论；只登记"因发布某些文件而失效的表述"以及
> **对 V-10 结果的判定规则**。
> 不启动正式实验、不训练、无 GPU、不碰最终种子；不改生产认证器与安装入口。

---

## 1. 触发：六个此前**故意不发布**的文件被授权发布

| 文件 | 此前状态 | 现在 |
|---|---|---|
| `scripts/tests/test_q3_g_tmin_candidate.py` | 未跟踪（本地资产） | 发布 |
| `scripts/tests/test_q3_g_tmin_v2.py` | 未跟踪 | 发布 |
| `scripts/tests/test_q3_archive_regression.py` | 未跟踪 | 发布 |
| `scripts/tests/test_q3_door_interval_thresholds.py` | 未跟踪 | 发布 |
| `scripts/q3_qualification/q3_t_min_candidate.py` | 未跟踪 | **发布（见 §4 R-1 披露）** |
| `scripts/q3_qualification/q3_t_min_candidate_v2.py` | 未跟踪 | **发布（见 §4 R-1 披露）** |

前四个正是 **V-10 取代行台账**所需的"四个本地 Q3 套件"
（`test_q3_gtmin_install_verification.py` 的 `local_suites` 判定：
`test_q3_g_tmin_candidate.py`、`test_q3_g_tmin_v2.py`、`test_q3_archive_regression.py`、
`test_q3_door_interval_thresholds.py`，加上已入库的 `q3_theorem_b_audit.py`）。

---

## 2. 因此失效的两处已提交表述（就地登记，不改写原文）

| 已提交文件 | 原文 | 现状 |
|---|---|---|
| `GTMIN交付后独立勘误_对象保留与复现性_2026-10-17.md` §7.2 | "只剩 V-10 SKIP：它需要 4 个**未提交**的本地 Q3 套件" | **失效**：四个套件已入库 |
| 同上 §7.5 风险表 **R-3** | "V-10 的取代行台账依赖 4 个未提交套件 / 只能本机复核" | **失效**：台账现在在干净 checkout 中**可评估** |
| `G2_勘误裁决与G1隔离候选_2026-10-17.md` §6 缺口表 | "V-10 取代行台账 / 不可复现" | **失效**（同上） |

**没有任何数值或判定被改写**：上述三处只是"当时为真"的事实描述；本勘误把它们标记为
**被本次发布取代（superseded by publication）**，并给出新的判定规则（§3）。

---

## 3. V-10 结果的判定规则（**先声明，后测量**）

在干净 checkout 中，`have_local_suites` 现为真，因此
`test_q3_gtmin_install_verification.py` 的 **V-10** 会真正执行那 12 条被取代行的台账。

**事先声明的判定规则（不允许事后解释）**：

| 若 V-10 结果 | 则整体结论 | 必须做什么 |
|---|---|---|
| V-10 **PASS**（`observed == expected`，且 7 个 unaffected 套件全绿） | 干净 checkout 由 `INCOMPLETE` 升为 **`PASS`**（12/12） | 在本文件追加实测记录；**不得**改写旧报告的 INCOMPLETE 历史 |
| V-10 **FAIL**（出现**未声明**的失败行，或已声明行未失败） | 干净 checkout 为 **`FAIL`** | **必须**作为**新缺陷**登记并上报，**不得**把它重新解释为"套件版本差异"而放行 |
| 某个套件**跑不起来**（缺依赖/异常退出） | V-10 记为 FAIL（`<no artifact>`） | 同上：登记为缺陷，不得跳过 |

**这条规则的意义**：V-10 之所以此前一直 SKIP，是因为台账所依赖的证据不在仓库里。
现在证据进仓库了，V-10 就从"未评估"变成"**可被第三方复核的判定**"——
它**可能通过，也可能暴露真问题**。两种情况都必须如实记录。

---

## 4. R-1 披露：两个**已作废**的候选模块被发布

`q3_t_min_candidate.py` 与 `q3_t_min_candidate_v2.py` 都会对**调用方提供的数据**执行 `repr()`
——这正是 G-TMIN 轮次从生产认证器中移除的那类缺陷（登记为残余风险 **R-1**，
原登记只点名 v2，实测 **v1 同型**）。

| 项 | 内容 |
|---|---|
| 发布理由 | **证据完整性**：G-TMIN 报告的 V-07 等条款以"该字符串只存在于候选模块"为依据，候选不入库则该条款第三方不可核验；两者的字节身份其实已随报告公开 |
| 发布状态 | **已作废（SUPERSEDED）**，**不得**复用、**不得**作为任何资格依据、**不得**恢复为生产候选 |
| 依据 | 隔离替代物是 `scripts/q3_qualification/q3_g1_time_identity_candidate.py`（已发布，26/26） |
| 残余风险 R-1 的现状 | **由"不发布"改为"已发布且明确作废"**；风险性质不变（存在一份有 `repr()` 缺陷的候选文本），但不再有"不可核验"的成分 |

**一句话**：按证据完整性发布已作废的候选；它们的存在本身就是那轮缺陷的证据，
不是可用的实现。

---

## 5. 与状态的关系

不改变任何科研状态：`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、
`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0、
最终测试种子 `SEALED`、`PRODUCTION_NOT_QUALIFIED` —— 全部不变。

**不主张**：V-10 会通过；干净 checkout 会变成 PASS；已作废候选可用于任何资格。
