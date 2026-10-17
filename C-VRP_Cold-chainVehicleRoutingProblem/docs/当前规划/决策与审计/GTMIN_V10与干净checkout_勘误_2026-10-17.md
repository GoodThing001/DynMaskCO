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

---

## 6. 实测结果（按 §3 规则登记，不改判据）

**测试环境**：真 `git clone`（`D:\gtmin_clean_clone`），HEAD 为发布这四个套件后的提交，
`git status` 0 项。命令与之前一致：

```
python scripts/tests/test_q3_gtmin_install_verification.py --out <json>
```

**结果**：

```
=== 12 checks: 11 PASS, 0 SKIP, 1 FAIL -> GTMIN_INSTALL_VERIFICATION_FAIL ===
```

**按 §3 的预声明规则：这是 FAIL，必须作为缺陷登记，不得重新解释为"套件版本差异"。**

### 6.1 V-10 的 declared vs observed

| 套件 | declared | observed（干净 checkout） |
|---|---|---|
| `test_q3_g_tmin_candidate.py` | TMIN-C-01/02, TMIN-N-02/03 | **完全一致** ✅ |
| `test_q3_g_tmin_v2.py` | C1, C10, C11 | **完全一致** ✅ |
| `test_q3_archive_regression.py` | ARC-01/05/06/07 | **`<no artifact>`**（崩溃） |
| `test_q3_door_interval_thresholds.py` | ERR-12 | **`<no artifact>`**（崩溃） |
| `test_q3_event_segmentation.py` | `[]` | **`<no artifact>`**（崩溃） |
| `test_q3_float_replay_qualification.py` | `[]` | **`<no artifact>`**（崩溃） |
| `test_q3_time_budget_audit.py` | `[]` | **`<no artifact>`**（崩溃） |
| `test_q3_plan_layer_controls.py` | `[]` | **`["PL-12"]`（未声明的失败行）** |
| `test_q3_strict_float_bounds.py` | `[]` | `[]` ✅ |
| `test_q3_thermal_interval_controls.py` | `[]` | `[]` ✅ |
| `q3_theorem_b_audit.py` | `[]` | `[]` ✅ |

### 6.2 根因（逐条实测，不是推测）

五处崩溃与一处失败行，**根因是同一类**：这些套件依赖**不在仓库里**的证据。

| 套件 | 缺失输入（异常原文） |
|---|---|
| `test_q3_archive_regression.py` | `results/_q3_next_20261017/SHA256_MANIFEST.json`（`FileNotFoundError`） |
| `test_q3_door_interval_thresholds.py` | `results/_q3_next_20261017/SHA256_MANIFEST_V2.json` |
| `test_q3_event_segmentation.py` | `data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz` |
| `test_q3_float_replay_qualification.py` | 同上 |
| `test_q3_time_budget_audit.py` | 同上（TBA-03 之后崩溃） |
| `test_q3_plan_layer_controls.py` **PL-12** | **不崩溃**，但 `n_files = 0`：它要普查 `data/**` 的冻结实例文件，一个也找不到 ⇒ 断言不成立 |

`git ls-files 'C-VRP_.../data/m0_scale*'` **为空** ⇒ **登记数据集本身未入库**；
`results/**` 按交付纪律从不入库。

### 6.3 这对"R-3 缺口"的含义（更正我自己的上一轮判断）

我上一轮把 V-10 的缺口描述为"需要四个未提交的本地套件"，并暗示**发布这四个文件即可闭环**。
**该判断不完整。** 实测表明：

> **缺口的本质是"证据本地性"（evidence locality），不是"四个文件未提交"。**
> 即使四个套件全部入库，V-10 仍会 FAIL，因为它还需要
> ① 登记的 `data/m0_scale` 数据集、② 两份私有 `SHA256_MANIFEST`。

因此：

- 早先的 **`INCOMPLETE`（SKIP）** 与现在的 **`FAIL`** **都是各自输入下的正确结论**，
  前者说"未评估"，后者说"评估了但不通过且原因是缺证据"；
- **不得**把这次 FAIL 写成"套件过时"或"版本差异"；它是**证据不可复现**的直接证据；
- **R-3 的状态更新**：从"四个套件未入库"改为
  **"V-10 无法仅凭仓库复现：依赖未入库的数据集与私有清单"**（更严重的登记）。

### 6.4 若要让它 PASS，只有两条路（都需要用户决定）

| 路径 | 内容 | 代价 |
|---|---|---|
| **A** | 把 `data/m0_scale`（三份 npz）与两份 `SHA256_MANIFEST` 入库 | 与"结果入库"纪律冲突；数据集体积与公开面扩大 |
| **B** | 让每个套件**显式声明证据前提**，缺失时输出**非失败的"未评估"行** | 需要改五个套件（含两个已入库的）；且 `unaffected` 的语义要从"全绿"改为"有前提的全绿" |

**本轮不擅自选任何一条**：A 违反既有纪律，B 会改动已发布结论的语义。两者都需要单独授权。

### 6.5 不主张

不主张 V-10 将来会通过；不主张 `PL-12` 是"无害的"（它确实是一条未声明的失败行，
只是根因与崩溃同源）；不主张旧报告的 `INCOMPLETE` 应改成 `FAIL`（历史记录不改写）。
