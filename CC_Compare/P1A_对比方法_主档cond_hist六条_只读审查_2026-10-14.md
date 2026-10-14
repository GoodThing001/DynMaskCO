# 对比方法线：**主档 `cond_hist` 六条记录 只读审查**（2026-10-14）

```
性质：**只读**。仅限**本地已有资料**；**未访问服务器**；**未修改任何 gate**；
      **未重新计算效用**；**未重新判定旧结果**。
授权：用户 2026-10-14「L1 就绪审计后的五项裁定」④ —— 批准只读盘查，
      建议先审查六条主档 cond_hist 记录。
分层：VERIFIED / NOT_VERIFIABLE / MISMATCH（tier 语义见文末）
机器可读：CC_Compare/P1A_对比方法_主档cond_hist六条_只读审查_2026-10-14.json
复算工具：CC_Compare/tools/role_c_main_tier_cond_hist_review.py（只读、可重跑）
```

---

## 0. 样本界定

来源：`CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_2026-10-14.json`。
用户给出的 20 条"最干净子群"组成经**独立复算逐条确认**：

| 组成 | 复算 |
|---|---|
| `cond_hist` 记录 | **13**（`p_c=0` 7 + 主档 6） |
| `uncond_hist` 记录 | **7**（全为 `p_c=0`） |
| 主档 `p_c=(5,10,15)` | **6** |
| 敏感性档 `p_c=0` | **14** |

⚠️ 这 20 条是**资格较易盘查的方法臂记录**，**不是** 20 种算法，
也**不是** 20 条已通过正式资格的独立证据。

**本件对象 = 主档 `cond_hist` 的 6 条**，其 gate 路径与 provider：

| # | provider | gate |
|---|---|---|
| 1 | `cada` | `results/a1_ext_formal2/cada_51015/gate.json` |
| 2 | `coenrich`（`coenrich_nopchgs`） | `results/a1_ext_formal2/coenrich_nopchgs_51015/gate.json` |
| 3 | `coenrich`（`coenrich_pchgs`） | `results/a1_ext_formal2/coenrich_pchgs_51015/gate.json` |
| 4 | `mvmoe` | `results/a1_ext_formal2/mvmoe_51015/gate.json` |
| 5 | `routefinder` | `results/a1_ext_formal2/routefinder_51015/gate.json` |
| 6 | （gate 未记 provider） | `results/a1_ext_formal2/rrnco_51015/gate.json` |

---

## 1. 分层结果

**合计 72 项检查：VERIFIED 43 / NOT_VERIFIABLE 24 / MISMATCH 5**
（5 个 MISMATCH 同属一**类**，见 §3；第 6 条因 gate 未记 provider 而不触发该类。）

| 检查项 | 分层 | 六条的实测 |
|---|---|---|
| `config_time_limit` | **VERIFIED** | 六条均 `10.0` |
| `max_decide_within_time_limit` | **VERIFIED** | `max_decide_s ≈ 1.719 s` ≤ `10 s` ⇒ 求解层面**未见越限** |
| `per_decision_timing_present` | **VERIFIED** | `solver_meta.max_decide_s` / `mean_solve_time_s` 存在（`≈0.2368 s`） |
| `fallback_semantics_recorded` | **VERIFIED** | `solver_meta` 有 `n_fail=0`、`reject_reasons={'certify_budget': 5278, 'deferred': 3}`；**但 gate 里没有** `fallback_applied`/`fallback_kind` 这类回退语义字段 |
| `identity_problems_from_matrix` | **VERIFIED** | 六条 `identity_problems=[]`、`identity_verified=true` |
| `per_day_arm_identified` | **VERIFIED** | 5 条 `per_day` 归属 `ordering_accept`、1 条归属 `rrnco_accept`（按 utility 均值精确匹配，见 §2） |
| `booking_horizon_in_gate` | **NOT_VERIFIABLE** | 六条 `config` **均无** `booking_horizon` ⇒ `NOT_VERIFIABLE_FROM_ARCHIVED_GATE`（**不推定**泄漏、**不追溯**改旧裁决） |
| `identity_chain_hash_level` | **NOT_VERIFIABLE** | gate 里只有**路径**（`provider_ckpt`、`baseline_from`），**无哈希** ⇒ 身份链**不可仅从该 gate 核验**；矩阵侧另有 identity 块，两者**不是同一证据源** |
| `strict_online_config_completeness` | **NOT_VERIFIABLE** | `config` 缺席：`K`、`booking_horizon`、`inner_gate`、`future_policy`、`energy_pricing`、`gate_seed`、`arm_seed`、`rho` |
| `per_decision_timeout_events_present` | **NOT_VERIFIABLE** | gate 只有**逐日** `timeouts`，**没有**逐决策事件 ⇒ **"真实硬超时"不可证明** |
| `timeout_magnitude_selfconsistent` | **NOT_VERIFIABLE** | `per_day` 只覆盖**候选臂**（见 §2），**不覆盖** `cond_hist` 臂 ⇒ `cond_hist` 的 `mean_timeouts=0.75` / `timeout_days=15` **无法逐日对账**（只有汇总） |
| `arm_label_identifies_provider` | **MISMATCH** | 矩阵 `arm`/`method` 记 `cond_hist`，而 `gate.config.provider` 是 `cada`/`coenrich`/`mvmoe`/`routefinder`（4 条触发；第 6 条 gate 未记 provider） |

---

## 2. 一个必须先说的归档结构隐患（**先修工具，再报结论**）

第一版审查工具把 `per_day.timeouts` 直接与 `arms['cond_hist']` 对账，
得出 `cada_51015` **"逐日合计 0 vs 声明 0.75"** 的 **MISMATCH**。
**该 MISMATCH 是我的工具错误，不是 gate 的矛盾。**

**真相**：`results[seed][p_c]` 里 `per_day` 是**单个无标签列表**，
与**多臂** `arms`（`cond_hist` / `explicit_feat` / `ordering_accept` / `uncond_hist`）并列，
**文件里没有任何字段标明它属于哪个臂**。按 utility 均值精确匹配可确定：

| 臂 | utility 均值 | 与 `per_day` 均值一致？ |
|---|---|---|
| `ordering_accept` | **-190.2519** | ✅ **（5 条 gate 的 `per_day` 属它）** |
| `cond_hist` | 633.9846 | ❌ |
| `explicit_feat` | 673.5656 | ❌ |
| `uncond_hist` | 643.9376 | ❌ |

⇒ `per_day` 是**候选臂**的逐日记录，**不是** `cond_hist` 的。
⇒ 工具已改为**先识别归属、再对账**，无法唯一识别时判 `NOT_VERIFIABLE`：
唯一性匹配失败 ⇒ **不得**臆断归属（该错误若不修，会产生**假 MISMATCH**）。

**这条本身就是接口契约（C1–C4）必须解决的问题之一**：
归档必须显式携带"这条逐日/逐事件记录属于哪个臂"。

---

## 3. 唯一的 MISMATCH 类：**arm 列不标识被比较对象**

* 矩阵的 `arm` / `method` 列对 6 条主档记录**都记 `cond_hist`**；
* 而 `gate.config.provider` 分别是 `cada` / `coenrich` / `mvmoe` / `routefinder` / `rrnco`；
* ⇒ **`arm` 列是"场景抽样臂名"，不是被比较的 provider**；
  provider 只能从 `gate_path` / `config` 恢复。

**影响**：任何按 `method` 列聚合、去重或配对的脚本，都会把
**6 个不同 provider 的记录当成同一方法的重复运行**（用户点名的
"相同方法在不同运行中的重复身份"问题 —— 这里是它的**反面**：
**不同方法被标成同一方法**）。
**本件只报事实，不修改矩阵、不重新判定旧结果。**

---

## 4. 用户要求的"报告超时量级"口径已落实

审计报告里的 `真实超时量级 ≥5/天` **已改名**为 **`报告超时量级 ≥5/天`**
（键名 `reported_timeout_magnitude_ge_5_per_day`）。
理由（用户给定，照录）：**目前没有逐决策事件记录，不能仅凭 `mean_timeouts`
证明每次都是独立可核验的真实硬超时。**
本件的 `per_decision_timeout_events_present = NOT_VERIFIABLE` 即同一口径的落地。

---

## 5. 本件**未**做 / **不**声称

* **未**访问服务器；**未**修改任何 gate；**未**重新计算效用；**未**重新判定旧结果；
* **不**声称这 6 条"通过"或"未通过"正式资格 —— 本件只做**证据可用性分层**；
* `NOT_VERIFIABLE` **不等于**"该项为空/未发生"：它表示**现有归档不足以核验**；
* **不**把这 6 条当成 6 种算法，也**不**把它们当成独立证据。

**tier 语义**：`VERIFIED` = 可用现有归档机械核验；
`NOT_VERIFIABLE` = 现有归档不足以核验（**不等于**不存在/未发生）；
`MISMATCH` = 现有记录之间**互相矛盾**。
