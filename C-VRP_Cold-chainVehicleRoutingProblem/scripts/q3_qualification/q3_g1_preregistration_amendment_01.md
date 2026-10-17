# G-1 预声明 · 勘误 01：时间上界的**距离模型域**（2026-10-17，追加式）

> **性质：勘误（amendment），追加式，不修改原预声明。**
> 对象：`scripts/q3_qualification/q3_g1_preregistration_2026-10-17.md`。
> 原文件登记 `T_clock_sup = horizon + travel_bound`，并把 `travel_bound` 的来源首选写成
> `coordinate_box_diagonal = √2 / tw_speed`。**该表述过强**，本勘误按独立审查（S-8）收窄为
> "**仅在受认证的距离模型内**成立"，并补上距离模型身份。

---

## 1. 为什么要改（两条源码事实）

1. **`dist_mat` 可以是外部提供的矩阵**：
   `strict_online_env.py:186-190` —— 数据集带 `dist_mat` 就用它，否则才用坐标欧氏距离；
   `:193-194` → `_travel = dist_mat[i, j] / tw_speed`。
   **实测（本仓三份登记数据）**：`dist_mat` 键**存在**，`dist_mat` 与 `euclid(coords)` 的最大偏差
   `1.1e-7 ~ 1.2e-7`、对称、`max = 1.2407/1.2763/1.2658 < √2` ⇒ **登记域确实等价于欧氏模型**，
   但这是**逐实例见证**出来的性质，**不是**仓库级不变量。
2. **生成器有两条抬高边权的分支**（`generate_coldchain_data.py`）：
   - `asymmetric=True`（`:81-89`）：`euclid × ((x + 1/x)/2) × rush`，
     `x ∈ [0.85, 1.35]`、`rush ∈ [1.2, 2.0]` ⇒ 倍率上界 **1.045741 × 2.0**；
   - `perturb_prob > 0`（`:141-143`）：被选中的边 `× U(1.5, 5.0)` ⇒ 倍率上界 **5.0**。
   两者可复合。⇒ `√2` **不是**仓库全数据域的上界。

---

## 2. 修正后的 `travel_bound` 规则（取代原 §8.1 相关表述）

```
travel_bound(model_composition, tw_speed) = √2 · Π(multiplier_ceiling) / tw_speed   ，向外膨胀 1 ulp
```

| 距离模型 | 倍率上界 | 见证方式 |
|---|---|---|
| `euclidean` | **1.0** | **逐实例（离线）**：`dist_mat` 对称 且 `max|dist_mat − euclid(coords)| ≤ 1e-6` |
| `asymmetric_transit` | **1.045741 × 2.0 = 2.090741** | **域级声明**（`asymmetric=True` 及生成器区间）；单个矩阵无法见证 |
| `long_tail_perturbation` | **5.0** | **域级声明**（`perturb_prob > 0` 及 `[1.5, 5.0]`） |
| `external_undeclared` | **无界** | **REFUSED**（`DISTANCE_MODEL_UNBOUNDED`），不得近似 |

**附加条件（全部必需）**：

1. `tw_speed` 必须 finite 且 `> 0`，且其身份进摘要（速度变了等于把每个时钟缩放）；
2. `dist_mat` 的最大边权**不得**在在线阶段计算 —— 它包含通往未揭示订单的边
   （`require_offline(stage)` 拒绝 `stage='online'`）；
3. 浮点保守方向：倍率由生成器在浮点里相乘，故上界向外膨胀 **1 ulp**；
4. **`T_clock_sup` 的完整成立前提**（补全原 §3.1）：
   - `horizon` 已按 G-2 的**逐实例见证**判定为公开量；
   - 返仓完成推进的时钟 `depart + travel(current, 0) ≤ horizon + travel_bound`（G-2 勘误 E-01）；
   - `travel_bound` 由第 1–3 条给出；
   - 且 `n_max`（推进次数）取 `max_iter` 单界 `20003` 并配**执行监视器**（原预声明未含监视器，本勘误补上）。

---

## 3. 对原判据/负控的增补（不删不改原编号）

| 编号 | 增补 |
|---|---|
| **C5'** | 时间单位之外，**距离模型**也是身份：模型名、倍率上界、`tw_speed` 三者进摘要；改任一项 ⇒ 摘要变 |
| **C8'** | "缺可信时间上界 ⇒ `NOT_CERTIFIABLE`" 扩充为：缺**距离模型见证**、缺**监视器收据**、或 `tw_speed` 非法，任一成立即 `NOT_CERTIFIABLE`/`REFUSED` |
| **N-9'** | 新增：`external_undeclared` 距离矩阵 ⇒ `REFUSED: DISTANCE_MODEL_UNBOUNDED` |
| **N-12** | 新增：`asymmetric` 或 `perturb` 域下仍使用 `coordinate_box_diagonal` ⇒ 必须 `REFUSED`（因为上界被真实边权超过） |
| **N-13** | 新增：`tw_speed ∈ {0, -1, nan, inf, None, 'fast'}` ⇒ `REFUSED: TW_SPEED_INVALID` |

---

## 4. 与 G-2 的关系（重要）

- G-2 勘误轮的 `D-1…D-5`（`test_q3_g2c_travel_domain.py`，**12/12 PASS**）已经用可执行反例确认：
  `√2` 在 `asymmetric` 与 `perturb` 域下**被真实边权超过**，且 `external_undeclared` **无界**。
- 因此 G-1 的身份对象若引用 `coordinate_box_diagonal` 而**不带距离模型域**，其
  `certifiable` 标志必须为 **False**；G-1 的"身份一致"**不得**替 G-2 的时间上界背书
  （原预声明 §8.3 已写此原则，本勘误把它机械化为 `G2_BUDGET_NOT_QUALIFIED` 拒绝码）。
- **G-2 的生产参数资格仍未建立**：本轮**不**把 `T_clock_sup`、`n_max`、任何距离域承诺写入生产预算。

---

## 5. 状态

不改变任何科研状态；不追认任何既往结论；本轮不接线、不改生产件、不 commit/push。
