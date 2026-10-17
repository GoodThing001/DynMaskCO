# G-TMIN 最小安全修复候选 · 七要素预声明（2026-10-17）

> **性质：预声明（pre-registration），先于实施登记。** 本文件在写任何候选代码之前登记假设、对照、
> 判据、参数与次数、预期方向、失败条件、资源门与代码身份。
> **本轮不启动正式科研实验、不修改生产认证器、不接生产安装入口、不自动 commit/push。**
> 全部既有科研状态保持不变。

---

## 0. 代码身份（执行前登记）

| 项 | 值 |
|---|---|
| HEAD | `759fe7fb81dd869c212a28eed352f82ff99a1b1e` |
| 被审对象 `scripts/m1b_adapter/m1b_q2_certifier.py` | `8bd105f6be0bb9dbbacae133e9f2d905ad1fc544c4f879dec5ce7aeaf399ef35` |
| `scripts/m1b_adapter/m1b_quality_certificate.py` | `2aa801d09c26e6851e75ca4c1672ff9f84f1d59d3e5b95d402fcfc8f96ba4d22` |
| `scripts/coldchain/coldchain_contract.py` | `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| 候选模块（待创建） | `scripts/q3_qualification/q3_t_min_candidate_v2.py` |
| 测试（待创建） | `scripts/tests/test_q3_g_tmin_v2.py` |
| 既有第一轮候选（**只读，不改**） | `scripts/q3_qualification/q3_t_min_candidate.py` |

---

## 1 假设（assumption）

**H-1**：`m1b_q2_certifier.py:122-139` `_resolve_t_min()` 在 `verified_t_min is not None` **且**
`self.env is not None` 时，只做 `float()` / 非 NaN / `0 ≤ v ≤ 1e9` 三项检查，即
`t_min_source = 'VERIFIED_AGAINST_SUPPLIED_ENV'` 并**返回该值**；它**从不读取 `env` 的任何字段**。
因此 `env = object()` 即构成"合格见证"，`verified_t_min` 可把 `H_max` 压缩到 0，
把 `q_lb` 抬到初值（8.999999999 而非 8.54807998017）。**该通道与已封堵的裸 `t_min` 数学同型。**

**H-2**：该缺陷**未被现有负控覆盖**。`test_m1b_q2_final_entry_controls.py` 的 B2 只传**裸 `t_min`**
（第 131 行 `t_min=t_min`），从未把 `verified_t_min` 作为关键字传入，也从未在构造期传 `env`。
**B2 的 PASS 与 G-TMIN 危险可以同时存在。**

**H-3**：`t_min` 只通过 `certify_quality_lb(..., t_min=...)` 影响
`H_max = T_end + ε_return − t_min` 与 `q_lb = max(0, q0_safe − k·H_max)`，故 `dq_lb/dt_min = +k > 0`：
**降低 `t_min` 只会让认证更难通过**。

**H-4**：由于**不存在**可独立重算的时间下界见证（第一轮已证：仓内无此类工件），
本轮**唯一被接受的计算输入**是 `t_min = 0`；非零 `t_min` 的可信见证机制**本轮不实施**。

---

## 2 唯一主对照（the single primary control）

> **主对照 = 生产解析器 `_resolve_t_min` 的逐位复刻（`production_resolver_replica`），
> 在同一组输入上逐位比较"是否改变了数值结果"。**

**为什么只有一个主对照**：本轮判据是"候选**绝不**比生产更宽松"，这只需要一个方向性比较对；
引入第二个对照（例如"候选是否更准确"）会把一个**安全性**问题变成**精度**问题，而精度不是本轮
的目标。所有其它行为（标签、拒绝记录、收据语义）都是**辅助判据**，不得替代主对照。

**主对照的输入集（14 组，预先固定，不得事后增删）**

| # | 输入 | 生产 `t_min`（已实测） | 生产标签 |
|---|---|---|---|
| 1 | 无参数 | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 2 | 裸 `t_min=1e6` | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 3 | 裸 `t_min=24.0` | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 4 | `verified_t_min=1e6`（无 env） | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 5 | `verified_t_min=1e6` + `env=object()` | **1e6** | **`VERIFIED_AGAINST_SUPPLIED_ENV`** |
| 6 | `verified_t_min=24.0` + env | **24.0** | **同上** |
| 7 | `verified_t_min=0.0` + env | 0.0 | **同上** |
| 8 | `verified_t_min=-5.0` + env | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 9 | `verified_t_min=nan` + env | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 10 | `verified_t_min=inf` + env | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 11 | `verified_t_min="1e6"` + env | **1e6** | **同上** |
| 12 | `verified_t_min=1e10` + env | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 13 | `verified_t_min=None` + env | 0.0 | `CONSERVATIVE_CLOCK_ORIGIN` |
| 14 | `verified_t_min=True` + env | **1.0** | **同上** |

---

## 3 判据（criteria，全部机械化、可否决）

| # | 判据 | 通过条件 |
|---|---|---|
| **C1**（主） | 方向安全 | 对全部 14 组：`candidate.t_min ≤ production.t_min`；且 `candidate.t_min == 0.0` |
| **C2** | 数值不变性 | 对第 1/2/3/4/8/9/10/12/13 组：候选的 `H_max`、`q_lb` 与生产**逐位相同** |
| **C3** | 严格失败语义（**本轮新增**） | 对第 5/6/7/11/14 组（`verified_t_min` 非 None）：候选 `refusal is True`、标签 `UNVERIFIED_T_MIN_REFUSED`，且 `apply_to_receipt` 把收据置为 `REFUSED`、`composite_pass=False` |
| **C4** | 禁用标签 | 候选代码中**不存在**字符串 `VERIFIED_AGAINST_SUPPLIED_ENV` 作为标签；该字符串出现在收据里即被 `receipt_gate` 判 `REFUSED` |
| **C5** | `env` 不被读取 | 传入"任何属性访问都抛异常"的 env，候选**不抛异常**，且 `env_consulted is False` |
| **C6** | 不得抬高 `q_lb` | 对全部 14 组：`candidate_q_lb ≤ production_q_lb + 0`（逐位） |
| **C7** | 不得缩小 `H_max` | 对全部 14 组：`candidate_H_max ≥ production_H_max` |
| **C8** | 无 FALSE_PASS | 在候选的收据语义下，**没有任何**输入会让一次本应被拒绝的认证变成 `composite_pass=True` |
| **C9** | 不修改冻结件 | `m1b_q2_certifier.py`、`m1b_quality_certificate.py`、合同源码 sha256 不变 |
| **C10** | 旧测试回归 | `test_m1b_q2_final_entry_controls.py` 仍为 6/6、`0 FALSE_PASS`；第一轮 `test_q3_g_tmin_candidate.py` 仍 11/11 |
| **C11** | B2 缺口独立覆盖 | 候选测试**自己**构造 `verified_t_min` + 构造期 `env` 的旁路，并证明生产仍被绕过（记录事实），候选阻断 |

---

## 4 参数与次数（parameters and counts）

| 项 | 值 |
|---|---|
| 输入组合数 | **14**（§2 表，固定） |
| 判据数 | **11**（C1–C11） |
| 正控（应通过） | 9 组（1/2/3/4/8/9/10/12/13） |
| 负控（应拒绝） | 5 组（5/6/7/11/14） |
| 重复次数 | 每输入 **1 次**（纯函数、无随机性；`env=object()` 无身份） |
| 环境对象 | `object()`（与既有 B2 同型） + 1 个 `HostileEnv`（属性访问抛异常） |
| 实例 / 数据 | **不读**任何实例数据；只用合成常数与合同常量 |
| 随机种子 | 不使用 |
| 运行次数上限 | 测试套件 **1 次**；不重跑、不调参 |

---

## 5 预期方向（expected direction）

| 量 | 预期 |
|---|---|
| `t_min` | 恒为 `0.0` |
| `H_max` | 恒为 `24.000001`（默认合同、`T_end=24`、`ε_return=1e-6`） |
| `q_lb` | 恒为 `8.54807998017`（非 8.999999999） |
| 收据终态 | 第 5/6/7/11/14 组为 **`REFUSED`**；其余不受影响 |
| 相对生产 | 数值**只可能更保守**（相等或更低），认证**只可能更难通过** |
| 相对第一轮候选 | 第一轮只回落数值；本轮**额外把收据置 REFUSED**（严格更强） |

---

## 6 失败条件（failure conditions，任一成立即判本轮 FAIL）

| 编号 | 条件 |
|---|---|
| F-1 | 任一组 `candidate.t_min > 0` |
| F-2 | 任一组 `candidate_q_lb > production_q_lb`（出现任何抬高） |
| F-3 | 任一组 `candidate_H_max < production_H_max`（出现任何缩短） |
| F-4 | 第 5/6/7/11/14 组未能使收据 `REFUSED`（即仍是"静默回落 + 认证通过"） |
| F-5 | 候选代码或收据中出现 `VERIFIED_AGAINST_SUPPLIED_ENV` 作为**有效标签** |
| F-6 | 传 `HostileEnv` 时抛异常（说明读了 `env`） |
| F-7 | 出现任何 `FALSE_PASS`（应拒未拒） |
| F-8 | 修改了 `m1b_q2_certifier.py` / `m1b_quality_certificate.py` / 合同源码 |
| F-9 | 旧测试出现回归失败 |
| F-10 | 为实现候选而"顺手"引入非零 `t_min` 见证机制（超出本轮授权） |

**若任一条失败：记录 `BLOCKED_WITH_REASON`，不降低判据再试。**

---

## 7 资源门（resource gate）

| 项 | 上限 |
|---|---|
| 运行时间 | 测试套件 **< 30 s**（纯 CPU、无子进程网络、无 I/O 迭代） |
| 内存 | < 200 MB |
| GPU / 训练 | **禁止** |
| 数据读取 | **禁止**读取 `data/` 下的实例（只用合同常量） |
| 并发 | 单进程 |
| 外部网络 | 禁止 |
| 生产路径 | **不导入、不实例化任何生产安装入口**（`d3_geo_adapter.run_repair_transaction` 不得出现在测试里） |

---

## 8 交付物与"不做什么"

**交付**
1. `scripts/q3_qualification/q3_t_min_candidate_v2.py`（隔离候选，`INSTALLED = False`）；
2. `scripts/tests/test_q3_g_tmin_v2.py`（C1–C11，14 组输入）；
3. `results/_q3_next_20261017/gtmin/minimal_patch_for_authorisation.diff`
   —— **未应用**的生产最小 diff + 影响范围；
4. 中文最终报告（含兼容性清单）。

**明确不做**
- 不实施非零 `t_min` 的可信见证机制；
- 不修改生产认证器、不接生产安装入口、不改冻结源码；
- 不新增 gate 到 `composite_pass` 以外的任何判定；
- 不把候选的 PASS 写成"G-TMIN 已闭合"——闭合需要生产安装，属**单独授权**。

---

## 9 与状态的关系

本预声明**不改变任何科研状态**。`G-TMIN` 仍**未闭合**；
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、L2/A3 未建立、`stable_online_gain=False`、`HISTORICAL_BACKBONE_UNVERIFIED`、
C0 `AWAITING_USER_BUDGET_RULING`、最终测试种子 `SEALED` —— 全部保持不变。
