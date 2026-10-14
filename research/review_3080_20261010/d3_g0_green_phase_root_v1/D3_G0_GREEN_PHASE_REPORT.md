# D3-G0 G0-Safety Green Phase（第一批）报告

**日期**：2026-10-14 · **基准 commit**：`87c6a5c`（本地 `HEAD` == `origin/main`，起点干净）
**性质**：修复**生产源码**（最小修复，仅 `d3_geo_adapter.py` 一层）；新增验收测试与证据；**不改历史测试、不改冻结树、不改实验数据与裁决文件**
**授权**：仅 G0-Safety 第一批。A-3b 保持 **OPEN**，A-5c 持久化与契约/状态身份留第二批。

---

## 0. 结论摘要

**五项验收全部满足**，`chain_complete: true`：

| 验收项 | 结果 | 证据 |
|---|---|---|
| 1. 适用反例转为不变量满足，且**不以异常崩溃代替 fail-closed** | ✅ G 块 12/12 | 绿相位差分 `G1–G12`，每例断言 `exception is None` |
| 2. A1–A4 / R1–R4 / T-G1～T-G9 继续通过；被取代的成功用例已登记留痕 | ⚠️ **部分**：T-G `36/0`、R `10/0` 不变；A1–A4 **20/3**，3 条被取代断言逐条登记 | §4.2 |
| 3. 同一合法输入下 F 的动作选择与历史实现一致 | ✅ E 块 5/5 | 与冻结修复前模块逐字段比对 |
| 4. 所有失败路径保持已承诺状态不变，不安装部分计划 | ✅ G12 | 采样器返回原始计划 + 事务 0 安装 + 源计划未被就地改写 |
| 5. 测试/生产源码/环境/来源身份分别记录，不动冻结树与历史裁决 | ✅ | `GREEN_PHASE_IDENTITY.json` |

**第 2 项必须诚实标注为"部分"**：`test_d3_assembly_invariants.py` 里有 3 条断言**本质上就是被修掉的那个缺陷**（要求"无认证器也能安装成功"、以及给 1 个合法动作喂 2 个评分），它们与"强制认证门"和"评分长度契约"在逻辑上不可共存。我**没有修改历史测试文件**，而是把替换关系逐条登记进 `G0_SAFETY_DELTA.json`，并在新验收测试里给出对应的正确命题。详见 §4.2。

**修复规模**：`d3_geo_adapter.py` 25,486 → 41,770 字节（+280 / −19 行），sha256 `f7480925…` → `099c3517…`。

---

## 1. 批次划分（按你的分组）

| 组别 | 内容 | 本批状态 |
|---|---|---|
| **G0-Safety** | 强制 C0 认证门、评分 shape/长度校验、GEO 索引对齐、事务幂等、认证↔安装身份绑定、三态回滚 | ✅ 本批完成 |
| **G0-Evidence** | 持久化 `P_after_install`、真实 `VehicleState` 承诺冻结、完整装配层与身份闭包 | ⏸ 未动，明确留空 |

---

## 2. 冻结修复前版本（先冻结，再修复）

改任何生产代码**之前**执行 `freeze_prefix.py`，把 5 个文件按字节冻结到
`d3_g0_green_phase_root_v1/frozen_prefix/`，并做**三重**校验：

1. 冻结副本 == 当时工作区字节；
2. 工作区字节 == `87c6a5c` 提交里的 blob 字节（`git cat-file blob`）；
3. 修复后重跑时，校验被测 `PREFIX` 模块仍等于该 blob（绿相位 `F` 块，1/1）。

`PREFIX_FREEZE.json` → `all_frozen_copies_verified: true`。

**为什么基准是 `87c6a5c` 而不是 `dbc5971`**：`d3_geo_adapter.py` 在两个提交间字节相同（`f7480925…`），但 Red Phase 的反例 harness 只存在于 `87c6a5c`，冻结它才能把"修复前的反例原件"也纳入差分。

**顺带发现的一个陷阱**：`git archive` 在本仓会注入 CRLF（`core.autocrlf=true`，且 `scripts/` 无 eol 属性），导出后 adapter 变成 25,984 字节、sha256 `5d1b73df…`。验收驱动因此先做 CRLF→LF 归一化并**校验 sha256 回到 `f7480925…`**，否则拒绝报告基线（否则就会拿一个"看起来一样"的源码去报基线）。

---

## 3. 最小修复：5 条（S1–S5）

全部集中在 `d3_geo_adapter.py`。判决原则：**非法/未验证输入 = 新的安全语义（干净 fail-closed）；合法输入 = 决策路径不变**。

### S1 评分向量契约（A-1a / A-1b / A-1c）
在 `scores = np.asarray(score_fn(state))` 之后、`np.isfinite` 之前加两道守卫：
`ndim != 1` → `_rollback('score_shape')`；`len(scores) != len(legal)` → `_rollback('score_len')`。
`budget.bump('n_actions_scored', …)` 移到守卫之后（被拒绝的向量不算"已评分"）。
顺带修掉一个连带缺陷：原 `score_exact_ties` 里 `scores[idx]` 在 2-D / 短向量下会在**适配器自带的证据记录器**里崩，现在守卫在其之前，`scores[idx]` 恒为标量。

### S2 GEO 索引对齐（A-1d，含 A-1e 的索引）
GEO 分支使用 `state[GEO_SCORE_KEY]` 与 `state['vehicle_target_vid']`，两者都必须与 `legal` 等长，否则 `_rollback('geo_len')`。原因写进注释：`select_geo` 的 tie 集锚定在"它拿到的那个向量"的 `min` 上，截断向量会**静默丢掉真实最小值**；`vehicle_target_vid` 则决定 tie 的 canonical key。缺键/不可下标的状态供给方也一并干净拒绝。

### S3 安装边界强制认证（A-2）
- `stepwise_sample_d3` **仍是纯候选生成器**，`certify_fn=None` 不报错（它不安装任何东西）。
- `run_repair_transaction`：**无认证器即拒绝**，且在**采样之前**拒绝 → `failure_reason='no_certifier'`、0 安装、返回原始计划、不抛异常。
- `INSTALL_REQUIRES_CERTIFICATION = True` 把这条契约声明化、可测。
- `InstallLedger.finish` 的安装门要求全部成立：`certified is True`、`certified_plan_hash` 非空、**安装计划哈希 == 认证计划哈希**、`certifier` 身份非空、`deadline_ok is not False`。不满足 → `n_refusals` + `action='refuse'`，**不是** rollback（没改过任何东西）。

> 这直接回应你的第 2 点：**"认证次数 ≥ 1" 不是充分条件**。绿相位 `G2` 就是"认证过 A 计划却想安装 B 计划"的反例，被 `certified_plan_identity_mismatch` 拒绝；`G3` 覆盖决策时限。合同/状态身份绑定需要装配层，本批**声明为未做**（§6）。

### S4 身份绑定 + 不可变（A-4b）
认证时记录 `diag['certified_plan_hash']`（认证的**就是那个计划对象**）、`diag['certifier']`（module.qualname）。账本改为存**值拷贝** `_copy_plan`，并产出带 `plan_hash / certified_plan_hash / certifier` 的回执。`_plan_identity` 在任何异常下返回 `None` 而非抛出 —— 一个会在畸形候选上崩溃的闸门比会拒绝的闸门更糟，而 `None` 永远无法满足身份比较，所以不可哈希的计划必然被拒。

### S5 幂等 + 三态回滚（A-5b / A-5d）
- **一个账本 = 一笔事务**：`finish` 至多生效一次，第二次 → `already_finished` 拒绝（不是静默忽略、不是重复安装）。
- 回滚语义拆成三态，而不是把每次失败都记成"已成功恢复"：
  `candidate_discarded` / `committed_state_restored`（`rollback_fn` 真的跑了）/ `rollback_unverified`（没有 `rollback_fn`，恢复**未被证明**）。
- `n_rollbacks` 保持历史含义（"走过的失败路径数"），并新增 `n_state_restorations`、`n_rollback_unverified`、`n_refusals`、`n_pre_sampling_refusals`。采样前就被拒绝的不计入 `n_rollbacks`，因此它**永远不会被读成"回滚成功"**。

**刻意没有做的**：没有给 `select_historical_argmax` 加 ndim 断言（它就是 F 的规则，必须保持 `np.argmax`）；没有给适配器加 `committed_next` 字段；没有加任何持久化接口。

---

## 4. 证据链

### 4.1 绿相位差分（`test_d3_g0_green_phase.py`，30/30，`chain_complete: true`）

同时加载**两个模块**：`PREFIX`（冻结修复前，字节已校验）与 `LIVE`（修复后）。

| 块 | 含义 | 结果 |
|---|---|---|
| **F** | 被测前缀 == `87c6a5c` 的 blob | 1/1 |
| **P** | 修复前**必须违反**各不变量（否则"反例"不成立） | 9/9 |
| **G** | 修复后**必须满足**，且不得以异常代替 fail-closed | 12/12 |
| **E** | 合法输入下两模块决策完全一致 | 5/5 |
| **R** | 本批不负责的残留项**仍声明为未闭合** | 3/3 |

P↔G 对应关系（节选，全部 9 对见 JSON）：

| 项 | 修复前（P 块实测） | 修复后（G 块实测） |
|---|---|---|
| A-2 | `certify_fn=None` → **安装 1 次**、认证数 0 | 采样前拒绝，0 安装，返回 `False`，`no_certifier` |
| A-2 身份 | 只有 A 的哈希概念都不存在 → **照样安装 B** | `certified_plan_identity_mismatch`，0 安装 |
| A-1a | 1 个评分驱动 3 步、`failure_reason=None`（2 个候选从未被评分） | `score_len`，0 步，`n_actions_scored=0` |
| A-1b | argmax 落在合法区间内 → **静默成功**；漂出范围才 `IndexError` | 两分支都 `score_len`，无异常 |
| A-1c | 原语把 2-D 展平 → 索引 2 指代第 2 行以外的候选 | 适配器 `score_shape`，**永不进入选择** |
| A-1d | 截断 GEO 向量 → 静默选客户 4（真实最优是 6），`failure_reason=None` | `geo_len`，无异常 |
| A-4b | 无身份字段 + 账本按引用保存（外部改写即"改口"） | 回执哈希 == 认证哈希 == 计划哈希；外部改写**不影响**账本，且变异确实发生（`mutation_was_real`） |
| A-5b | `finish` 两次 → **安装 2 次** | 第二次 `already_finished`，恰好 1 次 |
| A-5d | `rollback_fn=None` → 计 1 次回滚、无任何标记 | `rollback_unverified=True`、`committed_state_restored=False`；有 `rollback_fn` 时才 `True` |

`G12`（验收第 4 条）对 `no_legal / nan / timeout / score_len` 四条失败路径同时验证：采样器返回的就是**原始计划**、事务**0 安装**、源计划对象**未被就地改写**、全程**无异常**。

### 4.2 历史三套测试：修复前 / 修复后对照（登记式留痕）

在 **`87c6a5c` 冻结树**（字节已校验）与**当前源码**上分别跑同一份测试文件：

| 套件 | 修复前 | 修复后 | 差异 |
|---|---|---|---|
| `test_d3_geo_adapter.py`（T-G1..T-G9 + X1/X2） | 36 / 0 | **36 / 0** | 无 |
| `test_d3_adapter_risks.py`（R1..R4） | 10 / 0 | **10 / 0** | 无 |
| `test_d3_assembly_invariants.py`（A1..A4） | 23 / 0 | **20 / 3** | 3 条 |

**被取代的 3 条断言**（`G0_SAFETY_DELTA.json → superseded_assertions`）：

1. `A3 the success path installs exactly once` —— 修复前 `run_repair_transaction` **不带认证器**却要求安装成功。这条断言**编码的就是 G0-01 反例本身**，与强制认证门不可共存。对应新命题：`G1/G2/G3/G10`。
2. `A2 end-to-end: … matches the ORIGINAL stepwise_sample step for step` —— 该 fixture 给 **1 个合法动作喂了 2 个评分**（`legal` 长度 1、评分长度 2），只因展平 argmax 恰好落在范围内才通过，即它**走的正是 A-1b 缺陷路径**。对应新命题：`E2/E3` 在**对齐**输入上证明 F 等价。
3. `A2 the only added step field is the pinned evidence field` —— 与第 2 条同一 fixture，非独立断言（0 步时 key 差集为空）。

**分类**：三条均为 `SAFETY-CONTRACT UPDATE, not a regression`。我没有改写这些历史测试文件（字节与 `87c6a5c` 一致），因此"历史套件继续全绿"这一条**无法字面成立**；能成立且已证明的是：**除这 3 条被登记的断言外，其余 69 条断言在修复前后结论完全一致**（36+10+20 vs 36+10+23 中未被取代的部分）。

### 4.3 Red Phase harness 在修复后的重跑（对照，不覆盖原件）

Red Phase harness 字节未动（`779cafaf…`），重跑后**先把它的修复前矩阵复制到 `D3_G0_RED_PHASE_MATRIX_post_fix.json`，再从 git 还原原矩阵**（验收驱动校验还原后的 sha256 == `327a9fb3…`，否则终止，绝不留下被覆盖的 Red Phase 证据）。

| 修复前（Red Phase 原件） | 修复后（同 harness、新代码） |
|---|---|
| REPRODUCED 9 / NOT_VERIFIABLE 1 / CONTROL 2 | **NOT_REPRODUCED 5 / FAIL_LOUD_ACCEPTABLE 1 / REPRODUCED 3 / NOT_VERIFIABLE 1 / CONTROL 2**，`harness_valid: true` |

剩下的 3 条 REPRODUCED 逐条有解释，**都不是未修复的缺口**：

- `G0-03 (A-1c)`：harness 断言在**选择原语**层（`select_historical_argmax` 对 2-D 输入返回展平 argmax）。该原语**就是 F 的规则**，必须保持 `np.argmax`；修复点在**适配器边界**，守卫在其之前，因此该路径已不可达。这是**测量点差异**，不是缺口 —— 适配器层不变量由 `G6` 证明。
- `G0-07 (A-1d)`：**我自己的 harness 缺陷**，见下。
- `G0-10 (A-5c)`：**本批刻意不做**（G0-Evidence 第二批），因此它必须仍然是 REPRODUCED。

#### ⚠️ 我主动纠正的 Red Phase 缺陷：`G0-07 (A-1d)`

Red Phase 的 G0-07 使用 `mask_set=[4]`，导致 `legal` 只有 **1 个**候选 —— "选中列表第 0 位客户 4"是**构造上必然**的，与 GEO 向量是否截断无关；而 `observed` 里的 `"n_legal": 3` / `"delta_len": 2` 是**硬编码字面量，不是测量值**。该用例**从未真正检验过 A-1d**。

**判定它无效的硬证据**：修复后 G0-07 **仍然报告 COUNTEREXAMPLE_REPRODUCED** —— 如果它测的是修复后的代码路径，这是不可能的。

**处置**：Red Phase 文件**不改写**（保留为"曾经的失败记录"），A-1d 改由绿相位差分在**正确 fixture** 下重新建立：
- `P6`（前缀，`mask_set=[4,5,6]`，GEO 向量截断为 2）：`failure_reason=None`，静默选客户 4，而真实最优是客户 6 → 反例**成立**；
- `G7/G8`（修复后）：干净 `geo_len`，无异常。

所以 A-1d 的结论不变（确为真反例、已被修复），但**原 Red Phase 的测量无效**，此点在 `G0_SAFETY_DELTA.json → red_phase_harness_defect` 中逐字记录。

---

## 5. 你的三条方法学修正如何落实

1. **分批处理** → 本批只做 G0-Safety 五项；A-3b / A-5c / 契约-状态身份 / 完整装配层全部留在第二批，并在绿相位 `R` 块**主动断言它们仍未闭合**（`R1` 断言持久化接口面为空、`R2/R3` 断言适配器既没有 `committed_next` 参数也没有在正文提及它）。我没有用同名字段去"修复" A-3b。
2. **判据精确化** →
   - A-2：安装门不再看"认证次数"，而是要求**认证哈希 == 安装哈希** + 认证器身份 + 决策时限；`G2/G3` 是这个判据的反例测试。
   - A-5d：拆成 `candidate_discarded` / `committed_state_restored` / `rollback_unverified`。`G11` 同时验证"没有 `rollback_fn` ⇒ 未证明恢复"与"有 `rollback_fn` ⇒ 确实恢复（`n_state_restorations=1`）"。
3. **不把失败原件改写成通过** → Red Phase harness 与三套历史测试**字节未动**；新验收测试是独立文件；差分对**冻结修复前版本**与修复后版本分别运行；历史套件在**冻结树**里的基线（36/0、10/0、23/0）与修复后结果并列留档；被取代断言逐条登记，且明确写出哪些新命题取代了它。

---

## 6. 仍未闭合（不得读作已修复）

| 项 | 状态 | 归属 | 说明 |
|---|---|---|---|
| **A-3b** 承诺/已执行前缀冻结检查 | **OPEN** | G0-Evidence 第二批 | 真实护栏在 `VehicleState` + mask 池，在本计划级适配器**之外**；加同名字段是假修复 |
| **A-5c** 持久化 `P_after_install` | **OPEN** | G0-Evidence 第二批 | 回执仅内存，随进程消失；本批未加任何持久化面 |
| 认证器的**合同/状态身份**绑定 | **PARTIAL** | G0-Evidence 第二批 | 计划身份已绑定（哈希相等）；合同/状态身份需要装配层 |
| **A-1c** 原语层展平 | 设计上不可达 | — | F 规则原语保持 `np.argmax`；守卫在其之前 |
| 完整装配层与身份闭包 | 未开始 | G0-Evidence 第二批 | 适配器至今**无生产调用方**，这仍是最大的结构性缺口 |

---

## 7. 未做 / 未授权

未训练、未采集、未做 F−D 对比、未重采 32 状态、未跑 G1 smoke / G2 batch、未触碰最终测试种子；未加载任何检查点、未使用科学数据。未修改 `MASKCO_code/`、未修改任何冻结树或历史裁决文件。

**并发写入事件（已记录）**：会话进行中，`maskco` 自动化修改了 `CC_Compare/tools/role_c_timing.py` 与 `CC_Compare/tools/test_role_c_resource_timing.py`（18:13:49 / 18:14:05，角色 C 计时可观测性）。这两处**与本批无关**，投递时用显式 `--paths` 精确暂存而**未捕获**；幻影 gitlink `CC_Compare/MAPT_code` 一如既往排除。

**科学判断不变**：D3 现有可复核的工程反例证据与已修复的安全缺口，**仍无可归因的 F−D 效果证据**；I3 正式负结果、旧 D2 `not_qualified`、`stable_online_gain=False` 均保持不变。

---

## 8. 复现

```powershell
$py = 'D:\AA_Py\A_Anaconda\python.exe'      # 必须：mpre_policy 链路 import jax
cd D:\PyCharm_\MASKCO-Main

# 1) 冻结修复前版本（已执行；幂等，会重算并校验三重身份）
& $py -B research\review_3080_20261010\d3_g0_green_phase_root_v1\freeze_prefix.py

# 2) 绿相位差分：前缀违反 / 修复后满足 / 合法输入不变 / 残留仍开
& $py -B C-VRP_Cold-chainVehicleRoutingProblem\scripts\tests\test_d3_g0_green_phase.py

# 3) 完整验收链：冻结树基线 + 修复后 + Red Phase 重跑 + 安全契约差异登记
& $py -B research\review_3080_20261010\d3_g0_green_phase_root_v1\run_green_phase_acceptance.py
```

第 3 步会自行用 `git archive 87c6a5c` 物化冻结树、归一化 CRLF 并校验 sha256；若被测源码与提交字节不符则**拒绝出报告**。

---

## 9. 产物

| 文件 | 作用 |
|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py` | 生产代码，最小修复（修复前 `f7480925…` → 修复后 `099c3517…`） |
| `…/scripts/tests/test_d3_g0_green_phase.py` | 绿相位差分验收（30 项，`chain_complete: true`） |
| `…/scripts/.gitattributes` | 6 个已登记字节容器的行尾冻结（精确限定，未波及 34 个 CRLF shell 脚本） |
| `d3_g0_green_phase_root_v1/frozen_prefix/` + `PREFIX_FREEZE.json` | 修复前字节冻结与三重校验 |
| `d3_g0_green_phase_root_v1/d3_geo_adapter.green_phase.diff` | 最小修复的统一 diff |
| `d3_g0_green_phase_root_v1/D3_G0_GREEN_PHASE_MATRIX.json` | 绿相位机器可读矩阵 |
| `d3_g0_green_phase_root_v1/G0_SAFETY_DELTA.json` | 验收链 + 3 条被取代断言 + Red Phase harness 缺陷 + 残留项 |
| `d3_g0_green_phase_root_v1/D3_G0_RED_PHASE_MATRIX_post_fix.json` | Red Phase harness 在修复后的判定（修复前原件已从 git 还原并校验） |
| `d3_g0_green_phase_root_v1/GREEN_PHASE_IDENTITY.json` | 生产/测试/环境/来源身份分列记录 |
| `d3_g0_green_phase_root_v1/raw/` | 全部原始 stdout/stderr（冻结树 3 套、修复后 3 套、Red Phase 重跑、绿相位；stderr 全为 0 字节） |
| `d3_g0_green_phase_root_v1/freeze_prefix.py`、`run_green_phase_acceptance.py` | 冻结与验收驱动 |
