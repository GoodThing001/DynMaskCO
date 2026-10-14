# D3 A2/A3 装配回归：夹具有效性复核与最小修正（工程证据，非方法裁定）

- 轮次标识：`A1_D3_FIXVALID_5273739d`
- 日期：2026-10-14
- 状态：`FIXTURE_VALIDITY_REVIEW = COMPLETE_AS_ENGINEERING_EVIDENCE`
- `algorithmic_gain_claim: none`；`F_vs_D_Geo_effect: NOT_COMPARED`（本轮**未做**任何 F−D-Geo 效果比较）
- 未改动：`freeze_tree_7b79ac57/`、`MASKCO_code/`、历史套件原文、冻结数据、sealed 结果

## 0. 本轮授权边界

用户授权范围：仅限**既有测试的夹具有效性复核与最小修正**、冻结身份只读、不修改既有冻结树、不做 F−D-Geo 效果比较。
本轮新增文件只有两个（均在 `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/`）：

- `test_d3_assembly_fixture_validity.py`（新证据套件；历史套件原文未改）
- `d3_fixvalid_import_proof.py`（导入证明工具，非测试）

## 1. 三条失败断言的逐条运行时归因

历史套件 `test_d3_assembly_invariants.py` 在 worktree 与冻结树副本上**结果一致**：`20 PASS / 3 FAIL`（原始记录见 §4）。
三条失败断言及其运行时原因是**调用方测试代码的夹具缺陷**，不是实现缺陷：

| # | 历史失败断言 | 运行时原因（本轮实测） | 证据 |
|---|---|---|---|
| 1 | `A2 end-to-end: ... matches the ORIGINAL stepwise_sample step for step` | 夹具非法：`legal_fn=_legal_fn(seq_f)` 配 `mask_set=[5]` 只有 **1** 个合法动作，而 `score_fn=lambda st: [1.0, 1.0]` 返回 **2** 个分数。新装配层**按契约 fail-closed**：`failure_reason='score_len'`、`n_steps=0`、plan 回滚原样。于是该断言实际比较 `[] == s_f(1 步)` | 行 `V-A2a`/`V-A2b`；`records.V_new_assembly_on_illegal_fixture` |
| 2 | `A2 the only added step field is the pinned evidence field (incremental_distance)` | 同上：`n_steps=0` 使 `extra == set()`，不等于 `{'incremental_distance'}` | 同上（`n_steps`、`n_records`） |
| 3 | `A3 the success path installs exactly once` | 该调用**没有传 `certify_fn`**，而 `INSTALL_REQUIRES_CERTIFICATION=True`，事务在采样前拒绝：`failure_reason='no_certifier'`、`n_installs=0`、`install_fn` 调用 0 次、无异常逃逸 | 行 `V-A3`；`records.V_frozen_A3_call_no_certify_fn.ledger` |

补充对照：同一非法夹具在**旧的** `mpre_policy.stepwise_sample` 上仍返回 1 步（旧路径没有 `len(scores)==len(legal)` 守卫），这正是历史 A2 断言拿"1 步"去比"0 步"的来源（行 `V-A2c`）。

**结论（对应 1/2/3 的归因）**：三条失败均为**夹具/调用方缺陷**；新装配层的 fail-closed 行为（`score_len` 回滚、`no_certifier` 拒绝）是**正确**的，不应被"放宽认证"或"改判夹具"来掩盖。

## 2. A2 合法分数长度对照（同客户、两动作、精确并列）

夹具：同一客户 5，两个合法锚定动作（`position=1` / `position=0`），`len(scores)==len(legal)==2`，`incremental_distance` 完全相同（精确并列），并且**规范键顺序与 legal 顺序相反**（run1：legal 头 = 规范键较大者）。

| 运行 | legal 顺序（position） | 规范键头索引 | 装配记录的首次选择（步记录） | 历史 F 投影 |
|---|---|---|---|---|
| run1 | `[1, 0]` | `1`（即 legal 尾） | 索引 **0** → position 1 | 索引 0 |
| run2 | `[0, 1]` | `0` | 索引 **0** → position 0 | 索引 0 |

- 行 `C-...-on an exact tie the assembly RECORDS the FIRST action in legal order as chosen, matching the historical F projection`：run1/run2 **均 PASS**。
- 行 `C3`：run1 中规范键头（索引 1）与 legal 头（索引 0）**确实不同**，而记录的选择仍是 legal 头 → 该夹具真的能区分"规范键并列规则"与"legal-order-first"，且实测规则是 **legal-order-first**（与 REV3 主规则一致）。
- 规范键仍被完整记录供审计（`len(canonical_keys)==len(legal)`，行 `C-...-canonical keys are still recorded for audit`）。
- 行 `C4`：改用**两个不同客户**（客户 5、4，规范序反向）复测，记录的首选客户 = 5 = legal 头 = 历史投影。

**附加不变量（本轮新增，行 `C5`）**：装配层**实际 apply 的动作**与其**记录的所选动作**一致（同客户探针中两者都是客户 5）。首轮尝试曾把这一条判为不一致，根因是**我自己的匹配器**用候选元组 repr 去比 `FleetAction` repr（工具 bug，非实现 bug）；修正为按动作 repr 匹配后通过。该次失败尝试原文保留在 `raw/attempt1_stale_tree_17999524/`。

## 3. A3 双向对照（必备认证 vs 仅测试用确定性认证器）

| 分支 | 调用 | 实测结果 | 行 |
|---|---|---|---|
| 负向（无认证器） | `run_repair_transaction(...)` 不传 `certify_fn` | `failure_reason='no_certifier'`、`n_installs=0`、`install_fn` 0 次、无异常；第二次独立 ledger **逐字段一致** | `D1a`、`D1b` |
| 正向（仅测试用认证器） | 同上 + `certify_fn=_test_only_deterministic_certifier` | `n_installs=1`、`install_fn` 恰好 1 次、认证器恰好调用 1 次、`ok is True` | `D2` |
| 身份绑定 | 同上 | `installs[0].plan_hash == installs[0].certified_plan_hash == '2eaf6bedd0923ee1'`；认证器身份被台账原样记录为 `__main__._test_only_deterministic_certifier` | `D2b`、`D3` |
| 反冒充 | 同上 | 台账中**不含** `certify_plan`、**不含** receipt（D8 安装收据未被触发）；认证器身份不是任何真实认证面（`compose_certification` 等），且定义在本测试进程内 | `D3` |
| 认证不越权 | 认证器存在 + **非法**分数长度夹具 | 仍然 `score_len` fail-closed、`n_installs=0` | `D4` |

结论：`certify_fn` 存在与否构成真正的双向开关；认证器只影响"是否允许安装"，**不能**绕过装配层的分数长度守卫；本轮使用的认证器**明确标注为仅测试用**（`TEST_ONLY_DETERMINISTIC_FIXTURE_CERTIFIER`），既非 `certify_plan` 也未产生 D8 收据。

## 4. 三套回归 + 新套件的原始记录

四套件各跑 **两处**（冻结树副本 / worktree），原始 stdout/stderr 全部落盘（无删改）：

| 运行位置 | 套件 | exit | PASS | FAIL |
|---|---|---|---|---|
| tree | `test_d3_geo_adapter.py` | 0 | 36 | 0 |
| tree | `test_d3_adapter_risks.py` | 0 | 10 | 0 |
| tree | `test_d3_assembly_invariants.py`（历史，原文未改） | 1 | 20 | **3**（§1 已逐条归因） |
| tree | `test_d3_assembly_fixture_validity.py`（新） | 0 | **21** | **0** |
| worktree | `test_d3_geo_adapter.py` | 0 | 36 | 0 |
| worktree | `test_d3_adapter_risks.py` | 0 | 10 | 0 |
| worktree | `test_d3_assembly_invariants.py`（历史，原文未改） | 1 | 20 | **3** |
| worktree | `test_d3_assembly_fixture_validity.py`（新） | 0 | **21** | **0** |

原始文件：`raw/A1_D3_FIXVALID_<suite>_{tree,worktree}_{stdout,stderr}.txt`、行级 JSON `raw/A1_D3_FIXVALID_rows_{tree,worktree}_test_d3_assembly_fixture_validity.json`、汇总 `A1_D3_FIXVALID_RECORDS.json`。
历史套件**保持红色**是**预期**结果：它的调用方夹具本身非法，本轮按授权**不修改**该套件，只在新的证据套件里给出合法夹具的对照。

## 5. 新版本冻结身份 + 导入证明

- 冻结树：`freeze_tree_5273739d/`（24 文件 = 已登记的 22 文件身份 + 本轮 2 个新文件，逐字节校验，`copy_mismatches=[]`）
- 清单：`A1_D3_FIXVALID_FREEZE_MANIFEST_5273739d.json`（含每个文件 sha256、创建时 HEAD、`historical_trees_modified=false`、`maskco_code_modified=false`）
- 身份：`A1_D3_FIXVALID_IDENTITY.json`（`is_freeze_seal=false`、`is_method_approval=false`）
- **导入证明**：`raw/A1_D3_FIXVALID_import_proof_tree_{stdout,stderr}.txt` + `.json` → **24 / 24 模块 `FROM_TREE`**，`VERDICT: ALL_FROM_TREE`；`d3_geo_adapter.__file__` 指向树内副本；关键常量实测为
  `EPS_D_PRIMARY_REL=0.0`、`EPS_D_PRIMARY_ABS=0.0`、`TIE_BREAK_LEGAL_ORDER='legal_order_first'`、`TIE_BREAK_CANONICAL='canonical_key'`、`INSTALL_REQUIRES_CERTIFICATION=true`。
- 辅助对照：`raw/A1_D3_FIXVALID_import_proof_worktree_*.txt` 是对**整个** `scripts/{simulation,tests}`（147 模块）的无范围扫描，其中 9 个与本轮身份无关的旧模块导入失败（缺 `tensorboardX`、或引用已移除 API），因此其 exit=1；24 个登记模块在该扫描中同样 `FROM_TREE`。**权威导入证明是树内那一次**。

## 5b. 字节复现检查（`git archive HEAD`）与一处**检查工具**缺陷的更正

提交号：`722f0f9`（`codex/o0cc-cal-phys` 与 `main` 同步，`ahead/behind 0 0`）。

- **冻结树复现**：`git archive HEAD` 取出 `freeze_tree_5273739d/` 后逐文件比对 → **24/24 与登记的 sha256 一致（0 mismatch）**。该目录自带 `* -text` 钉死（实测 `git check-attr text` = `unset`），因此不受行尾转换影响。
- **更正（保留原始失败，不改写为通过）**：我的第一次字节复现检查用"解包 `git archive` 结果"来比对两个新增的 `scripts/tests/*.py`，报出 2 处 mismatch。**这是检查工具缺陷，不是仓库缺陷**。实测原因：这两个路径的 `text` 属性为 `unspecified`，`git archive` 导出时应用了 `core.autocrlf=true`，于是解包结果为 CRLF：
  - `scripts/tests/test_d3_assembly_fixture_validity.py`：解包 488 CRLF / 0 LF，25329 字节，sha `10f47d75…`；而 worktree 与 **blob** 均为 488 LF / 0 CRLF，24841 字节，sha `741d7c91…`
  - 同一文件在冻结树内的副本：解包 0 CRLF / 488 LF，24841 字节，sha `741d7c91…` = worktree
- **权威检查**：`git cat-file blob HEAD:<path>` 与 worktree 逐字节比对 → 6/6 `SAME`（2 个新文件、3 个历史 D3 测试文件、`d3_geo_adapter.py`）；两个新文件的**登记 sha256 与 worktree/blob 完全一致**（`registered_identity_matches_worktree_bytes: {both: True}`）。
- 记录的限制：在 `core.autocrlf=true` 且 `scripts/` 无 `* -text` 钉死的情况下，`scripts/tests/*.py` 的**全新检出**会物化为 CRLF，因此这两个源路径的登记 sha 只能由**钉死的冻结树副本**复现；这与既有历史 D3 测试文件的情况完全相同（`text: unspecified`），是本轮之前就存在的仓库级状态，本轮未改变。本轮**未**新增 `scripts/**` 的行尾钉死，因为那会改变既有冻结文件的检出字节，超出本轮授权。
- 原始记录：`raw/A1_D3_FIXVALID_GIT_ARCHIVE_BYTE_CHECK.json`

## 6. 保留的失败尝试（未改写为通过）

| 尝试 | 保留位置 | 失败原因（工具/流程，非实现） |
|---|---|---|
| attempt1 | `raw/attempt1_stale_tree_17999524/`（23 文件） | 匹配器用候选 repr 比 `FleetAction` repr → 误判"应用动作与记录动作不一致"；树运行缺 `coldchain_state` 依赖路径。旧树（未登记）已删除，仅原始记录保留 |
| attempt2 | `raw/attempt2_helper_path_incomplete/`（23 文件） | 树内运行时未把 `scripts/coldchain` 等**非冻结**辅助目录加入回退路径 → 冻结模块导入失败 |

## 7. 本轮**不**证明的（claim limits）

1. 未做任何 F 与 D-Geo 的**效果比较**；本轮任何结果都不得读作"算法可比/不可比"或任何增益。
2. 三条历史失败被归因为**夹具缺陷**，这**不**等于"历史套件已被修复"——该套件仍红，且本轮**无权**修改它。
3. `D2` 的认证器是**仅测试用**桩，不构成任何 C0 物理/合同认证，也不代表 D8 安装通道已被打通（台账中无 receipt）。
4. `C5` 的"记录=应用"一致性只在两个同进程探针夹具上验证，样本极小；不是全装配路径的不变量证明。
5. 冻结树是**源码身份**，不是方法批准，也不是科学结论；`PENDING` 值仍未冻结。
6. 导入证明是**同进程、同用户、只读脚本**证据，不是外部第三方复现。

## 8. 对用户裁定规则的回答

用户规则："若合法夹具修正后仍失败，则判定为实现缺陷——不得继续归咎夹具或放宽认证。"
本轮结果：**合法夹具（`len(scores)==len(legal)`、精确并列、同客户两动作、规范序反向）全部通过**；失败仅存在于**非法夹具 + 缺失认证器**的历史调用中，且原因是历史调用本身不满足接口契约。因此**不构成实现缺陷**；不修改历史套件、不放宽认证、不做效果比较。
