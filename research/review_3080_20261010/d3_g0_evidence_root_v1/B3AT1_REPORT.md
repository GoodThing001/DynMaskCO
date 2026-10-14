# D3-G0 Evidence B3-AT1 — 结果报告（授权提交事务的隔离实现）

- 阶段：`D3_G0_Evidence_B3AT1`（受限授权；`B3-AT2` 未授权）
- 预登记：`B3AT1_COUNTEREXAMPLE_SPEC.json`，sha256 `01803fd5428a83a8b262b49c6dfa4710b60c2f15e8c8d8bc52283b19bfe476a4`
- 设计：`B3AT1_DESIGN.md`
- 矩阵：`B3AT1_TRANSACTION_MATRIX.json`；原始输出：`raw/B3AT1_stdout.txt`、`raw/B3AT1_instance{0..5}.txt`、`raw/B3AT1_TRANSACTION_MATRIX_instance{0..5}.json`
- 命令：`python -B scripts/tests/test_d3_b3at1_transaction.py --instances 0 --heavy-cap 8`

## 0. 结论一句话

**AT1 的事务成功有机械证据（排他性 / 一次性 / 失败一致性各自可复现），全部只在隔离副本上；认证能力（D8）与生产写入资格（AT2/P2）均未建立**；事务**未接管**真实生产路径。

## 1. 预登记行与判定（13/13 全部到达预登记判定）

| id | 判定 | 关键证据 |
|---|---|---|
| `PRE` | `SETUP_OK` | 8 个捕获决策点全部只读；**谓词与逐候选结果全部记录**（`n_matching=8`） |
| `AT-F8` | `REFUSED_NO_MUTATION` | 边界内注入漂移 → `refusals=['state_changed_inside_boundary']`；`pre_drift_state_equalled_the_authorised_state=true`；`transaction_published_nothing=true`；边界内写入原语调用数 = **0**；车辆字段逐字节等于漂移后字段；**对照（不注入漂移）→ `COMMITTED`（边界内 3 次写入）**，故拒绝由漂移引起 |
| `AT-F10` | `DETECTED_NOT_PREVENTED` | 静态：排除事务模块自身后，`d3_*` 的写入原语调用 = **0**、live 状态赋值 = **0**；运行期：监测记录到越权 `_plan_and_commit`/`_commit`（含 caller 归属）且事务自身写入记为 `inside_transaction=true`。**`detected=true, prevented=false`** |
| `AT-ONCE` | `REFUSED_NO_MUTATION` | 第一次 `COMMITTED`（票据 `73776bf8e78839d6` 消费）；**新建事务对象**第二次 → `ticket_already_consumed`（`first_consumer=tx-first, attempted=tx-second-NEW-OBJECT`），状态不变 |
| `AT-PARTIAL` | `NO_PARTIAL_PUBLISH_ATOMIC_AND_PARTIAL_REPORTED_FOR_NONATOMIC` | 原子路径：候选构造期异常 → 失败尝试内写入原语调用 **0**、`n_published=0`；非原子路径：第 2 辆车抛异常 → **`PARTIAL_COMMIT`**，`n_published_before_failure=1` + 真实逐车字段，`rollback_claimed=false` |
| `AT-IDENT` | `REFUSED_NO_MUTATION` | `anchor_time+1.0`、`anchor_load+7.0`：**legacy `plan_hash` 不变**（`fd3380e889f3610c`），`commit_identity` 由 `548a68be…` 变为 `ed803116…` → `commit_identity_mismatch`，状态不变 |
| `AT-D8` | `BLOCKED_BY_D8` | `blocking_capability_gaps=['quality_salability']`、`certification_reasons=[]`、`executable_checks_all_ok=true`、状态不变；预算角色 **`ISOLATION_ONLY`** |
| `AT-POS-REAL` | `BLOCKED_BY_D8` | **真实门**下 `certification_complete_used=false`、`reached_boundary=false` → 事务**根本无法进入提交状态** |
| `AT-POS` | `COMMITTED_ON_ISOLATED_COPY` | 仅机制控制：`publish=atomic`、`ticket_state=consumed_in_boundary`；`real_qualification_claimed=false`、`install_permission_claimed=false`、`not_a_production_path=true`、`does_not_establish_install_feasibility=true` |
| `AT-WAIT` | `WAIT_BY_ENV_PROTOCOL` | 无待提交动作，`gate_invoked=false`，`counts_as_refusal_capability=false`，状态不变 |
| `ISOLATION` | `OK` | `non_test_modules_importing_the_transaction=[]`、`registered_on_env_anywhere=false`、历史旁路本批修改数 = **0** |
| `PREREG` | `OK` | 9 条反例全部覆盖，无遗漏 |
| `AUDIT` | `OK` | 无 hook 全新 env 重跑，整条生产轨迹哈希相同 → 事务与探针未扰动生产 |

`verdict_counts = {SETUP_OK:1, REFUSED_NO_MUTATION:3, DETECTED_NOT_PREVENTED:1, NO_PARTIAL_PUBLISH_ATOMIC_AND_PARTIAL_REPORTED_FOR_NONATOMIC:1, BLOCKED_BY_D8:2, COMMITTED_ON_ISOLATED_COPY:1, WAIT_BY_ENV_PROTOCOL:1, OK:3}`

**实例稳定性**：`--instances 0..5` 六次运行给出**完全相同**的 13 项判定（各实例矩阵已存 `raw/`）。

### 1.1 拒绝能力计账（不含糊）

`refusal_capability = {counted_as_capability: [AT-F8, AT-ONCE, AT-IDENT], waits_not_counted: [AT-WAIT], detection_not_prevention: [AT-F10]}`

- 只有**按指定理由归因**的拒绝计入；`AT-D8` 只算**品质维度**（且本行在矩阵里单列 `BLOCKED_BY_D8`，不并入能力清单）。
- `AT-WAIT` 永不计入。
- `AT-F10` 是**检出**，不得读成阻止能力。

### 1.2 两条使用闸门的行被明确标注

`AT-F8 / AT-ONCE / AT-IDENT / AT-PARTIAL / AT-POS` 需要一个**允许**的门判定才能进入边界，而真实门在 D8 未闭合时对一切计划都拒绝（`AT-POS-REAL` 已证）。这些行因此使用 `control_gate_allow`：

- 矩阵中逐行记录 `gate_used='control_gate_allow (test double; not a certification)'`；
- `real_qualification_claimed=false`；
- **没有注入任何假品质检查**——替身是"调用方给出的允许布尔值"，不是伪造的品质证书（与 B3-AC 的 `AC-R10` 探针同一处理方式：只证明机制，不宣称资格）。

## 2. 三个原子性条件的机械证据

| 条件 | 判定行 | 证据形式 |
|---|---|---|
| **排他性** | `AT-F8` | 边界内"发布前状态版本复检"+ 零写入 + 对照组会提交 → 检出并拒绝；**残余局限**：不能排除发布**进行中**的并发写入（需 AT2 拥有写入能力后的互斥） |
| **一次性** | `AT-ONCE` | 票据在**共享注册表**中按 id 记名消费；**跨事务对象**重放被拒 → 保护不在事务对象内部。范围**仅进程内** |
| **失败一致性** | `AT-PARTIAL` | 原子发布前候选完整构造+校验 → 失败尝试内写入原语调用 = 0；非原子发布中途异常 → 报告 `PARTIAL_COMMIT` + 真实字段，且 `rollback_claimed=false` |

这与 B3-AC 的 `AC-R7/AC-R8/AC-R9`（三条均**未达成**）构成前后对照：AT1 用**新的隔离事务**把三条各自做成了可复现的通过证据，同时**没有**改动 `_commit` 的历史语义、也没有让事务接管生产。

## 3. 夹具准入六条的机械化结果

`AT-F8 / AT-ONCE / AT-IDENT` 三行的 `qualification.all_ok = true`，六条逐项落库：

- **Q1 非空性**：每行记录 12+ 个 `(customer, vehicle_id)` 待提交对（夹具共 25 个）。
- **Q2 前置状态**：行声明谓词，`candidates_examined` 记录全部 8 个候选及其 `n_pending`，`silent_filtering=false`。
- **Q3 计划身份**：记录 `commit_identity` 与 legacy `plan_hash`，并显式声明"incumbent/candidate/certified 在本批是同一对象"、以及"重准备克隆 == 夹具"（`reprepared_plans_match_the_fixture=true`）。
- **Q4 合同真实性**：行标注 `budget_role`；`AT-D8` 行 `ISOLATION_ONLY` + `does_not_establish_install_feasibility=true`；其余为 `FORMAL`（B=300）。
- **Q5 阻断归因**：断言指定理由且要求"指定检查项是唯一失败原因"。
- **Q6 实际状态**：`evidence_source='live_vehicle_fields'`，逐车记录 `status/current_node/ready_time/committed_next/mutable_suffix`。

另外，`AT-D8` 记录了**正式预算下的观察**：同一夹具在 `B=300` 时**组合认证器自身**先失败（`formal_certification_reasons=['energy_budget_exceeded']`），真实门以 `certification_incomplete` 拒绝 → 正式预算下根本轮不到 D8 问题；`ISOLATION_ONLY` 只用于隔离该机制，**不是**安装可行性证据。

## 4. 实现自查发现并修掉的 6 个缺陷（全部会制造假结果）

1. **状态哈希函数不一致**（`live_state_fingerprint` vs `snapshot_state_hash`）→ 所有行报出虚假 `state_unchanged=False`；已统一为 `state_version()`。
2. **拒绝路径不计算后状态** → `state_unchanged` 退化为 `None == hash → False`，看起来像"拒绝也改了状态"；已统一经 `_finalize()`。
3. **夹具用 pre-prepare 克隆**（`prepare_decision_point` 抬高 `ready_time`）→ 被误报为 `state_changed_inside_boundary`；已改为 post-prepare 并断言与夹具一致。
4. **票据确定性 + 模块级默认注册表** → 行间互相干扰；已让各行用私有注册表，**唯独 AT-ONCE 故意共享**。
5. **`AT-F10` 静态口径** → 事务模块本身就是授权写入者，必须**排除自身**再声明"D3 其余模块零写入原语"。
6. **"无部分发布"的证据方式** → 拿 pre-prepare 哈希比 post-prepare 哈希无意义；改为**直接证据**：失败尝试内写入原语调用数 = 0。

## 5. 只读与禁用项

| 禁项 | 状态 | 证据 |
|---|---|---|
| B3-AT2 / 生产唯一写入者改造 | 未做 | `registered_on_env_anywhere=false`；`non_test_modules_importing_the_transaction=[]` |
| 注册为 env 真实提交入口 | 未做 | `isolation_report()` 机械派生 |
| 真实安装写回 | 未做 | `production_state_written=false`；每行只写脱离副本；`AUDIT=OK` |
| 修改 `_commit` 历史语义 | 未做 | `strict_online_env.py` sha256 `fd17fe77…`（已 pin） |
| 解除 D8 | 未做 | `AT-D8`/`AT-POS-REAL` 均 `BLOCKED_BY_D8`；`no_relaxation` 字段 |
| 七条历史路径统一重写 | 未做 | `historical_bypass_sites_modified_this_batch=[]` |
| G1/G2 或 F−D 实验 | 未做 | 本批无任何效用/收益数值 |

## 6. 本批未得出（不得声称）

- **P2 未建立**：D3 模块不引用历史入口 ≠ 所有真实提交都经过门；事务没有接管生产。
- **禁止即阻止**：`AT-F10` 是检出。
- **无崩溃恢复原子性**：本批不提供、也不声称；`PARTIAL_COMMIT` 从不称为 rollback。
- **无安装资格**：`AT-POS` 是机制控制；`AT-POS-REAL` 在真实条件下被阻断。
- G-L / G-C / P2 均 `NOT_ESTABLISHED`；`stable_online_gain=False`；I3 正式负结果、旧 D2 `not_qualified`、G1/G2 未启动、MaskCO 可归因 F−D 增益未建立 —— 全部不变。

## 7. AT1 停止条件的满足情况

用户设定的停止条件："全部预声明反例给出确定判定；事务的排他性、一次性、失败一致性有机械证据；并证明它未接管真实生产路径。失败项按失败记录，不靠重新运行或放宽 D8 翻盘。"

- 全部 13 行确定判定 ✓（且 6/6 实例一致，未为整齐结果扩大窗口或替换条件）
- 三项有机械证据 ✓（§2）
- 未接管真实生产路径 ✓（`ISOLATION=OK`）
- 失败/未达成项如实记录：**认证能力未完善（D8）**、**AT2 未授权**、**边界不能排除发布中的并发写入**、**票据注册表仅进程内** —— 均按原样登记，未经重跑或放宽 D8 翻盘 ✓
