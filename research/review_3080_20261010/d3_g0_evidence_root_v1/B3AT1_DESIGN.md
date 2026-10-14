# D3-G0 Evidence B3-AT1 — 授权提交事务的隔离实现（设计）

- 阶段：`D3_G0_Evidence_B3AT1`（用户授权的**受限 B3-AT1**；`B3-AT2` **未授权**）
- 预登记（先于实现冻结）：`B3AT1_COUNTEREXAMPLE_SPEC.json`，sha256 `01803fd5428a83a8b262b49c6dfa4710b60c2f15e8c8d8bc52283b19bfe476a4`
- 基础提交：`100018e139ad6978010f866f0751a840df4d0cd7`
- 产物：`scripts/simulation/d3_commit_transaction.py`（事务）、`scripts/tests/test_d3_b3at1_transaction.py`（验收）→ `B3AT1_TRANSACTION_MATRIX.json`

## 0. 三项必须分别验收（用户裁定）

| 验收项 | 本批结论 |
|---|---|
| **AT1 事务成功**（隔离副本上的排他性/一次性/失败一致性） | **有机械证据**（见 §3），但仅在隔离副本上 |
| **认证能力完善**（D8 `quality_salability`） | **未达成**：`CAPABILITY_GAP_BLOCKING` 依旧，本批未放宽、未注入假品质检查宣称资格 |
| **AT2 生产写入资格（P2）** | **未授权、未建立** |

## 1. 四个设计要点的落地方式

### R1 单一授权写入接口（门判定 / 事务写入分离）

`AuthorizedCommitTransaction` 是范围内**唯一的授权写入接口**：`commit()` 先调用调用方传入的 `gate_decision_fn`，只有它允许才进入写入。`default_gate_decision_fn` 只把 `compose_certification(...)['installable']` 作为 `certification_complete` 传给 `pre_install_reverify` —— **判定权在门，写入权在事务**。

**边界声明（不得含糊）**：一个调用方完全可以自己先过门、再直接调 `env._commit`；本模块**不阻止**那件事，因为阻止它需要事务**拥有写入能力**，那属于 B3-AT2。本批给出的替代证据是**运行期写入监测 + D3 范围内的机械不可达**（§2）。

### R2 凭据一次性（绑定决策点 + 状态版本 + 计划身份）

- 提交编号 `commit_identity(plans)` 对 **(vid, anchor_node, anchor_time, anchor_load, suffix)** 取 32-hex —— 明确包含 `anchor_time` / `anchor_load`，这是 `action_contract.plan_hash`（16-hex，只含 vid/anchor_node/suffix）**盲**的两项。
- 授权票据 `ticket_id = H(inst|event|clock|state_hash|commit_id|certifier)[:16]`。
- 消费记录在 `CommitTicketRegistry` 中按 `ticket_id` 记名：**新建一个事务对象不能刷新授权**（跨对象重放同样被拒）。`scope = 'in_process_module_singleton'` —— **仅进程内**，不声称跨进程或持久化。
- 消费发生在**边界内**；拒绝路径**不消费**（授权未被使用），唯 `PARTIAL_COMMIT` 记录为 `consumed_on_partial_commit`，以免部分发布被静默重试。

### R3 复验与写入不可分离（受控临界区）

`_Boundary` 覆盖：**复验 → 消费票据 → 构造候选 → 校验候选 → 发布**。边界内设置了一个**发布前状态版本复检点**：进入边界时取版本，`pre_publish_hook`（测试注入的"另一个写入者"）之后、真正发布之前再取一次版本，两者与授权锚定的版本不一致即拒绝。

这就是"只增加一次哈希比较不足以解决 F8"的落地：**比较点必须与写入在同一临界区，且必须在写入之前**。

**残余局限（如实登记）**：边界能**检出**状态移动，但**不能排除**在发布动作**进行中**并发写入的作者。真正的排他性需要 AT2 拥有写入能力后的互斥实现。

### R4 失败一致性

- 默认 `publish='atomic'`：候选先在一份独立 staging 结构上**完整构造并校验**（越界/已服务/重复），通过后发布只是纯属性赋值 + 调用环境自身的 `_commit` 头部提交，**不存在"写到第二辆车才发现异常"的失败面**。
- `publish='non_atomic'`（显式提供 `publish_fn`）会逐车写入；中途抛异常 → 结果记为 **`PARTIAL_COMMIT`**，带真实的逐车字段证据、`n_published_before_failure`、`rollback_claimed=false`，并在理由里写明"**不得称为 rollback**"（无持久事务机制、无状态级恢复证明）。
- 另设 `OUTCOME_UNKNOWN` 结果常量，供"无法判定是否发布"的情形使用；本批未触发，登记为可用状态而非已用状态。

## 2. 可达调用图、运行期写入监测与隔离声明

- **可达调用图（机械）**：`isolation_report()` 用 `ast` 扫描，得到 `non_test_modules_importing_the_transaction = []`（只有测试文件导入），`registered_on_env_anywhere = False`。
- **运行期写入监测**：`WriteMonitor` 在**一个 env 实例**上包裹 `_commit` / `_plan_and_commit`，记录 `(caller file:line, vehicle, node, inside_transaction)`，退出即还原。它补的正是 AST 做不到的部分：动态属性调用与外部注入。
- **D3 范围机械不可达**：除事务模块自身外，`d3_*.py` 中**没有任何** `_commit` / `_plan_and_commit` 调用，也**没有任何** `mutable_suffix` / `committed_next` 赋值（AST 派生，注释/文档字符串不计）。
- **与历史入口的隔离声明**：七个历史旁路站点本批**一个都没有修改、也没有迁移**（`historical_bypass_sites_modified_this_batch = []`）。"D3 模块不引用历史入口"**不等于** P2；反之，"历史调用点未被改动"也不等于"新事务存在旁路"。

## 3. 预登记反例与判定（详见 `B3AT1_REPORT.md`）

9 条反例 + 3 条控制行 + 隔离声明，全部到达预登记判定。**两条使用闸门的行被明确标注**：`control_gate_allow` 是**标签化的测试替身**，唯一目的是打开事务边界；使用它的行一律记 `real_qualification_claimed=false`，且**不注入任何假品质检查**（替身是"调用方给出的允许布尔值"，不是伪造的品质证书）。

## 4. 夹具准入门槛已机械化为六条检查（用户裁定）

| 条件 | 实现 |
|---|---|
| Q1 非空性 | 每个非 WAIT 行必须至少有一个真实可定位的待提交客户，并记录 `(customer, vehicle_id)` 对 |
| Q2 前置状态 | 行声明谓词；**每个**捕获候选都记录通过/失败，不允许静默筛除；无候选满足即 `NOT_VERIFIABLE` |
| Q3 计划身份 | 记录 candidate/incumbent/certified/实际提交四者身份，**并显式声明它们何时是同一对象** |
| Q4 合同真实性 | 行标注 `budget_role ∈ {FORMAL, ISOLATION_ONLY}`；`ISOLATION_ONLY` 行必须带 `does_not_establish_install_feasibility=true` |
| Q5 阻断归因 | 断言指定理由，并要求**指定检查项是唯一失败原因**；否则 `WRONG_ATTRIBUTION` |
| Q6 实际状态 | 每行 `evidence_source='live_vehicle_fields'` 并记录真实字段前后值；**不得**以回调返回值代替 |

`AC-R3` 的大预算在本批对应 `AT-D8` 的 `ISOLATION_ONLY` 行：用户已认可其机制测试资格，但**不属于正式预算下的安装可行性证据**；同时记录该夹具在 **FORMAL 预算下由组合认证器自身**（能耗理由）先失败，因此真实门下根本轮不到 D8 问题。

## 5. 非声称

- 不声称 P2；不声称生产路径已被接管；不声称真实安装资格。
- 不声称崩溃恢复意义上的原子性；`PARTIAL_COMMIT` 从不被称作 rollback。
- 不声称检出=阻止（`AT-F10` 明确 `detected=true, prevented=false`）。
- 不声称 D8 已闭合；不把 `AT-POS` 的机制控制读成资格成立。
- 七个历史旁路未修改、未迁移；`_commit` 历史语义未改。
- G-L/G-C/P2 均 `NOT_ESTABLISHED`，`stable_online_gain=False`，I3 与旧 D2 `not_qualified` 不变，实例 3 的 `NOT_VERIFIABLE` 保持原判。

## 6. 本批自查发现并修掉的实现缺陷（记录在案）

1. **状态哈希函数不一致**：事务用 `live_state_fingerprint`（返回快照自带 `state_hash`），凭据锚点用 `snapshot_state_hash`。两者不可互换 → 所有行都报出**虚假的 `state_unchanged=False`**。已统一为 `state_version()`。
2. **拒绝路径不计算后状态**：早退分支在设置 `state_hash_after` 之前返回，`state_unchanged` 退化成 `None == hash → False`，看起来像"拒绝也改了状态"。已统一经 `_finalize()` 在**每条**返回路径上填写。
3. **夹具用了 pre-prepare 克隆**：`prepare_decision_point` 会抬高 `ready_time`，因此 pre-prepare 状态与 `post_hash` 本就不同；第一版把这件事误报成 `state_changed_inside_boundary`。已改为在 post-prepare 克隆上作业，并把"重准备克隆 == 夹具"作为断言。
4. **票据确定性造成的行间干扰**：`ticket_id` 对同一（决策点, 状态, 计划）是确定的，默认注册表是模块级单例，导致后续行被前一行的消费结果拒绝。已让各行使用私有注册表，**唯独 AT-ONCE 故意共享一个**以测试跨对象重放。
5. **AT-F10 静态声明口径**：事务模块本就是授权写入者，因此"D3 模块不引用写入原语"必须**排除事务模块自身**，并把它单独列出。
6. **"无部分发布"的证据方式**：拿 pre-prepare 克隆的哈希去比夹具的 post-prepare 哈希是无意义的比较；已改为**直接证据**：失败的那次尝试内 `n_write_primitives_inside_the_failed_attempt == 0`。
