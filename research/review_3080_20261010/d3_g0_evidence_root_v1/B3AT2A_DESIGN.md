# D3-G0 Evidence B3-AT2A — D3 提交协调器（唯一写入者与互斥）设计

- 阶段：`D3_G0_Evidence_B3AT2A`（受限架构阶段；`B3-AT2B` 未授权）
- 预登记（先于实现冻结）：`B3AT2A_COUNTEREXAMPLE_SPEC.json`，sha256 `0305471b5745d5ce25caf2518afed44bc385fe3d63f10f8394d3a45c192407e6`
- 基础提交：`c3320ed69a011b6c451626d1d254e28d1d8ed32a`
- 产物：`scripts/simulation/d3_commit_coordinator.py`、`scripts/tests/test_d3_b3at2a_coordinator.py` → `B3AT2A_COORDINATOR_MATRIX.json`

## 0. 三个硬条件如何落地

| 用户硬条件 | 落地 |
|---|---|
| 先预注册、后实现，且把**并发发布窗口**作为独立主反例 | 预登记先冻结（sha 见上）；`AT2-RACE` 为独立主反例，用 **真实线程 + Event/Barrier** 驱动，并记录 `max_concurrent_publish_windows`（=1 才算通过），不以顺序调用冒充并发 |
| 只在隔离的真实环境副本上验证写入，不提前接入正式运行 | 全部写入发生在 `clone_state(snapshot)` 衍生副本；守卫只装在探针 env 实例上；`ISOLATION=OK`（零非测试导入者、`registered_on_env=False`、历史旁路修改 0）；`AUDIT` 用无 hook 新 env 复现同一生产轨迹哈希 |
| 写入能力排他 / 票据一次性 / 发布失败一致性**分别验收** | 矩阵 `acceptances` 三项各自独立结论；`AT2-ONCE` 与 `AT2-PARTIAL` 的 detail 里显式写入"不得用于推断另外两项" |

## 1. 架构：唯一写入者

- **写入能力令牌 `WriteCapability`**：协调器持有，身份比较、不可序列化、不交给其他模块。
- **实例级守卫 `D3WriteBoundary`**：在每个 D3 执行实例上包裹 `_commit` / `_plan_and_commit`；只有当**当前线程位于某个协调器的临界区内**且**开启该临界区的令牌仍在该边界上有效**时才放行，否则抛 `UnauthorizedWrite` 且**不执行任何写入**。这是**实例级强制阻止**（强于 AT1 的检出），且**不修改 `_commit` 源码语义**（`restore()` 还原）。
- **协调器 `D3CommitCoordinator`**：其他模块只能 `submit(候选计划, 授权凭据)`；协调器在临界区内完成复验、消费票据、构造并校验候选、发布。
- **三种写入方式不可共存**（用户要求）：实测结果是**部分达成**——旧直接提交入口与新事务的独立发布接口都被强制拒绝，但 AT1 事务仍能通过**逐车 `mutable_suffix` 赋值**触达状态，而属性赋值不在实例守卫覆盖范围内（见 §3 与报告 §4）。该缺口如实记为**失败项**，关闭它需要"拥有状态对象"或 `VehicleState` 级守卫，属 AT2B。

### 1.1 并发模型（含一次自查修复）

第一版守卫询问**单个协调器对象**"当前线程是否在它的临界区内"，导致第二个协调器在同一实例上的每次写入都被拒（`AT2-ONCE` 行返回异常而不是干净的票据拒绝）。现改为**进程级线程局部临界区标记**（`_CRITICAL`，携带开启临界区的令牌），因此多个协调器可以合法共享同一受守卫实例，同时每次写入仍绑定到**正在发布那个协调器的令牌**。

## 2. 发布状态词汇表（按用户裁定，AT1 术语已同步更正）

`STAGED_VALIDATED` → `PUBLISHING` → `COMMITTED` →（独立重读通过）`COMMITTED_VERIFIED`；异常/中止时为 `PARTIAL_COMMIT`，无法判定时为 `OUTCOME_UNKNOWN`。

- **`atomic` 只作为实现模式名**：已更名为 `publish='staged_then_single_pass'`（旧名保留为弃用别名），并写入 `PUBLISH_MODE_NOTE`：staging 失败不发布、已测试注入下无部分发布，但**不**声称多车辆发布不可被观察到中间状态、也**不**声称任意中途异常都不留部分状态。
- **`COMMITTED_VERIFIED` 只在独立重读后授予**：AT1 模块因此显式 `committed_verified=false`（它没有重读）；协调器承担重读（`AT2-POST`）。
- **负向控制**：发布后篡改状态再重读 → 降级为 `OUTCOME_UNKNOWN`，不得被祝福为 verified。

## 3. 三类原因分别报告（用户要求）

`classify_causes()` 对每个使用真实门的行分别给出：
`energy_budget` / `quality_salability` / `transaction_authorization`。

- `AT2-D8` 行：`ISOLATION_ONLY` 预算下 `quality_salability=true`、`energy_budget=false`、`transaction_authorization=true`（门拒绝）。
- 同时记录 **FORMAL 预算观察**：同一个夹具在 `B=300` 时组合认证器自身报 `energy_budget_exceeded` → **能源原因被单独报告，未被归因于品质**；正式预算**未**被放宽以让事务测试成功。

## 4. 夹具准入（继承六条 + 两条 AT2A 专用）

继承：Q1 非空性 / Q2 前置状态 / Q3 计划身份 / Q4 合同真实性（`ISOLATION_ONLY` 必带"不构成安装可行性证据"）/ Q5 阻断归因 / Q6 实际状态。

AT2A 新增：

- **Q7 并发可复现**：并发行必须由真实线程/事件实现，并记录交错证据（`max_concurrent_publish_windows`），不得用顺序调用模拟并发。
- **Q8 守卫作用域**：守卫只安装在隔离执行实例上，并记录它未安装到任何生产实例（`installed_on_a_production_instance=false`）。

## 5. 非声称

- 不声称 P2（协调器未接入正式运行）；不声称 G-L/G-C；不声称 D8 已解除。
- 不声称"唯一写入者"已完整达成：**AT2-F10 为 `REFUSED_BUT_ONE_WRITE_PATH_ESCAPED`**，属性写入路径未被阻止（只被 epoch 复检检出）。
- 不声称原子性：模式名已降级；`PARTIAL_COMMIT` 从不称为 rollback；`OUTCOME_UNKNOWN` 不称为成功。
- 票据注册表**仅进程内**；无持久事务、无跨进程排除。
- 正向控制不计作安装资格（`counts_as_install_qualification=False`）。

## 6. 本批自查发现并修掉的实现缺陷

1. **守卫绑定单个协调器对象** → 第二个协调器全部写入被拒（AT2-ONCE 行异常）。已改为线程局部临界区标记 + 令牌绑定。
2. **行间票据冲突**：`ticket_id` 对同一（决策点, 状态, 计划）确定，共享注册表使后续行被前一行消费结果拒绝。已支持每次 `submit` 传入独立注册表；`AT2-ONCE`/`AT2-RACE(b)` 故意共享以测跨对象重放。
3. **`_plan_and_commit` 与 `D3WriteBoundary` 语义冲突**：AT1 事务在部分夹具下没有可提交的 idle/ready 车辆，因此它的 `_commit` 从未被调用，事务"成功"却只做了属性赋值 → 第一版把 `AT2-F10` 误判为全量拒绝。现在按写入路径逐项记录，得出诚实的**部分拒绝**结论。
4. **`AT2-POST` 正向控制曾拿到 `post_read=None`**：根因是行间票据冲突（见 2），不是协调器缺陷。
5. **`PUB_OK()` 残留引用与 `qualification()` 实参顺序**：运行期即暴露并修正。
