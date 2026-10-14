# D3-G0 Evidence B3-AT2A — 结果报告（唯一写入者与互斥机制的受限落地）

- 阶段：`D3_G0_Evidence_B3AT2A`（受限；`B3-AT2B` 未授权）
- 预登记：`B3AT2A_COUNTEREXAMPLE_SPEC.json`，sha256 `0305471b5745d5ce25caf2518afed44bc385fe3d63f10f8394d3a45c192407e6`
- 设计：`B3AT2A_DESIGN.md`；矩阵：`B3AT2A_COORDINATOR_MATRIX.json`
- 原始输出：`raw/B3AT2A_stdout.txt`、`raw/B3AT2A_repeat{1,2,3}.txt`、`raw/B3AT2A_instance{0..5}.txt`、`raw/B3AT2A_COORDINATOR_MATRIX_instance{0..5}.json`
- 命令：`python -B scripts/tests/test_d3_b3at2a_coordinator.py --instances 0 --heavy-cap 8`

## 0. 三项分别验收的结论（用户硬条件 3）

| 验收项 | 结论 | 依据行 |
|---|---|---|
| **A 写入能力排他** | **未达成（部分达成）** | `AT2-F8` ✓、`AT2-RACE` ✓、`AT2-F10` = `REFUSED_BUT_ONE_WRITE_PATH_ESCAPED` |
| **B 票据一次性** | **达成**（仅进程内） | `AT2-ONCE` = `AT_MOST_ONE_CONSUMED` |
| **C 发布失败一致性** | **达成** | `AT2-PARTIAL` = `PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS` |

三项在矩阵 `acceptances` 中各自独立成字段；`AT2-ONCE` / `AT2-PARTIAL` 的 detail 显式写明不得用于推断另外两项。**三项结论在实例 0–5 上完全一致；实例 0 连跑 3 次（含真实线程行）判定完全相同**。

## 1. 逐行判定（13 行全部到达预登记判定）

| id | 判定 | 关键证据 |
|---|---|---|
| `PRE` | `SETUP_OK` | 8 个捕获决策点只读；谓词与**每个**候选的通过/失败都记录 |
| `AT2-F8` | `REFUSED_NO_MUTATION` | 发布开始前漂移 → `refusals=['state_changed_before_publish']`、`epoch_at_publish_start != epoch_at_authorisation`、`publish_state=null`（**未发布**）；漂移机制是**直接 `mutable_suffix` 赋值**（守卫不覆盖，由 epoch 复检捕获） |
| `AT2-RACE`（**主反例**） | `RACE_PREVENTED_OR_SERIALIZED` | (a) 发布期间流氓线程直调 `_commit` → `UnauthorizedWrite`，协调器自身 `COMMITTED_VERIFIED`；(b) 两个协调器同一票据 → 一个提交、一个 `ticket_already_consumed`；**`max_concurrent_publish_windows=1`**（真实线程 + Event/Barrier） |
| `AT2-F10` | `REFUSED_BUT_ONE_WRITE_PATH_ESCAPED` | `direct_commit` 与 `plan_and_commit` 均被强制拒绝；**AT1 事务仍以逐车属性赋值触达状态**（未被守卫拦截）→ 如实记为部分拒绝 |
| `AT2-ONCE` | `AT_MOST_ONE_CONSUMED` | 两个真实线程同一票据：恰一个消费、恰一个 `ticket_already_consumed` |
| `AT2-PARTIAL` | `PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS` | 第 2 辆车异常 → `PARTIAL_COMMIT`、`n_published_before_failure=1`、`rollback_claimed=false`、`committed_verified=false`、**未做重读**（不假装成功） |
| `AT2-POST` | `COMMITTED_VERIFIED` | 正向：发布后独立重读与授权结果逐车一致 → `publish_state_trace` 结尾为 `COMMITTED_VERIFIED`；**负向控制**：发布后篡改再重读 → 降级 `OUTCOME_UNKNOWN`（verified 不是自动的） |
| `AT2-D8` | `BLOCKED_BY_D8` | 三类原因分开报告：`quality_salability=true`、`energy_budget=false`、`transaction_authorization=true`；`ISOLATION_ONLY` 行标注不构成安装可行性证据 |
| `AT2-WAIT` | `WAIT_BY_ENV_PROTOCOL` | 未调用门、不计入拒绝能力 |
| `AT2-POS` | `COMMITTED_VERIFIED` | 无并发、状态稳定、机制专用门、合法非空计划 → 隔离提交完成并验证；`counts_as_install_qualification=false` |
| `ISOLATION` | `OK` | 零非测试导入者、`registered_on_env=False`、守卫未装到生产实例、历史旁路修改 0、排除两个授权写入者后 D3 内写入原语调用为 0 |
| `PREREG` / `AUDIT` | `OK` | 9 条反例全覆盖；无 hook 新 env 复现同一生产轨迹哈希 |

## 2. 并发窗口（主反例）的证据形态

用户裁定："锁只有在所有相关写入者都遵守时才有效"，因此本批不只加锁，还在**实例**上施加写入约束：

- 流氓写入者（真实线程）在协调器发布期间直调 `_commit` → 被守卫拒绝，证据记录含 `capability_present=false`、`critical_section_owner`、`caller_thread`；
- 两个协调器并发 + 同一票据 → 串行化并只有一个消费；
- 交错度量：`max_concurrent_publish_windows = 1`（若出现 >1 即为交错，判定会失败）。

## 3. 发布状态词汇表与 `atomic` 降级（用户裁定）

- 词汇表落地：`STAGED_VALIDATED / PUBLISHING / COMMITTED / COMMITTED_VERIFIED / PARTIAL_COMMIT / OUTCOME_UNKNOWN`，每次发布都有 `publish_state_trace`。
- `atomic` 已更名为 `publish='staged_then_single_pass'`（旧名仅作弃用别名），并把"未建立多车辆发布不可观测性"写进 `PUBLISH_MODE_NOTE`。
- **AT1 同步更正**：AT1 模块现在显式 `committed_verified=false`（无重读），其设计/报告已追加 Erratum（§7 / §8），原文不改写。AT1 复跑后 13/13 判定不变。

## 4. 本批的失败项与残余缺口（按失败记录，不翻盘）

1. **`AT2-F10` 部分拒绝**：AT1 事务通过 `mutable_suffix` **属性赋值**触达状态，实例守卫只覆盖 `_commit` / `_plan_and_commit`。按用户标准（三种写入方式不得同时修改同一 D3 状态），**"唯一写入者"因此未达成**。关闭它需要拥有状态对象或 `VehicleState` 级守卫 → **AT2B 范围**。
2. **正因上述缺口，`AT2-F8` 的漂移检测靠 epoch 复检而非阻止**：能拒绝、能保证不发布，但不能阻止那次越权赋值本身。
3. **票据注册表仅进程内**：无持久化、无跨进程排除，故 B 项达成范围有限。
4. **无崩溃恢复语义**：不声称持久事务；`PARTIAL_COMMIT` 明确不称 rollback。
5. **D8 仍阻断**：本批未放宽、未注入假品质检查；`AT2-POS` 使用机制专用门且不计作安装资格。

## 5. 只读与禁用项

| 禁项 | 状态 | 证据 |
|---|---|---|
| B3-AT2B / 注册为正式唯一提交入口 | 未做 | `coordinator_registered_on_env=false`；`ISOLATION=OK` |
| 修改历史七条独立基线 | 未做 | `historical_bypass_sites_modified_this_batch=[]`（AST 派生） |
| 解除 D8 | 未做 | `AT2-D8` 阻断；`d8_relaxed=false` |
| 真实批次写回 | 未做 | `production_state_written=false`；`AUDIT=OK` |
| 修改 `_commit` 历史语义 | 未做 | 守卫是实例级包裹并 `restore()` 还原；`strict_online_env.py` sha256 `fd17fe77…` 未变 |
| G1/G2 或 F−D 实验 | 未做 | 无任何效用/收益数值 |

## 6. 本批未得出（不得声称）

- **P2 未建立**；G-L/G-C 未建立；D3 G0 未通过；G1/G2 未启动。
- **"唯一写入者"未达成**（见 §4.1）；协调器只在隔离副本 + 受守卫实例内成立。
- 不声称原子性、不声称崩溃恢复、不声称跨进程排除、不声称安装资格。
- I3 正式负结果、旧 D2 `not_qualified`、`stable_online_gain=False` 均不变；B3-AC 实例 3 的 `NOT_VERIFIABLE` 保持原判。

## 7. AT2A 停止条件的满足情况

用户设定的 AT2A 目标与硬条件："全部预声明反例给出确定判定；写入能力排他/票据一次性/发布失败一致性分别有机械证据；并证明未接入正式运行。失败项按失败记录，不靠重新运行或放宽 D8 翻盘。"

- 全部 13 行确定判定 ✓（实例 0–5 一致、实例 0 三连跑一致）
- 三项分别验收 ✓，其中 **A 项如实记为未达成**（部分拒绝 + 残余缺口），未以 B/C 的通过推断 A
- 未接入正式运行 ✓（`ISOLATION=OK`、`AUDIT=OK`、`strict_online_env.py` 未变）
- 正式预算是原合同预算、隔离预算已标注 `ISOLATION_ONLY`，能源/品质/授权三类原因分开报告 ✓
