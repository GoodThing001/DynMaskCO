# D3-G0 Evidence B3-AC — 设计（只读路径分类 + 隔离副本上的提交拒绝能力验收）

- 阶段：`D3_G0_Evidence_B3AC`（用户 2026-10-14 授权的**阶段 A**）
- 预登记（先于实现冻结）：`B3AC_COUNTEREXAMPLE_SPEC.json`，sha256 `217ca641691c2ada22e85d82388fdeedab3ffd6ed5f32879493721591c0b2b33`
- 基础提交：`9c3a2816c4805bf7428a993c7c8a466b1c414ceb`
- 两个产物：
  - `scripts/tests/test_d3_b3ac_path_classification.py` → `B3AC_PATH_CLASSIFICATION.json`
  - `scripts/tests/test_d3_b3ac_gate_refusal.py` → `B3AC_REFUSAL_MATRIX.json`

## 0. 本批把两个命题强行分开

用户裁定把原先混在一起的问题拆成两个**独立验收命题**，本批只做 P1：

| 命题 | 内容 | 归属 |
|---|---|---|
| **P1** | **门能拒绝非法动作**（在隔离副本上可验证） | **B3-AC（本批）** |
| **P2** | **所有实际提交都必须经过这个门** | 独立授权的 **B3-AT**（本批不做、不声称） |

因此本批的任何结果**不得**被读成 F10 已在生产路径关闭，也不得被读成 G-L 或 G-C 通过。

## 1. 授权范围与禁用项（逐字）

授权："阶段 A：B3-AC（建议现在批准）只读路径分类 + 隔离副本上的 A-c6 拒绝能力验收。先预注册合法性、状态漂移、认证失败、WAIT 区分和不完整质量认证的反例。允许使用隔离副本验证新门的行为，但不接管真实生产路径。"

本批不做：生产环境唯一写入者改造、七条历史路径的统一重写、修改 `_commit` 的历史语义、解除 D8、真实安装、G1/G2 或 F−D 科学实验。

## 2. 第一部分：只读路径分类（可机械复核）

### 2.1 为什么要机械复核

用户的原话是判定标准：

> "七处直接调用认证原语" 不等于 "七处能够绕过 D3 提交门的状态写入者"

所以分类不能是一段散文意见。`test_d3_b3ac_path_classification.py` 用 `ast` 从源码推导，并把每条判定的**证据锚点（文件:行 + 该行源码）**写进矩阵：锚点在每次运行时重新读取，代码漂移即失败，而不是让一条记录悄悄过期。

派生量（全部来自 AST，不是人工断言）：

| 派生量 | 机械定义 |
|---|---|
| `class_or_ancestor_assigns_write_attrs` | 该类**或解析到的祖先类**存在对 `mutable_suffix` / `committed_next` 的赋值；祖先解析在本次扫描集合内按类名递归（带环保护），并记录每一跳 |
| `is_env_replanner_entry_point` | 该类或祖先定义了 `plan` 或 `on_reveal`（这是 `StrictOnlineEnv` 唯一会把 **live** `vehicles` 交出去的两个入口） |
| `result_branched_on_as_filter` | 认证返回值是否被 `if` 分支使用（即是否至少是一个候选筛选器）；不支持的形式判为 `not branched`（**更弱**的声明，方向安全） |
| `referenced_by_runner_as_replanner` | 跨文件：哪些文件既向 `StrictOnlineEnv(replanner=...)` 接线、又提到该类名 |
| `referenced_by_any_d3_module` | 哪些 `d3_*`（或导入 `d3_` 的）模块提到该类名 |

**第一次实现的自查缺陷（已修，记录在案）**：只读站点自身类体时，7 个站点里有 6 个被判 `write=False`——因为写入发生在**继承来的** `C0ReserveReplanner.plan`。那是"用户警告方向的反面"的假阴性（把有能力写入的类说成没有）。修复后改为继承解析 + 证据链；同理，`referenced_by_runner_as_replanner` 第一版只在同文件内查找，对库式 replanner（rolling / rrnco / solver）报 `used_by=0`，也一并修成跨文件。

### 2.2 四类规则与优先级（预登记，逐字实现）

- **R1 `D3_REAL_COMMIT_ENTRY`**：调用链经 D3 门原语（`compose_certification` / `pre_install_reverify` / `run_repair_transaction`）且能到达真实写入。
- **R2 `CANDIDATE_EVAL_ONLY_NO_REAL_WRITE`**：认证结果只用于筛选候选，且**不可达**任何真实写入（含继承）。
- **R3 `HISTORICAL_BASELINE_INDEPENDENT_ENTRY`**：站点所在脚本自建 env/replanner 作为独立运行入口，且不导入任何 `d3_` 门原语。
- **R4 `INDIRECT_WRITER_CAN_WRITE_D3_STATE`**：站点所在类对 env 传入的 live `vehicles` 赋值 `mutable_suffix` / `committed_next`（即可导致一次**无门提交**）。
- **优先级 R1 > R4 > R3 > R2**；同时满足的标签全部记录，治理结论取最高优先级。

### 2.3 分类结果（实例化的机械输出，不在此处复述细节）

矩阵 `B3AC_PATH_CLASSIFICATION.json` 给出 7 个预登记站点的 `labels_all_satisfied` / `governing_category` / 每条证据。**关键结论必须成对陈述，不得合并**：

1. `D3_REAL_COMMIT_ENTRY = 0`，且 **D3 门原语的非测试调用点 = 0** —— D3 授权提交入口**尚不存在**（这正是 B3-AT 要建的东西）。
2. 7 个站点全部 `capable_of_writing_live_state = true`（继承来的 `plan()` 写 live 状态），但 `referenced_by_any_d3_module = 0`。
   - "**有能力**写入 live 状态" ≠ "**已接线**到 D3 生产路径"。两者必须分开报，永远不得合并成一句"D3 生产路径已被写入"。

## 3. 第二部分：隔离副本上的拒绝能力验收

### 3.1 受控提交路径只存在于 harness 内（H6）

`test_d3_b3ac_gate_refusal.py` 定义了一个函数 `controlled_commit(...)`，顺序固定为：

```
WAIT 检测 → compose_certification → pre_install_reverify → （全通过才）写入隔离副本
```

- 它**从不**注册到 env、从不被任何 `d3_` 模块导入、不修改 `_commit` / `_plan_and_commit`；
- 写入只发生在 `d3_state_assembly.clone_state(snapshot)` 产生的脱离副本上；
- 写入时使用的是**环境自己的原语**（先按计划设置副本的 `mutable_suffix`，再调 `env._plan_and_commit`），不另造一套写入语义。

### 3.2 五条必须给出的反事实比较（用户逐条指定）

| 用户要求 | 行 | 判定词 |
|---|---|---|
| 旧路径：非法后缀能够写入 | `AC-R1` | `COUNTEREXAMPLE_REPRODUCED` |
| 受控新路径：相同非法后缀被拒绝、真实状态不变 | `AC-R2` | `REFUSE_NO_MUTATION` |
| 合法 WAIT：无待提交动作，不得计入拒绝能力 | `AC-R5` | `WAIT_BY_ENV_PROTOCOL`（`counts_as_refusal_capability=false`） |
| 状态漂移：凭据过期，即使计划未变也必须拒绝 | `AC-R6`（+ 对照行） | `REFUSE_NO_MUTATION` |
| D8：品质认证缺失，不得把投影升级为真实安装许可 | `AC-R3` | `REFUSE_NO_MUTATION`（品质维度） |

### 3.3 诚实规则（预登记 H1–H6）

- **H1** 复现缺陷 → `COUNTEREXAMPLE_REPRODUCED`，绝不写 PASS。
- **H2** 拒绝必须**按理由归因**：断言 `reasons` 集合里出现预期理由；理由不对 → `REFUSAL_WRONG_REASON`，**不计入**拒绝能力。
- **H3** `WAIT_BY_ENV_PROTOCOL` **不得**计入拒绝能力（它不是被拒绝的写入，而是协议正常输出）；这正是 B3-CG 附录 A-3 要求拆开的 D0 歧义。
- **H4** D8 拒绝只证明**品质维度**上的拒绝能力，不是 installability，不是 G-L。
- **H5** 无持久事务机制时**不得**宣称崩溃恢复意义上的原子性。
- **H6** 受控提交路径只允许存在于 harness 内、只写隔离副本。

## 4. 原子性：本批只给定义，不给"已达成"结论

用户要求给"原子性"一个更严格的定义。本批把它冻结为**三个必须同时成立的条件**，并明确现状：

| # | 条件 | 定义 | 本批实测状态 |
|---|---|---|---|
| **A1 排他性** | 其他写入路径不能在身份复验与第一次状态写入之间改变相关状态 | 凭据层会拒绝漂移（`AC-R6`），但**写入层**（`InstallLedger.finish`）只比较 `plan_hash`、从不咨询凭据，而 `plan_hash` **盲于 `anchor_time` / `anchor_load`** → 漂移下仍安装（`AC-R7`） | **未达成**（`AC-R7 = COUNTEREXAMPLE_REPRODUCED`） |
| **A2 一次性** | 同一认证凭据不得被重复用于不同提交 | `pre_install_reverify` 是其参数的**纯函数**，无 nonce / 计数器 / 序列 → 同一凭据可授权任意多次提交；`InstallLedger.finish` 的 `already_finished` 守卫只在**同一个 ledger 对象**内生效 | **未达成**（`AC-R8`） |
| **A3 失败一致性** | 写入过程中发生异常时，不得把部分变更报告为完整提交；没有状态级恢复能力时必须显式标记未知或部分提交 | `d3_geo_adapter.py:689` 的 `self.install_fn(plan)` 没有任何守卫，`finished` 在它返回**之后**才置位 → 异常逃逸、`n_installs==0`、无事件、`finished==False`，且部分变更留在副本上（`AC-R9`） | **未达成**（`AC-R9 = FAILURE_ATOMICITY_ABSENT`） |

**并且（H5）**：在这些条件未达成、且没有任何持久事务机制的情况下，**不得**把任何结果描述为"崩溃恢复意义上的原子性"。本批不提供该性质，也不声称提供。

三个条件全部属于 **B3-AT** 的实现范围（唯一授权提交事务 + 受控原子写入 + 生产调用点接入），本批只负责测量它们当前是否成立——答案是**三条都不成立**，且每条都有可复现的测量行。

## 5. 非声称（本批明确不说什么）

- 不说 F10 已在生产路径关闭（`D3_REAL_COMMIT_ENTRY = 0`，门还没有生产调用点）。
- 不说 G-L / G-C 通过（`NOT_ESTABLISHED`）。
- 不说 installability 达成：`AC-R3` 的拒绝恰恰表明**没有**品质认证时 `installable` 必须为 `False`；`AC-R10` 的 `installable=True` 仅在**注入假品质检查**的探针下出现，且显式记录 `not_a_production_path=True`、`install_permission_claimed=False`。
- 不把 `WAIT_BY_ENV_PROTOCOL` 计入拒绝能力。
- 不修改历史源码、不改 `_commit` 语义、不解除 D8、不做真实安装、不启动 G1/G2 或任何 F−D 科学实验。
- 研究的最终状态不变：G-L/G-C 未建立、D3 G0 未通过、G1/G2 未启动、MaskCO 可归因学习增益未建立、I3 正式负结果与旧 D2 `not_qualified` 不变、`stable_online_gain = False`。
