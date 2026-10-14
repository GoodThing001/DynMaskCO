# D3-G0 Evidence B3-AT2B-1 — 默认关闭的 D3 专用接线（设计 + 结果）

- 预登记（先于实现冻结）：`B3AT2B1_COUNTEREXAMPLE_SPEC.json`
- 机械矩阵：`B3AT2B1_WIRING_MATRIX.json`；原始输出：`raw/B3AT2B1_stdout.txt`、`raw/B3AT2B1_instance{0,1,2}.txt`
- 新模块：`scripts/simulation/d3_execution_authority.py`；验收：`scripts/tests/test_d3_b3at2b1_wiring.py`
- **`strict_online_env.py` 未被修改**（`FROZEN=OK`，sha256 仍为 `fd17fe77…`）：D3 模式通过**实例级接线**装入，缺省不装

## 1. 三项先决条件的落实

### 1.1 reveal 清尾正式裁定为 `REVEAL_INVALIDATION`

| 操作 | 权限 | 必须满足 |
|---|---|---|
| 普通执行推进 | `EXECUTION_ADVANCE` | 只消费合法执行前缀，**不任意改变剩余客户集合** |
| reveal 清尾 | `REVEAL_INVALIDATION` | 只清可撤销后缀；**冻结承诺不变**；凭据立即失效（epoch 变化）；记录 reveal 事件 |
| 形成新后缀 | `DECISION_WRITE` | 必须重新认证后提交 |

`B3AT2H` §2.2 不变量 1 与 §2.4 的**无条件冲突已修正**：不变量 1 现表述为"EXECUTION_ADVANCE 不得改变未执行后缀的客户集合；**唯一例外**是 REVEAL_INVALIDATION 的**单调收缩**（after ⊆ before，且被清空的车必须带重规划触发）"。

**验证器（V1–V5）**：V1 承诺不得被撤销（`before.committed_next is not None ⇒ after 同值`；把 None 变为值属 DECISION_WRITE，不算撤销）、V2 单调收缩（不得新增客户）、V3 被清尾的非 closed 车必须 `needs_replan=True` 且 `replan_reason` 非空、V4 `served_mask` 不变、V5 epoch 必须改变（否则凭据不会失效）。**含负向控制**：构造"撤销承诺"的假状态，验证器必须拒绝——否则一个什么都接受的验证器什么也证明不了。

**测量窗口**：包装的 `_advance_fleet` **返回之后**快照（`strict_online_env.py:519` 返回、reveal 块在 `:523-528`），因此窗口恰好只含 reveal 块与钩子；若更早快照就会把**合法的服务完成清理 `committed_next`** 误判为撤销（这是第一版的实测缺陷，已修）。

### 1.2 `needs_replan` 与认证锚点的时序（按**实际事件时序**裁定，不按变量名）

| 时点 | 事件 | `needs_replan` |
|---|---|---|
| T0 | `_advance_fleet`（`:519`） | 若 ready 车 plan 用完 → `True` / `plan_exhaustion`（仅当原为 False） |
| T1 | reveal 块（`:523-528`） | **置 True**、`replan_reason='reveal'`、`mutable_suffix=[]`（**同一代码块**，不存在"清尾但未置位"窗口） |
| T2 | `replanner.on_reveal`（`:536`） | — |
| T4 | `_plan_and_commit`（`:545`）：prepare → `replan_ids` 由 `needs_replan` 得出 → replanner → 提交头部 → 仅对 idle/ready 车置 `False`（`:656`，committed 车保留） | 咨询过 replanner 之后才清除 |

**预声明不变量 R-NR**：reveal 之后不允许出现"后缀为空且 `needs_replan=False` 且无 `replan_reason`"的非 closed 车（V3 机械检查，25 个真实 reveal 事件全部通过）。
**结论（依实测，不依变量名）**：现有 `_run_loop` **不会**在清尾后静默失去重规划机会——它是**置位**而不是清除。

**锚点顺序（循环依赖已解决）**：`prepare_decision_point` 归 `DECISION_PREPARATION`（只改 `current_node`/`current_load`/`ready_time`，不建不删客户，**自身不需要凭据**），认证锚点在它**之后**生成（post-prepare 状态）。因此不存在"要求尚未生成的凭据去授权生成凭据所需的准备步骤"。

### 1.3 严格默认关闭

`state_authority=None` 等价路径 = **完全不装线**（`EA.make_authority(..., attach=False)`）。但"默认值为 None"不作为零影响证明：本批**实跑完整兼容性回归**（同初始状态/揭示序列/决策输入，六项比较 + 服务记录/卸货记录/服务集合逐条哈希），结果 `IDENTICAL_HISTORY_BEHAVIOUR`。

## 2. 架构（实例级接线，一个权威状态）

- **权威状态**：容器 `adopt()` **接管 env 自己的对象（不深拷贝）**，env 只持有借用引用，容器是唯一 mutator ⇒ **不存在 env 一份 + container 一份并行**（`SINGLE_AUTHORITATIVE_STATE`）。
- **replanner 隔离**：`_plan_and_commit` 期间把 `env.replanner` 换成隔离包装器，replanner 只拿到**深拷贝**；其意图随后作为**分类过的 DECISION_WRITE** 发布进权威对象。**非法候选一律不施加**（与凭据策略无关——拒绝非法计划是计划本身的性质），并记录 `illegal_candidate_not_applied`。
- **三类接口**：`decision_commit`（DECISION_WRITE）、`_advance_fleet`/`_advance_coldchain_to`（EXECUTION_ADVANCE，带 authority 身份）、`reveal_invalidation`（REVEAL_INVALIDATION，验证而非阻止）。
- **凭据策略**：本批为 `observation`（分类、记录、不变量、隔离），**不强制运行时凭据**——因为真实门在 D8 未闭合时拒绝一切计划，在接线层强制凭据会把合法执行推进全部锁死（用户明示的相反错误）。认证提交的强制执行属 AT2B-2/3；**P2 未建立**。

## 3. 结果（14 行全部为预登记判定；实例 0/1/2 一致）

`FROZEN=OK`｜`B1-DEFAULTOFF=IDENTICAL_HISTORY_BEHAVIOUR`｜`B1-AUTH-SINGLE=SINGLE_AUTHORITATIVE_STATE`｜`B1-COMPAT-METRICS=COMPATIBILITY_HOLDS`（D3 模式轨迹与历史**逐条一致**）｜`B1-REVEAL=REVEAL_INVALIDATION_VERIFIED_WITH_NEGATIVE_CONTROL`（**25 个真实 reveal 事件全部通过**，清除车辆如 `0,1,2`）｜`B1-ANCHOR-ORDER=ANCHOR_AFTER_PREPARATION`｜`B1-REPLANNER-ISOLATION=COPY_MUTATION_INEFFECTIVE`｜`B1-ADVANCE=LEGITIMATE_ADVANCE_NOT_BLOCKED`｜`B1-BYPASS=BYPASS_BLOCKED_OR_INEFFECTIVE`｜`B1-D8=D8_STILL_BLOCKING`｜`B1-PRODUCTION-OFF=ZERO_REAL_WRITE_BACK`｜`B1-WRITE-GRAPH=WRITE_GRAPH_RECORDED`｜`B1-SAFETYFIX=SAFETY_FIX_LABELLED_SEPARATELY`｜`PREREG=OK`

**写入图（运行期）**：4736 行 = `DECISION_WRITE 1988` + `DECISION_PREPARATION 100` + `EXECUTION_ADVANCE 2648`，其中**间接（经 replanner 副本）1775 行**，**UNCLASSIFIED = 0**。`REVEAL_INVALIDATION` 在图里为 0 行，因为 reveal 清尾是 `_run_loop` 内的属性赋值，**由契约验证而非路由**——这一点必须如实说明，不能读成"reveal 没有发生"（矩阵另有 25 个 reveal 事件记录）。

## 4. 八项最低验收判据核对

| 判据 | 结果 |
|---|---|
| 默认关闭：旧路径在相同输入下逐项一致 | ✅ 实跑回归，六项比较+记录哈希全等 |
| 权威状态：D3 模式只有一个权威状态对象 | ✅ `adopt` 接管，无并行副本 |
| replanner 隔离：改传入副本不改 live | ✅ rogue `[999]` 无效且被记为非法候选 |
| reveal 失效：清尾/承诺冻结/凭据过期/重规划触发 | ✅ V1–V5 + 负向控制 + R-NR，25 事件 |
| 直接旁路：未授权 `_commit` 与属性写入无法修改权威状态 | ⚠️ **部分**：`_commit` 未授权调用被路由分类并可被 `certified` 策略拒绝；**对象级属性写入不被阻止**，只被 epoch 检出（如实记录，见下） |
| 执行推进：合法派车/服务完成/返仓不被锁死 | ✅ 轨迹与历史一致 |
| D8 继续 `CAPABILITY_GAP_BLOCKING` | ✅ 未放宽、未注入假检查 |
| 生产接入默认关闭；真实写回为零 | ✅ `ZERO_REAL_WRITE_BACK` |

**必须如实说明的残余点**：`B1-BYPASS` 的结论是 `BYPASS_BLOCKED_OR_INEFFECTIVE`，其**诚实读法**是：进程内直接对权威对象赋值的**属性写入不被对象级阻止**，由 epoch 变化**检出**，并可被 `certified` 策略或 AT2G 的只读面拒绝。这与 AP2G 的威胁模型声明一致（不声称抵御任意反射），但**不能**写成"旁路已被完全阻止"。

## 5. 非声称与研究方向

- **P2/G-L/G-C 未建立**；本批验收**不**自动把 P2 判为 `PROVEN`；**不自动连续执行 AT2B-2**（`does_not_authorise_AT2B_2: true`）。
- 不解除 D8、不启动 G1/G2 或效果实验；`real_write_back_enabled=false`。
- 威胁模型：覆盖正常执行路径、意外可变引用、历史 replanner 直接写入、已登记动态调用；**不**覆盖任意反射/恶意内存修改。
- **研究方向**：本批是提交安全工程，**不构成** MaskCO 算法创新；`algorithmic_gain_claim: none`，不得把"拥有安全提交机制"写成"生成模型提升了在线决策质量"。
- I3 负结果、旧 D2 `not_qualified`、`stable_online_gain=False`、D3 G0 未通过、G1/G2 未启动均不变。

## 6. 本批自查发现并修掉的四个实现缺陷

1. **验证窗口错位**：早期快照会把合法的服务完成清理 `committed_next` 误判为撤销 → 改为在 `_advance_fleet` 返回后快照。
2. **V1 语义错**：`!=` 比较把提交循环的**合法头部提交**（None→值）误判为撤销 → 改为只判定"已存在的承诺被撤销"。
3. **容器 epoch 崩溃**：为单一决策点构造的容器被复用于整段运行，`inst/clock` 为 `None` 时 `state_iteration` 崩溃；且 `run()` 会**先**走 `_plan_and_commit` 再走 `_advance_fleet`，traces 仍为空导致 `capture_recourse_snapshot` 报 `need at least one array to stack` → 加 `set_context` + traces/served_mask 接管 + 空 traces 时回退字段哈希。
4. **非法候选被施加**：越界客户（`[999]`）被写入权威对象后又索引越界崩溃 → 非法候选**一律不施加**并记录 `illegal_candidate_not_applied`。
