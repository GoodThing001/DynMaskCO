# D3-G0 Evidence B3-AC — 结果报告（只读路径分类 + 隔离副本拒绝能力验收）

- 阶段：`D3_G0_Evidence_B3AC`（用户授权的**阶段 A**；阶段 B `B3-AT` 未授权）
- 预登记（先于实现冻结）：`B3AC_COUNTEREXAMPLE_SPEC.json`，sha256 `217ca641691c2ada22e85d82388fdeedab3ffd6ed5f32879493721591c0b2b33`
- 设计：`B3AC_DESIGN.md`（含原子性三条件与非声称）
- 产物矩阵：`B3AC_PATH_CLASSIFICATION.json`、`B3AC_REFUSAL_MATRIX.json`
- 原始输出：`raw/B3AC_paths_stdout.txt`、`raw/B3AC_refusal_stdout.txt`、`raw/B3AC_refusal_instance{0..5}.txt`、`raw/B3AC_REFUSAL_MATRIX_instance{0..5}.json`
- 本批结论：**P1（门能拒绝）在隔离副本上成立且按理由归因；P2（所有真实提交都经过门）不属于本批，D3 门原语的非测试调用点实测为 0**

## 0. 两个命题分开报告（用户裁定的核心）

| 命题 | 本批结论 |
|---|---|
| **P1 门能拒绝非法动作** | **可验证、已验证（仅隔离副本）**：合法性、能耗失败、状态漂移三类拒绝均按预期理由归因，且拒绝后副本状态逐字节不变 |
| **P2 所有实际提交都必须经过这个门** | **本批不声称**。实测 D3 门原语（`compose_certification` / `pre_install_reverify` / `run_repair_transaction`）的非测试调用点 = **0**，即唯一的授权提交入口**尚不存在** |

## 1. 第一部分：路径分类（机械派生）

扫描 346 个 `scripts/**/*.py`（排除 `scripts/tests/**`），得到 12 个"决定型"原语调用点与 6 个"写入型"调用点。7 个预登记站点的分类（`B3AC_PATH_CLASSIFICATION.json`）：

| 站点 | governing 类别 | 是候选筛选器 | 有能力写 live 状态 | 被 D3 模块引用 | 接线到的 runner 文件数 |
|---|---|---|---|---|---|
| `rolling_optimizer.py:197` | `INDIRECT_WRITER_CAN_WRITE_D3_STATE` | 是 | 是（继承 `C0ReserveReplanner.plan`） | 否 | 1（`run_exp_a1_compare.py`） |
| `rrnco_accept_replanner.py:150` | 同上 | 是 | 是（继承） | 否 | 1（`run_a1_external_accept.py`） |
| `run_a1_downstream_ablation.py:88` | 同上 | 是 | 是（继承） | 否 | 1（自身） |
| `run_exp_energy_c0.py:302` | 同上 | 是 | 是（同类 `plan()`） | 否 | 4 |
| `solver_accept_replanner.py:424` | 同上 | 是 | 是（继承） | 否 | 3 |
| `pickup_budget_sweep.py:96` | 同上 | 是 | 是（继承） | 否 | 2 |
| `pickup_rejection_attribution.py:97` | 同上 | 是 | 是（继承） | 否 | 2 |

**成对陈述（不得合并）**：

1. `n_governing_D3_REAL_COMMIT_ENTRY = 0`，`n_d3_gate_primitive_non_test_call_sites = 0` → **D3 授权提交入口不存在**。
2. `n_capable_of_writing_live_state = 7`，`n_wired_to_a_d3_module = 0` → 这 7 个站点**有能力**对 live `VehicleState` 写入（继承来的 `plan()` 写 `v.mutable_suffix`，随后 `_plan_and_commit` 无门提交头部），但**没有任何 D3 模块引用它们**。

即：用户的判断得到机械证实——“七处直接调用认证原语”既不等于七个提交门，也不等于七条已接线的旁路。正确的表述是：**七个"候选筛选器 + 潜在写入者"，全部位于研究/基线运行入口，零个属于 D3 授权提交入口**。因此按用户的四条处置规则，这七条应归入"历史基线的独立运行入口"（保留原路径、登记协议差异）+ "可写入状态的间接调用"（登记为**能力**，关闭/证明不可达属于 B3-AT）。

### 1.1 分类器自身的两个自查缺陷（已修，记录在案）

- **假阴性（严重方向）**：第一版只读站点自身类体，7 个中 6 个被判 `write=False`——因为写入在**继承来的** `plan()` 里。这会把"有能力写入"误报成"没有能力"，正是用户警告的合并错误的镜像。已改为集合内继承解析并记录每一跳证据。
- **同文件局限**：`referenced_by_runner_as_replanner` 第一版只在定义文件内查找，对库式 replanner 报 `used_by=0`。已改为跨文件（`rolling`/`rrnco`/`solver` 因此分别显示出 1/1/3 个接线文件）。

## 2. 第二部分：拒绝能力验收（10 行 + 3 个控制行）

`--instances 0 --heavy-cap 8`（规范矩阵）。全部行均为预登记项，无新增、无删除（`PREREG=OK`）。

| id | 判定 | 关键实测 |
|---|---|---|
| `AC-R1` | `COUNTEREXAMPLE_REPRODUCED` | 旧路径（env 自己的 `_plan_and_commit`）：`ready / committed_next=None / suffix=[32]` → `committed / committed_next=32 / suffix=[]`，`gate_invoked=False` |
| `AC-R2` | `REFUSE_NO_MUTATION` | 同一非法后缀经受控路径：`reasons=['legality_short_circuit:no_served_in_plan']`，`legality_ok=False`，`route_feasible.not_run_illegal_order_set=True`，副本指纹前后相同 |
| `AC-R3` | `REFUSE_NO_MUTATION`（**仅品质维度**） | 9 个可执行组件**全部 ok**、`reasons=[]`，唯一阻塞是 `blocking_capability_gaps=['quality_salability']`，`quality.state=CAPABILITY_GAP_BLOCKING`；副本不变 |
| `AC-R4` | `REFUSE_NO_MUTATION` | `reasons=['energy_budget_exceeded']`（B 收紧到 0.0），副本不变 |
| `AC-R5` | `WAIT_BY_ENV_PROTOCOL` | 无待提交动作 → 受控路径与 env 自己的循环**都不写**；`counts_as_refusal_capability=False`（**不计入拒绝能力**） |
| `AC-R6` | `REFUSE_NO_MUTATION` | 状态漂移（S1→S2，计划逐字节未变）：对照（同一状态）`allow=True`，漂移后 `allow=False`，`refusals=['state_hash_mismatch']` → **拒绝由漂移引起，而非一律拒绝** |
| `AC-R7` | `COUNTEREXAMPLE_REPRODUCED` | 写入层：`plan_hash` 对 `anchor_time`/`anchor_load` 盲（两计划 hash 相同而 `anchor_time` 差 1.0），`InstallLedger.finish` **安装成功**（`ledger_installed=1`）→ 排他性缺口 |
| `AC-R8` | `COUNTEREXAMPLE_REPRODUCED` | 门层：同一凭据两次提交都 `allow=True`（纯函数、无 nonce）；写入层：同一 ledger 第二次被拒（`already_finished`，`n_installs=1`）→ 一次性只在同一 ledger 对象内成立 |
| `AC-R9` | `FAILURE_ATOMICITY_ABSENT` | `install_fn` 改一车后抛 `RuntimeError`：异常逃逸、副本上**可见部分变更**、`n_installs=0`、无事件、`finished=False` |
| `AC-R10` | `PROBE_ONLY_GATE_NOT_A_WALL` | 未注入品质检查：`installable=False`；**注入假品质检查**：`installable=True` → 门不是一堵墙，唯一阻塞项就是 D8 能力缺口。`not_a_production_path=True`、`install_permission_claimed=False` |
| `PRE` | `SETUP_OK` | 夹具取自真实运行，捕获被证明只读 |
| `AUDIT` | `OK` | 无 hook 的全新 env 重跑同一实例，整条生产轨迹哈希相同 → 探针未扰动生产路径 |
| `PREREG` | `OK` | 10 个预登记项全部覆盖 |

`verdict_counts = {SETUP_OK:1, COUNTEREXAMPLE_REPRODUCED:3, REFUSE_NO_MUTATION:4, WAIT_BY_ENV_PROTOCOL:1, FAILURE_ATOMICITY_ABSENT:1, PROBE_ONLY_GATE_NOT_A_WALL:1, OK:2}`

### 2.1 拒绝能力只计"按理由归因"的拒绝

`refusal_capability = {counted_as_capability: [AC-R2, AC-R4, AC-R6], counted_on_the_quality_dimension_only: [AC-R3], waits_not_counted: [AC-R5], wrong_reason_refusals: []}`

品质维度那一行被**单独列出**、刻意不并入 `counted_as_capability`，以免被读成 installability 或 G-L（H4）。

### 2.2 每次拒绝都必须让副本逐字节不变

`AC-R2 / AC-R3 / AC-R4 / AC-R6` 全部记录 `state_hash_before == state_hash_after`（用与 live 相同的授权捕获路径计算），即 `REFUSE_NO_MUTATION` 是真测出来的，不是措辞。

### 2.3 5/6 实例完整复现；1 个实例按前置条件缺失如实降级

`--instances 0..5` 各自保存矩阵。实例 0/1/2/4/5 给出**完全相同**的 13 行判定；**实例 3** 的 `AC-R1`/`AC-R2` 为 `NOT_VERIFIABLE`（该实例 8 个捕获决策点内没有任何已服务客户，反例前置条件不存在）——如实降级，不回落成"通过"。其余 11 行在 6/6 实例一致。

## 3. 夹具自查：两次"看起来通过"的假象

本批在夹具上踩到两个陷阱，两个都会**制造虚假的成功**，因此必须记录：

1. **空计划让认证"全过"**：第一版把"D8 是唯一阻塞项"建在 `all(len(p.suffix) > 0)` 的夹具上，结果 40/40（实例 0）与 20/20（实例 1–5）候选全部被跳过，`AC-R3` 变成 `NOT_VERIFIABLE`；更早一版则用**空计划**去跑，9 个组件里 8 个"trivially ok"，`AC-R3` 差点以一个空洞的通过收场。**空计划上的认证没有区分力**——这与 B3-QC 的"覆盖率"教训同源。
2. **候选筛选条件过强**：实测实例 0 只有**服务完成事件**上才有非空尾部（如 clock 1.2554，3 辆 committed 车共 19 个客户），且大多数车尾部为空。把条件从 `all` 改成 `any` 之后才真正拿到"客户非空"的可认证计划。矩阵现在保留 `why_roomy_B` 与 `attempts`，说明为什么需要 B=1e9 才能把 D8 缺口单独隔离出来（标称 B=300 下该计划**能耗不可行**，这本身是有用事实，并让 `AC-R4` 的能耗行有意义）。

## 4. 原子性三条件的实测状态（全部未达成）

| 条件 | 判定行 | 实测 |
|---|---|---|
| **A1 排他性** | `AC-R7` = `COUNTEREXAMPLE_REPRODUCED` | 凭据层拒绝漂移（`AC-R6`）但写入层只比 `plan_hash`、从不咨询凭据，而 `plan_hash` 盲于 `anchor_time`/`anchor_load` → 漂移下安装成功 |
| **A2 一次性** | `AC-R8` = `COUNTEREXAMPLE_REPRODUCED` | 门是纯函数，同一凭据可重复授权任意多次提交；`already_finished` 只在同一 ledger 对象内生效 |
| **A3 失败一致性** | `AC-R9` = `FAILURE_ATOMICITY_ABSENT` | `install_fn` 在无守卫处调用、`finished` 事后置位 → 异常逃逸且部分变更未标记 |

**H5**：在没有任何持久事务机制的前提下，本批**不**把上述任何结果描述为崩溃恢复意义上的原子性；本批不提供该性质，也不声称提供。三条件属 B3-AT 实现范围。

## 5. 只读与禁用项

| 禁项 | 状态 | 证据 |
|---|---|---|
| 生产环境唯一写入者改造 | 未做 | 受控提交路径仅存在于 test 文件，未注册到任何 env |
| 七条历史路径统一重写 | 未做 | 只读取 AST；相关源码 sha256 与预登记一致 |
| 修改 `_commit` 历史语义 | 未做 | `strict_online_env.py` sha256 `fd17fe77…`（不变，已 pin） |
| 解除 D8 | 未做 | `AC-R3` 仍因 `quality_salability` 拒绝；`AC-R10` 显式 `install_permission_claimed=False` |
| 真实安装 / G1 / G2 / F−D | 未做 | `production_state_written=False`、`real_writeback_enabled=False`；`AUDIT=OK` |

## 6. 本批**未**得出（不得声称）

- F10 未在生产路径关闭（门无生产调用点；`D3_REAL_COMMIT_ENTRY=0`）。
- G-L / G-C 均**未建立**。
- `stable_online_gain` 仍为 `False`；I3 正式负结果、旧 D2 `not_qualified`、G1/G2 未启动，全部不变。
- P1 成立**只在隔离副本上**；它不构成"生产门已生效"，也不构成 B3-AT 的任何部分已完成。

## 7. 给 B3-AT 的输入（不是授权）

按 A2 的架构要求，B3-AT 需要落实的是：唯一写入者是**由闸门授权的提交事务**（判定能力与写入能力分离）、授权与写入绑定为一次原子操作（凭据必须被写入侧独立校验，含 `state_hash` 而不只是 `plan_hash`）、凭据一次性（nonce/序列）、以及写入异常的失败一致性标记。本批提供的是这三条**当前都不成立**的可复现测量。

## 8. 复现命令

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem
& D:\AA_Py\A_Anaconda\python.exe -B scripts/tests/test_d3_b3ac_path_classification.py
& D:\AA_Py\A_Anaconda\python.exe -B scripts/tests/test_d3_b3ac_gate_refusal.py --instances 0 --heavy-cap 8
# 稳定性扫描
for i in 0 1 2 3 4 5; do python -B scripts/tests/test_d3_b3ac_gate_refusal.py --instances $i --heavy-cap 8; done
```
