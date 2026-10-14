# B1 真实状态级反例与 no-op 装配报告（受限实现）

**日期**：2026-10-14 · **基准 commit**：`03aec6e` · **授权**：受限 B1（用户裁决 2026-10-14）
**性质**：**只读状态级实现**。真实 `StrictOnlineEnv` + 真实 `VehicleState`；零生产写回；未改 `_commit`；未接管生产决策；无跨进程哈希协议；无持久化回执；未训练；未跑效果实验。

| 新增产物 | 作用 |
|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_state_assembly.py` | 只读装配接口：post-prepare 状态捕获、按车辆解析的掩码资格、pool 报告、共同 legal、凭据 |
| `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_b1_state_level.py` | C1–C5 状态级反例 + no-op 等价 + 7 条验收 |
| `research/…/d3_g0_evidence_root_v1/B1_STATE_LEVEL_MATRIX.json` | 机器可读矩阵（含全部身份、环境、代码 SHA） |
| `research/…/d3_g0_evidence_root_v1/raw/B1_state_level_{stdout,stderr}.txt` | 原始输出（stderr 0 字节） |

**注册运行**：`--instances 0 1 2 3 4 5 --heavy-cap 60` → **600 个决策点**（其中 360 个做重工作），
exit 0，stderr 0 字节，**14/14 检查 PASS**，`all_conclusions_definite: true`，`stop_condition_triggered: false`。

---

## 0. 结论摘要：C1–C5 全部 CONFIRMED

| # | 反例 | 结论 | 关键实测 |
|---|---|---|---|
| **C1** | 已承诺订单 | **CONFIRMED** | **6109** 个 committed 客户全部被排除，0 违规 |
| **C2** | 冻结已执行前缀 | **CONFIRMED** | **5651** 个 served 客户全部被排除，0 违规；`served_mask` 与 `served_route` 在 360 个决策点上**零分歧** |
| **C3** | 保护车辆 | **CONFIRMED** | **695** 个 protected-tail 客户被排除；`validate_repair_plan` 正确报 `protected_vehicle_changed` |
| **C4** | 状态版本变化（+D4） | **CONFIRMED** | 354/354 相邻决策点 `state_hash` 改变；133 次 reveal **全部**清空可变尾；D4 两条盲区均成立；**现存安装门放行了"同计划、异状态"** |
| **C5** | 认证后提交失败 | **CONFIRMED** | 在**深拷贝**上 `_commit` 接受了**已服务客户**；提交循环静默跳过 `closed` 车；**live 状态未变** |
| **POOL** | 空 `pool_B` 保留 | **CONFIRMED** | 360 个重决策点中 **263** 个空池，全部以字面 `[]` + `empty=True` 记录，**决策点未被剔除** |
| **R2** | 池权威口径 | **CONFIRMED** | 82 个决策点出现 `pool_B ∩ reserved ≠ ∅`；这些成员**仍然留在 `pool_B`**（未被交集收缩） |

**七条验收标准**：全部满足（明细见 §4）。**未触发停止条件**——真实状态在不改动生产行为的前提下可被捕获。

---

## 1. 关键使能点：只读克隆真实状态

`recourse_snapshot.restore_recourse_snapshot(snapshot)` **不接收 `env` 参数**，返回
`(vehicles, traces, served_mask)`（`recourse_snapshot.py:214-315`），因此可以对决策点做**独立副本**，
在副本上跑 `prepare_decision_point` + incumbent 规划 + 共同 legal 枚举，**真实运行完全不受影响**。
这是"在不改变生产行为下捕获真实状态"的唯一入口，也是本轮未触发停止条件的原因。

**写回自证**：探针在每次决策点的**前后**各取一次 live 指纹（`snapshot_state_hash` 走同一捕获路径，
覆盖 `replan_count` / `force_suffix` / `replanner.export_state()` 及全部车辆字段）：
`RO` 检查在 600 个决策点上**全部相等**，即探针**一个字节都没写**。
更强的 `EQ` 检查：同一实例"装探针"与"不装探针"两次运行的 **traces + served_mask 逐字段一致**（6/6 实例）。

---

## 2. 逐条反例的证据

### C1 已承诺订单（6109/6109 排除）

按车辆解析：对每个可见客户记录"它出现在哪些 **mutable** 车的当前 suffix 中"，再叠加
`not committed ∧ not served ∧ not protected`，四项在**同一个 post-prepare 状态版本**上计算。

**用户澄清的落地（构造式演示，在深拷贝上）**：自然运行中**没有**出现"已承诺客户同时位于某可变车 suffix"的配置
（`naive_accept_configurations = 0`），所以这一条**不能靠自然样本证明**。我在副本上构造了它：
把已承诺客户 32 注入可变车 3 的 suffix →
- 并集式（union-of-suffix）成员测试：**会接受**（`true`）
- 按车辆解析 + committed 规则：**拒绝**，记录理由 `["committed"]`，且 `eligible_after_injection = false`

⇒ 用户的担心是**可复现的真实风险**，而不是理论顾虑；当前检查器按车辆归属解析，因此挡住了它。

### C2 冻结已执行前缀（5651/5651 排除）

独立计算 served 集合 = `served_mask` **∪** 每辆车 `served_route`，然后断言 `eligible ∩ served = ∅`（真检查，非重言式）。
360 个决策点上 `served_mask` 与 `served_route` **零分歧**（该诊断位本身有价值：目前口径一致）。
同样在副本上构造了 served 客户注入可变车 suffix 的演示：并集口径接受、按车辆规则拒绝（理由 `["served"]`）。

### C3 保护车辆（695 个排除）

**保护车选择规则（事先固定、统一适用）**：指定"**suffix 非空且最长**的 mutable 车"（并列取较大 vid）。
理由必须写明：若保护一辆 **suffix 为空**的车（例如"最高 id 的空闲预留车"），它排除不了任何客户，检查会**空转**——
我第一版用 `max(mut)` 只拿到 **1** 个观测，正是这个原因；改用上述规则后升到 **695** 个观测。
另：`validate_repair_plan(plans, base, mask, protected)` 对被改动的 protected 车正确报出 `protected_vehicle_changed`。

### C4 状态版本变化 + D4 计划身份盲区

1. **状态漂移可观测**：354/354 相邻决策点的 `state_hash` 不同。
2. **reveal 使可变尾失效**：以"可见集增长"判定 reveal（**不能用 `replan_reason`**，见下），
   133 次 reveal **全部**伴随非 closed 车可变尾被清空。
3. **D4 盲区（真实计划集上的构造）**：把真实计划集的 `anchor_time` 平移 +1.0（或 `anchor_load` +5.0）后，
   `plan_hash` **完全相同**（16 hex），而新加的 `plan_digest_anchor_v1` **不同**。
4. **现存安装门对状态漂移是盲的（精确隔离）**：用**同一个计划对象**构造两个凭据，只有 `state_hash` 不同
   （`same_plan_hash: true`、`different_state_hash: true`、`credential_diff.plan_hash: false`）→
   `InstallLedger.finish` **安装成功**（`existing_gate_installed_despite_state_difference: true`，拒绝理由 `null`）。

⇒ 你指出的第 2 点得到**可复现的证实**：只绑 `plan_hash` 不足够，而且不是理论问题。

**顺带发现的陷阱（已记录，供 B2 参考）**：`replan_reason` 是**粘滞**的——env 在 reveal 时置 `'reveal'`，
之后**从不清除**，所以用它判定 reveal 会把 353 行算成 reveal，而真实 reveal 只有 133 次。
任何未来的"事件类型"判定都不得依赖该字段。

### C5 认证后提交失败（深拷贝上）

在**深拷贝**上实测两件事（`live_state_unchanged: true`，live 指纹前后相等）：
1. `_commit(inst, clock, v, traces, c)` 对**已服务客户** c=1（`served_in_mask=true`、`in_served_route=true`）
   **照样提交**：`status: idle → committed`、`committed_next: null → 1`。即 `_commit` **零校验**。
2. 提交循环的守卫只对 `status in ('idle','ready')` 生效，`closed` 车被**静默跳过、无任何记录**。

⇒ "安装回调被调用"与"真实状态已提交"确实可以不一致。**但**：本轮的 C5 是**副本上的机制验证**，
运行期的安装级证伪需要安装写回授权（本轮**未授予**），因此 §5 把它列为待授权项。

---

## 3. 你要求澄清的那一点（掩码资格必须是按车辆解析的决策时刻条件）

已在 `d3_state_assembly.mask_eligibility` 实现，并且**不是**"客户 id 在某个列表里"：

- `b_eligible_by_customer_vehicle`：客户 → **拥有它的 mutable 车 id 列表**（逐车解析、可追溯）
- `committed_next_by_customer_vehicle`：客户 → 承诺它的车
- `served_nodes`：`served_mask` ∪ 全部 `served_route`
- `protected_tail_by_customer_vehicle`：客户 → 保护它的车
- `exclusion_reasons`：每个被排除客户的理由集合
- `eligible`：四项同时成立者

四项全部来自**同一批** `vehicles` / `plans` / `served_mask` 对象（同一 post-prepare 版本），
并在报告里记录该版本坐标 `state_version = {inst_idx, clock}`。

---

## 4. 验收标准逐条对照

| # | 标准 | 结果 | 证据 |
|---|---|---|---|
| 1 | 五个反例均有明确结论，允许 `NOT_VERIFIABLE`，不得筛除失败状态 | ✅ C1–C5 全部 **CONFIRMED** | 矩阵 `conclusions`；空池决策点也**未被剔除**（263 个全留） |
| 2 | 全部使用真实 `VehicleState`，不得再伪造 `committed_next` | ✅ | `REAL`：全部车辆类为 `strict_online_env.VehicleState`；未注入任何属性（对照 Red Phase A-3b） |
| 3 | 零写回 no-op 路径与原始 baseline 一致 | ✅ | `EQ` 6/6 实例逐步一致；`RO` 600 个决策点指纹不变 |
| 4 | strict-online 可见信息与认证状态明确分离 | ✅ | `SEP`：`pre_prepare_context` 与 `post_prepare_state` 在 360/360 上**不同**；`SEP2`：可见上下文**不含** post-prepare 专属字段 |
| 5 | 空 `pool_B` 保留，不事后剔除 | ✅ | `POOL`：263 个空池全部为字面 `[]` + `empty=True` |
| 6 | 身份、运行环境、代码 SHA 登记完整 | ✅ | 矩阵 `identities`：数据/ profile / 合同 / 场景 / 6 个代码 SHA / 运行环境 |
| 7 | 不修改历史冻结树、旧裁决、测试原件 | ✅ | `HIST`：**9 个历史测试文件**与冻结记录逐字节一致；生产源码等于 **Green Phase 登记的修复后哈希**（而非修复前） |

**关于第 7 条的自我纠正**：我第一版 `HIST` 检查把**生产源码**也要求"等于修复前哈希"，于是它报 FAIL ——
这是**检查口径错**，不是代码错（`d3_geo_adapter.py` 就该在 Green Phase 变过）。已改为：测试原件必须不变、
生产源码必须等于登记的**修复后**哈希。这条纠正本身留在矩阵里。

---

## 5. 本轮**没有**做（明确边界）

- **未做真实安装写回**；未修改 `_commit`（其零校验行为只被**观测**，未被改动）；未让组合认证器接管生产决策。
- **未定义跨进程状态哈希协议**。凭据里 `state_hash_scope = 'in_process_same_run_only'`；
  `commit_state_digest_v1` 一类的新审计身份属于 **B3**，本轮不宣称任何跨进程可复核性。
- **未做持久化回执**（`P_after_install` 仍不存在）。
- **未训练、未跑效果实验**；未做 F−D、未重采 32 状态、未跑 G1/G2、未触碰最终种子。
- **C5 的运行期证伪**（真实安装后再断言"状态未提交"）需要安装写回授权，本轮未授予。

**科学口径**：本报告**不构成**任何效果结论。no-op 路径与 baseline 逐字段一致，因此它
**不是**新的科学结果，也不是"MaskCO vs 几何基线"的证据。它的价值是：消除未来 F−D 比较中最严重的混杂
——模型看见的状态 / 认证使用的状态 / 环境真正提交的状态**可能并不相同**。

---

## 6. 登记的实测事实（供 B2/B3 使用）

| 项 | 值 |
|---|---|
| cal 数据 | `data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz`，sha256 `c35fa6834b8007922160f7979680c1fb8c6dda4d93309cd5ef509b57d83f20f1`（**与任务 B 登记的 cal npz 身份一致**） |
| profile | `o0cc-pilot-devmean-equal-v2`，`profile_hash = df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| 合同 | `contract_hash = edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833`（v1，`shared_capacity=50.0`） |
| 场景 | 16 实例 / 51 节点 / 25 车 / `dist_mat` 字节 sha `377fe552…` |
| 生产源码（修复后） | `d3_geo_adapter.py = 099c3517472d886ece4ea3e929df2ec54d0e459b2a2e2e07da5b658acc07adc5` |
| 只读装配 | `d3_state_assembly.py = 03fa7c164b269910cd33011de2a0f6017a97ce5bf1832a788ebbd81032545874` |
| 环境 | Windows / Python 3.12.7 / numpy 1.26.4 / `D:\AA_Py\A_Anaconda\python.exe` |
| 发现的陷阱 | `replan_reason` 粘滞；`plan_hash` 16-hex 且不含 `anchor_time`/`anchor_load`；`_commit` 零校验；提交循环静默跳过非 idle/ready 车 |

---

## 7. 复现

```powershell
cd D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
& D:\AA_Py\A_Anaconda\python.exe -B scripts\tests\test_d3_b1_state_level.py `
    --instances 0 1 2 3 4 5 --heavy-cap 60     # exit 0, ~21 s, stderr empty
```

---

## 8. 下一步（等你裁决，本报告不自动推进）

按你给的路线：B1（本报告）→ **B2** 完整认证组合与三重身份绑定 → **B3** 真实提交事务与持久安装回执 →
G0 独立验收 → G1 工程烟测 → G2 正式效果评价。
**B1/B2/B3 全部完成也不自动意味着 G0 通过**：之后仍需身份冻结、资源门、实际调用链与独立只读验收；
历史 32 状态恢复与 E2 训练仍是另外的资格事项。

当前状态不变：**D3 G0 未通过**，G1/G2 未启动，**无可归因的 F−D 学习增益**，
I3 正式负结果 / 旧 D2 `not_qualified` / `stable_online_gain=False`。
