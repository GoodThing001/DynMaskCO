# 32 状态恢复性：资格判断与恢复测试方案（任务 B，只读）

**日期**：2026-10-14 · **性质**：只读资产核查 + 方案设计；**未执行任何恢复、未运行任何实验**
**机器可读结果**：`D3_STATE_RECOVERABILITY.json`（本文件是其叙述版）

---

## 1. 追查结果

| 项 | 结论 |
|---|---|
| `RecourseSnapshotV2` 是什么 | `scripts/simulation/recourse_snapshot.py:20` 的 `SNAPSHOT_SCHEMA_VERSION = 'recourse_snapshot_v2'`；冷链变体 `'recourse_snapshot_v3_cc'`（`:21`）。不是独立文件/类名，而是**快照 schema 版本标识** |
| 生成接口 | `capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers=None)`（`:80`） |
| 恢复/续行接口 | `restore_recourse_snapshot`（`:214`）、`resume_from_snapshot`（`:350`）、`validate_snapshot_against_env`（`:318`）、`snapshot_state_hash`（`:68`） |
| 真实字段数 | **38** 个顶层字段（由 `snap = {...}` 字典字面量机械提取，不是转述） |
| 原始来源文件 | ① 现行 `scripts/simulation/recourse_snapshot.py`（sha `0af3535a…`）；② 冻结/归档副本多份（`archive/正式冻结包/.../source/...`、`archive/诊断与作废冻结包/...`、`03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/...`）⇒ 模块本身**有历史版本可核** |
| 文档口径 | `docs/当前规划/研究设计/论文方法部分草稿.md:29` 明确：`state_hash` 对快照字段与路线顺序做确定性绑定，**恢复前必须重算并精确匹配，缺失或不一致视为协议错误** |

## 2. 32 状态资产到底存了什么

`results/m0_scale/sgbs_fixed_cal/` **只有两个文件**（服务器；**本地无该目录**）：

| 文件 | 字节 | SHA-256 | 内容 |
|---|---|---|---|
| `per_state.json` | 10,795 | `c50a7db0…` | **32 条**记录，每条的键**只有** `inst` / `event` / `J0` + 7 个标量 J（`R`、`Mpre_greedy`、`Mpre_sgbs`、`Mtrained_greedy`、`Mtrained_sgbs`、`dist_sgbs`、`Mtrained_greedy7`） |
| `summary.json` | 1,522 | `bbe4d2d8…` | `split`/`n_states`/`n_instances`/`accepted_mean`/`wallclock_total_s` + 6 组配对差；**没有任何 identity/source/config 块** |

**⇒ 该资产自身不含来源身份登记**（这一点必须写明）。

## 3. 三件事必须分开（用户要求）

| 对象 | 现状 |
|---|---|
| **状态索引** | **有**：32 × `(inst, event, J0)` + 7 个标量 |
| **完整 snapshot** | **没有**：38 个快照字段**一个都没存**；`summary.json` 也没有快照 |
| **已执行的确定性恢复** | **没有**：从未对这 32 个状态做过 restore / resume / 恢复 |

**为什么当初没存快照**：采集器 `Probe.states.append({...})`（`run_fixed_state_quality.py:116`）
里存的是 **`'env': env` 活对象引用**，外加 `P0`/`vis`/`vehicles` 等对象 ⇒ **本来就不可直接序列化**；
实际落盘的只有标量。`sgbs_fixed_state.py` 随后只写 `summary` 与 `rows`。

## 4. 字段覆盖表（38 字段，缺失标记 MISSING，不猜测）

* **PRESENT（2）**：`instance_id`（即 `inst`）、`event_id`（即 `event`）
* **MISSING（36）**：`schema_version`、`replan_count`、`clock`、`reveal_idx`、`num_vehicles`、`num_nodes`、
  `served_mask`、`visible_mask`、`customer_universe`、`vehicle_status`、`vehicle_node`、`vehicle_ready`、
  `vehicle_load`、`committed_next`、`committed_arrive`、`committed_finish`、`needs_replan`、`replan_reason`、
  `mutable_suffix`、`mutable_suffix_len`、`served_route`、`served_route_len`、`dispatch_time`、`return_finish`、
  `trace_dispatch_time`、`trace_return_depart`、`trace_return_arrival`、`services_prev_node`、`services_node`、
  `services_depart`、`services_arrival`、`services_start`、`services_finish`、`services_len`、`force_suffix`、
  `replanner_state`
* **额外锚点**：`J0`（每状态一个浮点，不属快照字段，但**可作恢复锚点**）
* 逐状态同样适用：**每个状态都只有 3 个可用值，36 个字段 MISSING**

## 5. 资格判断

> **`index_only_recovery_requires_a_new_execution`**

* **仅凭封存资产无法恢复**这 32 个状态（没有 snapshot，也没有 `state_hash`）。
* **唯一可行路径是确定性重采集**：以登记的 split（`cal`）、`max_per_instance=2`、登记数据与登记代码，
  重跑采集并在每个决策点 `capture_recourse_snapshot`，再要求 `(inst, event, J0)` **精确复现**。
* **但这条路径不是只读的** —— 采集器从**实时 env** 派生状态，复现它就要**执行模拟器** ⇒
  属**新的执行**，**不在本轮授权内**，且必须先有独立预声明。

## 6. 未来恢复运行所需的身份（已在本次审计中读全）

| 类别 | 值 |
|---|---|
| 数据 | `cal_teacher.npz = c35fa683…20f1`、`M0_TEACHER_MANIFEST.json = b84e1716…af04`、`results/m0_scale/cal/manifest.json = 1fa4de83…a593` |
| 代码 | `recourse_snapshot.py = 0af3535a…27f8`、`run_sgbs_fixed_state.py = 87519219…3309`、`run_fixed_state_quality.py = fd38cb3c…6544`、`visible_state.py = 710fe75c…e046`、`action_contract.py = 9d64273a…b53b`、`mpre_policy.py = 5b03e4ca…a191` |
| 采集参数 | `split=cal`、`n_instances=16`、`n_states=32`、`max_per_instance=2`、`mask_size=2`、mask 种子规则＝该实例的访问计数 |
| **⚠️ 身份缺口** | **合同哈希与 objective profile 哈希未在该资产中登记**（`summary.json` 无 identity 块）⇒ 必须先定位或声明，否则未来的恢复运行**不能声称身份合格** |

## 7. 独立验收判据（写入 `D3_STATE_RECOVERABILITY.json`）

对**全部 32 个状态**逐条：

1. `(instance_id, event_id)` 与 `per_state.json` 记录**精确一致**；
2. `J0` 以**全精度 repr** 逐位复现；
3. `snapshot_state_hash` 在 capture 与 restore 之间**重算并相等**；
4. `validate_snapshot_against_env` 通过；
5. restore → resume 的后缀与原始 rollout 后缀**逐位一致**（resume parity）。

**分母规则**：32 个状态**全部留在分母**；复现失败者**记为 FAILED 行，不得剔除**；
空/不可产出的状态**原样记录**；**不得**用其他资产替代、**不得**从下游值反推缺失字段。

**身份规则**：上表每一项 sha256 必须逐一匹配，否则 `not_qualified`；
**合同/profile 身份缺口必须先闭合**。

**结论取值**：`recovered_verified` / `partially_recovered` / `not_reproducible` / `not_qualified`。

## 8. 本轮不主张

**本文件与 JSON 都不表示这 32 个状态已被恢复**，也**不重新确立任何 I3 资格**；
`stable_online_gain` 仍为 `False`。**本轮仅交付资格清单与恢复测试方案，不执行真实 32 状态 F−D 比较。**
