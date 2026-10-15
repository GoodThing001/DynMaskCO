# A1_D3_L1_..._V1.6：P0-C 修订（9 项阻断最小修复 + 可重放验收）

- 追加记录；**V1–V1.5 与既有产物、冻结资产、`MASKCO_code/`、旧报告全部不改写**
- 代码：`scripts/simulation/d3_l1_collection_runner.py`（重写，runner sha `f4cd20cc7e64fc014d0a6375…`）
- 证据：`l1_gates/A1_D3_L1_P0C_ACCEPTANCE_V16.json`（**13/13 PASS**）、
  `l1_gates/A1_D3_L1_P0C_IDENTITY_CLOSURE_V16.json`（运行时闭包）
- 本轮**未**执行正式 64 槽位采集、未做 F−D-Geo 比较、未训练、未改预声明参数

## 0. 预声明参数（全部保持）

`dev_check` 16 日 × R1 前 4 个**真实揭示事件**、64 槽位分母、当前生效合同 `edc1b2ea…4833`、
`booking_horizon=24.0`、预算 120/60/3600 秒、失败不重采、一次 pass、不使用最终测试数据。

## 1. 九项阻断：修复前反例 → 修复后结果

| # | 阻断 | 修复前反例 / 问题 | 修复后 | 验证 |
|---|---|---|---|---|
| 1 | `ATTEMPT_STARTED` 晚于 capture | 旧 worker 在 `run_baseline_rollout()`（hook 内已 capture）之后才写账 | 在 rollout **之前**为当日 4 个预登记槽位逐个写 `ATTEMPT_STARTED`（`registered_before_capture=true`） | **N10**：账本顺序 + `slot.clock == snapshot.clock` |
| 2 | 未真正执行 `bitwise_equal` parity | 旧 worker 只算投影，不比位 | 对恢复载荷执行**真实逐位比较**：`served_mask`、`visible_mask`（`bitwise_equal`，ndarray 路径）+ 后缀一致性；不通过即 `bitwise_parity_failed` | **N9**（数组逐位 EQUAL）、worker 记录 `bitwise_parity_ok` |
| 3 | `json.dumps(default=repr)` 有损 | NumPy/对象无法无损还原 | 快照改为 **pickle protocol 4 无损**落盘，且**重载后 `snapshot_state_hash` 必须相同** | **N9**：`hash_stable=true`、`bytes=8259`；**N9b** 记录序列化方式 |
| 4 | supervisor 未独立硬执行 60/3600 秒 | 只有 day 级 `communicate(timeout=)` | 轮询循环 + **独立时钟**同时判定 day / batch / **slot**（依账本 `ts` 计算 `ATTEMPT_STARTED` 滞留），命中即 `kill()` 并由 supervisor 记 `*_DEADLINE_KILLED` 与槽位终态 | **N11**：watchdog 从账本时间戳单独识别超时槽 |
| 5 | 默认身份 JSON 只登记 `cal` | 身份门只看文件内数据字段 | 身份门**显式要求 dev_check SHA** `66573ddc…39c89`（`--expect-dataset-sha256`），与源码 SHA 双条件 | **N2**：传入 train NPZ 被拒绝 |
| 6 | closure 仅静态 AST | 无运行时来源与稳定性 | `--runtime-closure`：逐模块 `importlib` 后记录 `__file__` + **运行时 SHA** 并与扫描值比对，另记 runner **起止 SHA 稳定性** | 本次：13 模块、**13/13 runtime origin 与扫描一致**、`sha_stable=true` |
| 7 | 车辆边界后缀未接入正式比较 | `planned_suffix_sha` 仅展平 | 投影侧新增 `planned_suffix_sha_boundary`，**正式判定要求边界版与展平版同时相等** | **N4**：展平相同、边界不同 → 被识别 |
| 8 | 损坏账本可绕过一次性 pass | 损坏行仅被记录，`slot_state` 忽略之 | `ledger_integrity()` **fail-closed**：任一 `corrupt_line` ⇒ `may_attempt` 返回 `(False,'LEDGER_CORRUPT')`，worker/supervisor 全部拒绝尝试 | **N8** |
| 9 | 缺可独立重放验收入口 | 无 | `--role accept`：**N1–N12** 一条命令重放，输出 `P0C_ACCEPTANCE.json`，任一失败即非零退出 | **13/13 PASS** |

## 2. 运行前轻量验收（本轮实测）

```
python scripts/simulation/d3_l1_collection_runner.py --role accept --outdir <dir>
python scripts/simulation/d3_l1_collection_runner.py --role closure --runtime-closure --outdir <dir>
```

**N1–N12 全 PASS（13 行，含 N9b）**。覆盖你要求的五类证据：

| 类别 | 负向证据 | 正向证据 |
|---|---|---|
| **真实 reveal hook** | — | **N10**：账本先于 capture、`slot.clock == snapshot.clock`；R1 五项槽位断言见此前因果门证据 |
| **监督器杀进程** | **N11**：watchdog 从账本判定超时槽并触发 kill 路径 | 见 §3 未覆盖项 |
| **因果比较** | **N3**：任一 SHA 不等即失败；有 error 键亦失败 | **N12**：dev_check 上扰动有效（24 个不可见客户、1848 项距离变化、可见-可见不变） |
| **状态序列化** | **N9b**：明确不再使用有损 JSON | **N9**：pickle 重载后 `state_hash` 一致且数组逐位 EQUAL |
| **身份校验** | **N1**：身份文件缺失 → 硬停止；**N2**：非 dev_check 数据集被拒 | 身份门同时校验 5 项源码 SHA 与 dev_check 数据 SHA |

## 3. 仍存限制（不夸大）

1. **N11 只验证"检测 + 触发 kill 路径"**，本轮**未真实注入慢 worker** 做端到端子进程 kill；
   `SUPERVISOR_ACTUAL_KILL = NOT_EXERCISED`。注入方式已定：在验收中加一个 `--inject-slow-slot`
   使指定槽 sleep 超过 `--slot-cap-s`，预期 `SLOT_DEADLINE_KILLED` + 该槽 `FAILED(terminated_mid_slot)`
   + 后续槽 `NOT_ATTEMPTED_DUE_TO_TERMINATION`，并断言这些槽**不可重采**。需你批准后补做（属新增验收，不改预声明）。
2. worker 的逐位 parity 覆盖**恢复载荷数组与后缀结构**；完整 38 字段的逐字段位比较仍由 P0-B 的
   `bitwise_equal` 工具承担（其数组路径已在 16/16 反例门内资格化）。
3. 损坏账本的 fail-closed 会在**整批层面**阻断后续尝试（本次设计），即"宁可停"；这是有意的保守语义，
   若你希望改为"仅阻断受影响槽位 + 记录"，需另行裁定。
4. 正式入口闭包含 `dynmaskco_cc_context`（投影用 `decision_pool_from_vehicles`），已登记为 F-free 依赖。
5. 未运行 16 日受监督采集：`states_collected_for_L1 = 0`。

## 4. 状态与 Claim limits

```
P0_A = CLOSED_ON_COUNTEREXAMPLE_GATE
P0_B = PASSED_ON_PLAN_B_GATE
P0_C = REVISED_V1_6 / ACCEPTANCE_13_OF_13_PASS / SUPERVISOR_ACTUAL_KILL_NOT_EXERCISED
FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED
```
未采集正式状态、未调用 F/D-Geo、未训练、未修改冻结资产或旧报告、未重跑旧诊断与服务器计时样本。
`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、
I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
**修订通过后再次提交签署申请，不自动执行采集。**
