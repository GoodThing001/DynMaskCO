# A1-D3-L1 合同协议身份冻结（CONTRACT-PROTOCOL-FREEZE）

- 轮次：用户授权的唯一交付——L1 合同协议身份文件 + 轻量回归
- 工具：`scripts/tests/test_d3_l1_contract_protocol_freeze.py`（只读；未改历史合同/profile/冻结源码/归档报告）
- 证据：`l1_gates/A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json` + `..._FREEZE_{stdout,stderr}.txt`
- 结果：**7 行 / 7 PASS / 0 FAIL → `L1_CONTRACT_PROTOCOL_FREEZE = FROZEN`**
- **独立只读复核**：同工具重跑，身份 JSON **逐字节一致**（sha `f44e3456db5fac67c85829e5…`）

## 1. 冻结的 L1 身份（按你的裁决登记）

| 字段 | 冻结值 |
|---|---|
| `calibration_source_contract_sha256` | `0f00af5d197cf35368b1b2e807bcaf53d9b58a1cd425a7f53ec66091f6acce28`（**仅来源身份**，不参与构造） |
| `l1_base_contract_sha256` | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0`（**当前物理协议**） |
| `l1_effective_contract_sha256` | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` |
| `objective_profile_sha256` | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| `booking_horizon` | `24.0`（独立公开协议常数） |
| `calibration_vs_execution_contract` | `CROSS_VERSION_EXPLICIT` |
| `historical_result_comparability` | `NOT_AUTOMATICALLY_ESTABLISHED` |
| `l1_uses_calibration_contract_as_base` | `false` |

同时登记：`coldchain_contract.py` / `coldchain_state.py` / `strict_online_env.py` /
`recourse_snapshot.py` / `d3_l1_state_collector.py` / 本工具的 sha256，数据 NPZ sha256 与 profile 文件 sha256。

标签（与你一致）：`CALIBRATION_SOURCE_IDENTITY = VERIFIED`、
`L1_CONTRACT_ROUTE = CURRENT_PHYSICS_PROTOCOL_RECOMMENDED`、
`L1_CONTRACT_PROTOCOL_SELECTION = APPROVED_IN_PRINCIPLE`、
`L1_COLLECTOR_CONTRACT_FROZEN = PENDING_MANIFEST_AND_RUNTIME_CHECK`（本轮身份清单已就位，故该标签的程序性前提已满足，最终签署仍归你）。

## 2. 回归结果（7/7）

| 行 | 检查 | 结果 |
|---|---|---|
| I1 | L1 基础合同来自**当前**模块且哈希为登记值；当前源码 sha ≠ 校准期源码 sha | PASS |
| I2 | `base + apply_objective_profile` → effective 哈希为登记值；profile 哈希与声明一致 | PASS |
| I3 | 校准合同仅作**来源身份**，不是 L1 基础合同 | PASS |
| I4 | 应用 profile **只改变 `objective` 段**：C0 物理/单位/品质约束原样带过 | PASS |
| I5 | **跨版本对象兼容性核查**（见 §3） | PASS（确认不兼容） |
| I6 | 在冻结合同下快照为 `recourse_snapshot_v3_cc`、合同哈希一致、`validate_snapshot_against_env` 通过 | PASS |
| I7 | L1 成本语义分离：F-free 采集器**不计算任何成本/效用** | PASS |

## 3. 你的跨版本技术判断：**机械证实**

- 当前 `coldchain_state.py` **确实引用** `contract.thermal.precool_energy_per_unit_per_c`；
- 由归档冻结源码重建的历史校准合同，其 `ThermalConfig` **缺少 8 个字段**：
  `dispatch_delay_hour`、`diurnal_amplitude_c`、`diurnal_phase_hour`、`precool_energy_per_unit_per_c`、
  `precool_min_temp_class`、`precool_station_discount`、`precooled`、`use_station`；
- 结论 `CROSS_VERSION_OBJECT_INCOMPATIBLE_CONFIRMED`：**把旧合同对象交给当前冷链状态转移函数，
  在相关取货路径上会因缺字段而失败**。故"历史合同可重建 ≠ 与当前仿真器运行兼容"成立，
  这也构成选择当前物理协议作为 L1 主诊断环境的**工程理由**（除研究理由之外）。

## 4. 写入身份清单的已知限制

1. **跨版本校准限制**：objective profile 是在历史合同 `0f00af5d…`（无 diurnal/precooling）下校准的；
   L1 走当前协议 `32124bfd… → edc1b2ea…`，**profile 未在该协议下重新校准**。L1 可把它作为
   **显式声明的跨版本诊断配置**使用（此层不判定正式效用优势）；**L2/L3 必须单独解决校准适用性与公平性**。
2. `historical_result_comparability = NOT_AUTOMATICALLY_ESTABLISHED`。
3. P1b 覆盖边界保留：`canon()` 归一化 NumPy 标量、跨类型数值路径对正负零有例外，
   "IEEE-754 位一致"不等于每个浮点叶节点均已逐位验证；归档 `FULL_PARITY` 标签不变。
4. 成本口径：L1 采集不做评价，故 `coldchain_cost` 与 `distance_cost` 在本层不可能混用；
   混用风险只在 L2/L3（`J = revenue − fuel − reject_loss`）评价处才需绑定。

## 5. 未做与未授权 / Claim limits

未执行：服务器计时资格（未申请）、64 槽位采集、F−D-Geo 比较、模型重训练；
未修改历史合同、profile、冻结源码或已归档报告；未重新搜索历史合同、未重跑上一轮溯源工具。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部原样保留。
本文件是**合同协议身份冻结**，不产生任何 MaskCO 增益证据，也不构成 L2/L3 资格。
