# D3-L1 恢复轨迹等价性归因裁决（P1b + 冷链身份补强）

- 授权范围：用户限定的 P1b 归因与身份补强（不授权服务器计时、不授权 64 槽位采集）
- 工具：`scripts/tests/test_d3_l1_resume_parity_attribution.py`（新增；未改任何冻结源码/数据）
- 证据（**同一次执行**绑定源码 SHA + run_id + argv + stdout/stderr + JSON）：
  `l1_gates/A1_D3_L1_RESUME_PARITY_ATTRIBUTION.{json,_stdout.txt,_stderr.txt}`，**run_id `20261015T103956-e9385c6a`**
- 结果：**9 行 / 8 PASS / 1 FAIL**；失败原件已保留（见 §5），未改写为通过

## 1. 判据（运行前固定）

| 情形 | 裁决 |
|---|---|
| semantic > 0 | `BEHAVIOR_DIVERGENCE` |
| strict > 0 且 semantic = 0 | `REPRESENTATION_ONLY`（strict 仍记失败） |
| strict = 0 且 semantic = 0 | `FULL_PARITY` |
| 比较无法完成（节点预算/深度守卫/对象不可读） | `NOT_VERIFIABLE` |

本轮另设 node budget 400000 与深度上限 14，**均未触发**（`complete=True`）；耗尽预算即强制 `NOT_VERIFIABLE`。

## 2. 结论：**`FULL_PARITY`**（两槽位、完整轨迹、零差异）

| 槽位 | strict typed | semantic value | 说明 |
|---|---|---|---|
| slot 1（clock 0.14656） | **0 差异**（complete） | **0 差异**（complete） | `served_mask` 全等；state identity 一致 |
| slot 2（clock 0.49480） | **0 差异**（complete） | **0 差异**（complete） | 同上 |

即：**上一轮的 `traces[0].services[0].arrival_time` 类型差异是（我的）比较器假象**，不是执行行为分歧：
- 旧 `canon()` 在深度 > 6 处返回 `'<depth-limit>'`，深层字段实际未被比较；
- 旧 `first_difference()` 先按 Python 类型判定，`numpy.float64` vs `float` 会提前返回类型差异。

**必须同时保留的限制（不夸大）**：本轮的 strict 比较运行在**规范化投影**上——`canon()` 会把 0 维 numpy 标量归一为 Python 数，
因此**原始对象的类型差异本轮未被直接检验**；semantic 比较则对数值做了**精确相等 + IEEE-754 位模式一致**的检查。
所以准确表述是：**数值与结构语义完全一致；原始类型层面的差异未复现，且不再有证据支持行为分歧**。
按预固定判据，semantic=0 → 不构成 `BEHAVIOR_DIVERGENCE`。

## 3. 冷链嵌套身份（P0a 补强）

| 检查 | 结果 |
|---|---|
| `schema_version` | **`recourse_snapshot_v3_cc`**（两槽位）✓ |
| 快照 `coldchain_contract_hash` == effective contract hash | ✓（两槽位） |
| 嵌套冷链字段齐全 | ✓ `vehicle_coldchain_state` / `coldchain_updated_time` / `committed_coldchain_before` / `coldchain_accounting_anchor` / `trace_final_coldchain_state` / `trace_depot_unload_records` / `trace_dispatch_preconditioning_energy_kwh` / `trace_return_segment_energy_kwh` / `services_arrival` |
| 逐车辆冷链状态 | slot1 **3/25**、slot2 **6/25** 车携带 `coldchain_state`（其余车尚未装载/派遣，符合语义） |
| 恢复载荷三对象 | `arity=3`（类型 `[list, list, ndarray]`）✓ |
| capture → restore → 再捕获 | `snapshot_state_hash` 一致 ✓ |

结论：**P0a 的 `coldchain=false` 是检查层次错误**（冷链字段嵌套在车辆/轨迹内，不是顶层），你的判断正确；
现在按真实层次验证通过。上一轮"限定范围 PASS"保持不变，本轮升级为**冷链嵌套字段已验证**。

## 4. profile 身份核对（8/11 → 3 个被过滤键）

被过滤键：`hard_gate`、`profile_hash`、`provenance`。
- `filtered_semantically_significant = []` → 无影响目标函数的参数被丢弃（fail-closed 检查通过）；
- `hard_gate = {'total_instances': 1152, 'hard_pass': '1152 / 1152'}` → 是**既有验收结果记录**，不是目标函数参数；
- **重算哈希断言通过**：`declared_profile_hash == recomputed_profile_hash == df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6`。

## 5. **本轮唯一的 FAIL：合同身份未闭合（A3，fail-closed）**

```
provenance.profile 声明的 contract_hash = 0f00af5d197cf35368b1b2e807bcaf53d9b58a1cd425a7f53ec66091f6acce28
实际 instantiate 的 effective contract  = edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833
base（default_pilot_contract）          = 32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0
                                                     => match = false
```

即：**注册 profile 是为 `0f00af5d…` 那份合同构建的，而本轮实际生效合同是 `edc1b2ea…`**。
这证实你的判断——`IDENTITY_CLOSED` 未闭合，且给出了确切落差。
按你的裁定，标签保持 **`L1_COLDCHAIN_CONTRACT_CANDIDATE / IDENTITY_PARTIAL`**；
升级为 `L1_COLLECTOR_CONTRACT_FROZEN` 前必须由你裁定：**加载 `0f00af5d…` 对应的已登记合同文件**，
或**正式重新声明**本数据线的 effective contract（并记录可重复计算的哈希与完整配置身份）。

## 6. 运行身份与失败保留

- 一次执行绑定：`run_id 20261015T103956-e9385c6a`、源码 SHA（工具/collector/env/recourse_snapshot/coldchain_contract）、argv、机器状态、stdout/stderr、结果 JSON；JSON 内嵌 sha == 工具文件 sha。
- 本轮**首次运行**（`run_id 20261015T103838-d5f7fffa`）因我的 `canon()` 状态字典缺 `nodes` 键而 `NOT_VERIFIABLE`，
  已按你的要求**保留原件**：`l1_gates/FAILED_A1_D3_L1_RESUME_PARITY_ATTRIBUTION_run20261015T103838.json`。
- 上一轮错配的旧产物仍在 `SUPERSEDED_*`（字节未改）。

## 7. 未做与未授权

服务器正式资源计时（NOT_RUN）、64 槽位采集、F−D-Geo 比较、模型重训练——**均未执行**。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；I3 正式负结果、D2 `not_qualified`、
D3 G0、D8、E2 与 `stable_online_gain=False` 全部不变。

## 8. 建议的下一步顺序（与你的排序一致）

1. **合同身份裁决**（加载 `0f00af5d…` 合同 vs 正式重新声明）→ 闭合 `IDENTITY_*`；
2. 服务器计时资格（独立 run_id、负载/线程/异常、单状态上限覆盖捕获+恢复+校验+完整 parity、增量落盘）；
3. 事前冻结 64 槽位采集协议 → 合格 L1 状态资产；
4. 真实 F−D-Geo 机制诊断（仍需单独授权）。
