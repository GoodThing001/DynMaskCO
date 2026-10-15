# A1_D3_L1_STATE_COLLECTION_EXECUTION_PREREGISTRATION_V1.2（P0-A/B/C 追加记录）

- 本文件是 V1 / V1.1 的**追加记录**；**V1、V1.1 原件与既有产物全部不改写**
- 触发：用户对 `788af74` 的裁决——`METHOD_APPROVED / EXECUTION_PREREG_V1_1 = REVISION_REQUIRED`，
  `BLOCKERS = [BITWISE_COMPARATOR_FALSE_NEGATIVE, PERTURBATION_UNDERSPECIFIED, FINAL_RUNNER_IDENTITY_PENDING]`
- 本轮状态：**1 项阻断已闭合（P0-A）**；**2 项仍未闭合**（P0-B 规格已纠正但门未通过；P0-C 未实现）
- `FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED`；`states_collected_for_L1 = 0`

---

## 1. P0-A 逐位比较器假阴性：**已修，且已资格化**

你指出的假阴性**成立**：旧 `bits()` 对任何带 `dtype` 的对象调 `float(x)`，对**数组**会失败并返回
`None`，两侧同为 `None` 即被判定相等；且 NumPy 原始类型/dtype 与 `list` vs `tuple` 未区分。
旧产物 `l1_gates/A1_D3_L1_COLLECTION_READINESS.json` 的 `2/2 PASS` **原样保留**，其
`bitwise_comparator_qualified=true` 已按你的要求追加为
**`BITWISE_COMPARATOR_QUALIFICATION = NOT_ESTABLISHED`（已被新工具取代）**。

**新工具**：`scripts/tests/d3_l1_bitwise_parity.py`（+ 证据 `l1_gates/A1_D3_L1_BITWISE_PARITY_GATE.json`）
实现你指定的规则：

- `ndarray`：原始类型 + `dtype.str` + `shape` + `tobytes()` 必须全部一致；
- NumPy 标量：原始类型 + `dtype.str` + **原始位模式**；
- Python `float`：IEEE-754 double 位模式（`+0.0` vs `-0.0` 判为**差异**）；
- `NaN`：**按位**比较（同 payload 相等；不同 payload 判差异）；
- `dict`/`list`/`tuple`/dataclass：**先比容器类型**，再逐成员比较，不做任何规范化；
- 无法比较或节点预算耗尽 → **`NOT_VERIFIABLE`**，**不得**判相等。

**最小反例门结果：16/16 全部符合预期**，其中包含你给的四例：
`np.array([1.,2.]) vs np.array([1.,3.])` → DIFFERENT；
`np.float32(1.0) vs np.float64(1.0)` → DIFFERENT；
`np.int64(1) vs np.float64(1)` → DIFFERENT；
`[1,2] vs (1,2)` → DIFFERENT；
另含 `array_dtype_differs`、`array_shape_differs`、`signed_zero_differs`、
`nan_same_payload_equal`、`nan_different_payload_differs`、`np_scalar_vs_python_float` → DIFFERENT 等。
⇒ **`BITWISE_COMPARATOR_QUALIFICATION = ESTABLISHED`**。
（前一轮把 `np.float64(1.0) vs 1.0` 预期为"相等"的写法是错的，与本文件"类型必须完全相同"一致，
已改为 DIFFERENT。）

## 2. P0-B 因果扰动：**规格已纠正，但门仍未通过（不得据现有结果宣称因果 PASS）**

### 2.1 本轮实测发现（fail-closed 暴露出的真实问题）

在真实快照上执行扰动纯函数 → **`NOT_VERIFIABLE`、`n_invisible_modified=0`、`fields_perturbed=[]`**。
原因已查明：快照 **53 个键** 只含**规划器/车辆/车队/冷链**状态
（`customer_universe`、`served_mask`、`visible_mask`、`vehicle_*`、`services_*`、`coldchain_*`、
`trace_*`、`replanner_state`…），**不含 `coords` / `demands` / `tw_*` / `reveal_time` 等实例数组**。
因此"在快照内改不可见订单属性"的设计**不成立**——订单属性在 **dataset**，不在快照。
该 FAIL 与其原因已随产物一并保留，未改写为通过。

### 2.2 纠正后的纯函数规格（**待实现、待过门**）

```
perturb(dataset, snapshot, day_id, position) -> (perturbed_dataset, report)     # PURE
```

- **不可见订单选择规则**：`c ∈ snapshot['customer_universe']` 且 `snapshot['visible_mask'][c] == False`
  （`visible_mask` 由既有 `validate_snapshot_against_env` 与 `reveal_time/clock` 交叉校验）；
- **确定性映射**：对每个不可见 `c`（升序），以
  `u = int(sha256('d3l1-perturb|day_id|position|c|field')[:8], 'big') / 2^64 ∈ [0,1)` 驱动：
  `coords += u·1.0`；`demands = max(demands·(1+u), 1e-6)`；`tw_start/tw_end += u·0.5`；
  `service_time *= (1+u)`；`temp_class`（分类型）**不扰动**并在报告中标注；
- **字段保留规则**：订单 ID 集合、数组维度/dtype、`reveal_time`（⇒ 已揭示集合与时间表不变）、
  `served_mask`、`clock`、已执行/已承诺状态、`replanner_state` 一律不变；
- **`n_invisible_modified` 计算方式**：实际被改动的不可见订单数（若为 0 → `no_valid_perturbation`，
  该槽**不得**报因果 PASS）；
- **独立重建要求**：三项输出（决策可见特征、合法候选集、基线 planned suffix）
  必须**分别从"原始 dataset + 快照"与"扰动 dataset + 快照"重新构造**，
  **不得复用**任何缓存的特征或候选结果；若 planned suffix 需重新调用 JF1-H-F，
  该调用必须登记在**两次投影的调用预算**内，不得隐含额外 rollout；
- **结论边界**：仅对**本预声明扰动族**成立，不构成"无未来信息泄漏"的数学证明（保留 V1.1 表述）。

**当前状态**：`PERTURBATION_PURE_FUNCTION = SPECIFIED_NOT_IMPLEMENTED`；
`PERTURBATION_GATE = NOT_PASSED`（本轮 FAIL 已归档）。

## 3. P0-C 正式执行源码闭包：**未完成**

现有 11 模块闭包（含 `joint_fleet 7dafcd6d…`、`action_contract 9d64273a…`）**仅以旧工具为入口**：
`d3_l1_state_collector`、`d3_l1_server_timing_qualification`、`test_d3_l1_resume_parity_attribution`。
你指出的两点成立：① 该闭包**未覆盖**尚未交付的正式 worker / 独立 supervisor / 因果投影 / 快照持久化程序；
② 审计里 `formal_runner_present=true` 只检查了**旧计时程序是否存在**，不能证明正式 runner 就绪 ——
该措辞已作废，本文件明确：
**`FORMAL_RUNNER_IDENTITY = PENDING`**、`FINAL_RUNNER_IDENTITY_PENDING = true`。

**最小修法（已定，待执行）**：实现正式 **worker** 与独立 **supervisor** 后，以其**真实入口**
重新生成一次源码闭包，记录实际 `import origin`、逐文件 SHA 与**启动/结束身份**；
既有 11 模块作为基础身份清单复用，不重做历史证据。

## 4. 无需再改的部分（保持 V1.1）

120 s/日、60 s/槽（`slot_span_wall_s` 承载）、3600 s/批（`batch_wall_s`）三硬截止与
独立 supervisor 机制；`slot_active_compute_s` 仅作报告；快照原子发布（temp → fsync → `os.replace`）
+ manifest（路径/字节数/SHA/state_hash/槽位状态）+ 运行后独立只读复核；
JSONL 增量落盘；**已失败槽位不得重采**、新 `run_id` 只能续跑尚未尝试的槽位；
`recorded_Q_server=NOT_PASSED` 与 `posthoc_identity_verification=MATCH` 的分账；
16 天 / R1 / K=4 / 64 槽位 / 合同与数据身份 / 失败语义 —— **一律不改**。

## 5. 阻断清单与下一步（按你的顺序）

```
BLOCKERS = [BITWISE_COMPARATOR_FALSE_NEGATIVE(CLOSED),
            PERTURBATION_UNDERSPECIFIED(SPEC_CORRECTED_GATE_NOT_PASSED),
            FINAL_RUNNER_IDENTITY_PENDING(OPEN)]
```

执行顺序：**① 比较器反例门（已闭合）→ ② 实现并过门因果扰动纯函数 → ③ 实现正式 worker/supervisor
并以真实入口重生成源码闭包**；三项通过后方可提交最终签署版。实际采集仍由你单独决定。

## 6. Claim limits

本轮不含任何科学结论或新实验：未采集状态、未调用 F/D-Geo、未改历史合同/profile/冻结源码/归档报告；
未重跑服务器计时样本。`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；
D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified`
全部不变。
