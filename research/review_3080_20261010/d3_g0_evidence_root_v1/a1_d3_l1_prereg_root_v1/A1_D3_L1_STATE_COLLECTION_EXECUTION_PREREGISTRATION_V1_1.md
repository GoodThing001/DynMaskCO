# A1_D3_L1_STATE_COLLECTION_EXECUTION_PREREGISTRATION_V1.1

- 本文件是 `..._V1.md`（提交 `5671af6`）的**最小修订**；**V1 原件不变**，本文件仅关闭三项执行层缺口
- 状态：`V1_1_DRAFT_FOR_SIGNATURE`；`EXECUTION_SIGNATURE_PENDING`
- 实际采集授权：**`NOT_AUTHORIZED`**
- 证据：`l1_gates/A1_D3_L1_COLLECTION_READINESS.json`（依赖闭包 + 比较器最小反例）

## 0. 审查裁决（A1-D3-L1 正式状态采集预登记审查裁决）

- 审查对象：`A1_D3_L1_STATE_COLLECTION_EXECUTION_PREREGISTRATION_V1.md`
- 提交身份：`5671af6313020ae95a7432868eaf426c5ae2ac54`
- 裁决：**`METHOD_APPROVED / EXECUTION_SIGNATURE_PENDING`**；实际采集授权：`NOT_AUTHORIZED`
- **接受并固定**：`dev_check` 全部 16 个开发日 × R1 前 4 个真实揭示事件、64 个预登记槽位、
  L1 effective contract `edc1b2ea…4833`、既有数据身份（`66573ddc…` / `b84e1716…`，执行前复核原件）、
  120 s/日、60 s/槽、3600 s/批预算、既定失败语义。
- **签署前必须完成（本文件 P0-1/2/3）**：因果反事实投影的构造与调用次数；计时跨度/独占计算时间/硬截止；
  运行源码依赖身份闭合 + 快照原子持久化 + 失败槽位不可重采。
- 本修订仅产生 V1.1 与新增审计工具，**不得**修改已登记目标、数据集合、合同、预算数值或原始运行证据。

## 1. P0-1 因果反事实投影（**不额外运行 rollout**）

**同一快照上的 F-free 反事实投影**，每槽**固定两次投影**（不新增 rollout、不新增 baseline 运行）：

| 项 | 规定 |
|---|---|
| 调用次数（每槽） | **2 次投影**：`project(original)`、`project(perturbed)`；加上 1 次 capture、1 次 restore、2 次 hash、1 次 parity —— 全部逐项计时登记 |
| 投影对象 | 槽位快照解码后的**隔离副本**（不触碰原快照、不写回、不包括环境对象） |
| 扰动族（**仅此一族**） | 仅修改**当前决策时刻不可见**订单的属性（坐标/需求/温度类），**保持**：订单 ID 集合、数组维度、已揭示集合、已执行/已承诺状态不变 |
| 必须核实 | `n_invisible_modified > 0`；若为 0 → 该槽因果结果为 **`no_valid_perturbation`**，**不得**报因果 PASS |
| 比较内容 | 两次投影的**决策可见特征**、**合法候选集**（顺序与内容）、**基线 planned suffix** 的完整内容与 **SHA256** |
| 判定 | 三者 SHA 全部相同 → 该槽 `causal_projection = PASS`；任一不同 → `causal_violation`，并输出**首个差异路径+值** |
| 结论范围 | **仅限本预声明扰动族**，不构成"无任何未来信息泄漏"的数学证明 |

## 2. P0-2 计时定义与硬截止（预算数值不变）

| 字段 | 定义 | 承载上限 |
|---|---|---|
| `slot_span_wall_s` | 该槽 **capture 调用开始** → 该槽全部验证（含两次投影）结束的**端到端跨度**（含等待/交叉作业） | **60 s** |
| `slot_active_compute_s` | 该槽 capture + restore + hash×2 + parity + 两次投影的**实际工作耗时之和**（独占计算成本，用于报告，不与 60 s 混用） | — |
| `day_rollout_wall_s` | 当日 baseline rollout 完整墙钟 | **120 s/日** |
| `batch_wall_s` | 全批次自启动至最终持久化结束 | **3600 s** |

**硬截止机制**：由**独立监督进程（watchdog/supervisor）**执行上限判定；到期限即终止 worker 并由
**监督进程**持久化 `timeout`、停止位置与未完成槽位清单。**不得**仅在阻塞函数返回后比较耗时，
**不得**依赖 worker 内部可能被 `except Exception` 吞掉的超时异常。
超时槽位记 `timeout` 并**保留在分母**；不放宽上限、不重跑。

## 3. P0-3 依赖身份闭包 + 快照原子持久化

### 3.1 依赖闭包（本次只读实算，11 个内部模块）

`d3_l1_state_collector`（种子）、`d3_l1_server_timing_qualification`（种子）、
`test_d3_l1_resume_parity_attribution`（种子）、`jf1h_repair` `233f8439…`、**`joint_fleet` `7dafcd6d…`**、
**`action_contract` `9d64273a…`**、`strict_online_env` `fd17fe77…`、`recourse_snapshot` `0af3535a…`、
`coldchain_contract` `0fb9e6e8…`、`coldchain_state` `a2ee98dc…`、`cvrptw_utils` `1fd6aa42…`
（每个模块的 sha256 与 **import 边**全部记录在 `A1_D3_L1_COLLECTION_READINESS.json`）。
用户点名的 `joint_fleet`、`action_contract` **已在闭包内**；且 `joint_fleet 7dafcd6d…` 与 profile
`provenance.code_sha256.joint_fleet` 一致（交叉佐证）。
**不在本路径**（按设计排除，已在审计 JSON 中记录原因）：`cc_lns_replanner`（LNS 参考线）、
`dynmaskco_cc_context`（模型/教师上下文）、`visible_state`（LNS 可见场景评估器）、
`repair_state`（F 侧特征提取器，F-free 要求排除）。

### 3.2 比较器（parity 判据，已由最小反例资格化）

`parity` 使用**逐位比较器**：每个浮点叶节点要求 **IEEE-754 位模式完全一致**（`struct.pack('>d')`），
类型必须完全相同，容器结构逐层比较。最小反例实测**全部符合预期**：
`+0.0 vs −0.0` 判为**差异**、`NaN` 按位比较（同位即相等）、NumPy 标量与 Python 数按位比较、
`int vs float` 判为类型差异、深层嵌套浮点位差被捕获。
先前"规范化 + 数值容差"的比较器**降级为诊断工具**，**不得**用作采集判据。

### 3.3 快照原子持久化与中断规则

1. 快照先写**临时文件**→ `flush + fsync` → **原子发布**（`os.replace`）为正式快照文件；
2. 随后追加 manifest 记录：`snapshot_path`、`bytes`、`sha256`、`state_hash`、`slot status`；
3. 运行结束后由**独立只读程序**复核 manifest ↔ 快照一致性（计数、SHA、state_hash）；
4. 中断导致 manifest 与快照不一致 → **留下失败/未完成记录**，**不得**默认视为恢复成功；
5. **恢复边界**：允许以**新 `run_id`** 继续**尚未尝试**的槽位；
   **已失败槽位不得重新采集**（一次采集 pass 的上限仍然有效）；两 run 记录**同时保留**。

## 4. 未修改项（保持 V1）

`dev_check` 16 天 / R1 / K=4 / 64 预算槽位；数据与合同 SHA 身份；失败槽位全部保留、不筛选不替换；
`comparator_not_qualified` 作为运行前阻断；strict-online 禁止未来订单进入决策信息；
F-free 采集不参与模型效果评价；跨版本 profile 仅用于 L1 诊断的既定限制。
比较器最小反例已由轻量测试完成（`d3_l1_collection_readiness_audit.py`），**不新增**服务器计时实验。

## 5. 仍未授权 / Claim limits

实际采集、F−D-Geo 机制比较、L2/L3、模型重训练**均未授权**；本文件不含任何新实验杠杆，
也不构成对 `cal` 两槽诊断结论的追认。`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；
D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified`
全部不变。V1.1 通过独立只读审查与运行前最小回归后方可提交最终签署。
