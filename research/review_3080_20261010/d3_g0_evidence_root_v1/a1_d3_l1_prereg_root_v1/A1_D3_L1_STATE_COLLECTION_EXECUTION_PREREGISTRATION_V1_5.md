# A1_D3_L1_..._V1.5：P0-C 正式采集工程交付 + 两点审查修正

- 追加记录，**V1–V1.4 与既有产物全部不改写**
- 交付物：`scripts/simulation/d3_l1_collection_runner.py`（worker + supervisor + closure 三入口）、
  `l1_gates/A1_D3_L1_P0C_ACCEPTANCE.json`（7 项阴性反例）、
  `l1_gates/A1_D3_L1_P0C_IDENTITY_CLOSURE.json`（真实入口闭包）
- 本轮**未**运行 16 日受监督采集（未授权）；**未**调用 F/D-Geo；**未**修改冻结资产

## 1. 你的两点审查修正：均已实现并有阴性反例

### 1.1 P0-B 样本来源更正（按证据 JSON 描述）

| 用途 | 样本 |
|---|---|
| 服务器**计时**资格（预算设计用） | `cal` 实例 0 × R1 前 2 个揭示事件（2 槽） |
| **P0-B 正式登记结果**（V1.4） | **`dev_check` 第 0 日第 1 槽**，并核查**全部 16 个开发日**的距离生成合同；证据 JSON 记录 `dataset_sha256 = 66573ddc…39c89`、`dataset_is_the_registered_dev_check = true` |

今后报告按证据 JSON 描述，**不再把两者混用**（我的交接文字曾把 P0-B 写成 `cal` 实例 0，属描述错误，已更正）。

### 1.2 集成断言补强（新增两条阴性反例）

- **三 SHA 必须显式全相等**：`compare_projections()` 对 `visible_inputs_sha` / `legal_candidates_sha` /
  `planned_suffix_sha` 逐项判定，**任一不等即失败**，**且存在 `errors` 键也失败**（即使三 SHA 相等）
  —— 不再使用 `bool(dict)` 这类真值捷径。反例 **N3 PASS**。
- **后缀身份含车辆边界**：`suffix_identity()` 以 `(vehicle_id, tuple(suffix))` 排序后取 SHA，
  **展平相同但车辆归属不同也会被识别**。反例 **N4 PASS**（展平序列相同，SHA 不同）。

## 2. 交付内容（P0-C 四项）

### 2.1 正式 worker（`--role worker`）

按 R1 逐日采集：`ATTEMPT_STARTED` 在 **capture 之前**先落盘 → capture → `state_hash` 两次 →
restore（记录 arity）→ **两次独立因果投影**（原始 / 扰动 dataset，各 1 次 replanner 调用）→
三 SHA 显式比较 → **原子发布快照**（`temp → fsync → os.replace`）→ manifest 记录
（路径/字节数/SHA/`state_hash`/槽位身份/状态）。任一失败即记 `FAILED` 与原因（`causal_violation`
单独分类），**不重采**。全程 `assert_f_free()`。

### 2.2 独立 supervisor（`--role supervisor`）

- **运行前硬校验身份**：冻结身份文件缺失/不可读/任一源码或数据 SHA 不符 → **`HARD_STOP_IDENTITY`**，
  不启动任何 worker（反例 **N1/N2 PASS**）；
- 每日一个 worker **子进程**，用**独立时钟**执行 `120 / 60 / 3600` 秒硬截止；
- **超时由 supervisor 自己记账**：`DAY_DEADLINE_KILLED` + 未到达槽位记
  `NOT_ATTEMPTED_DUE_TO_TERMINATION`、进行中槽位记 `FAILED(terminated_mid_slot)`，
  **不依赖 worker 的异常处理或正常退出**；
- 每日结果写 `WORKER_DAY%02d.json`，批次结果写 `SUPERVISOR_RESULT.json`（含 `slots_published`、
  固定分母 64、`batch_wall_s`）。

### 2.3 一次性 pass 与失败不可重采

`may_attempt()` 只允许**从未出现过任何记录**的槽位：一旦该槽出现
`ATTEMPT_STARTED`/`PUBLISHED`/`FAILED` 即**拒绝再次尝试**（反例 **N5 PASS**）；
已尝试槽位记 `SKIPPED_ALREADY_ATTEMPTED`。新 `run_id` **不覆盖**旧证据。

### 2.4 原子持久化与失败账本

快照与状态文件均 `temp + fsync + os.replace`（反例 **N6 PASS**：无残留 `.tmp`、SHA 可核）；
`STATE_LEDGER.jsonl` 增量追加并 `fsync`；**损坏行以 `corrupt_line` 显式暴露**（反例 **N7 PASS**），
不静默丢弃。

### 2.5 真实入口源码身份闭包（`--role closure`）

以**真实入口**（runner/worker/supervisor、因果投影、逐位比较器、采集器）重生成闭包：
**13 个内部模块**，含 `dynmaskco_cc_context`、`joint_fleet`、`action_contract`、`strict_online_env`、
`recourse_snapshot`、`coldchain_contract`、`coldchain_state`、`cvrptw_utils`、
`d3_l1_bitwise_parity`、`d3_l1_causal_perturbation`、`d3_l1_collection_runner`、`d3_l1_state_collector`、
`jf1h_repair`；逐文件 SHA256 与 import 边、入口 SHA（runner `e65e61eb28d69392…`）全部落盘。
**旧 11 模块清单不再作为"已通过"的依据**（V1.2 §3 的作废声明继续有效）。

## 3. 预先冻结的"日被强制终止"处理规则（按你的要求，签署前定）

1. 该日**未到达**的槽位 → `NOT_ATTEMPTED_DUE_TO_TERMINATION`，**留在 64 分母**，**该日不重跑**；
2. 该日**进行中**的槽位 → `FAILED(terminated_mid_slot)`，**不得重采**（一次 pass 上限仍然有效）；
3. 批次**继续**下一天（新的一天属于"尚未尝试"）；
4. 若整批触达 3600 s → 记 `batch deadline_killed` 并停止，未尝试槽位同样按上述规则登记；
5. 所有上述记录由 **supervisor** 写入，worker 被 kill 后不再承担记账责任。

## 4. 验收状态

| 项 | 结果 |
|---|---|
| 阴性反例（N1–N7） | **7/7 PASS** |
| 真实入口闭包 | **13 模块**，逐文件 SHA + 入口 SHA 落盘 |
| worker / supervisor / 原子持久化 / 失败账本 / 一次性 pass | 已实现并通过反例 |
| 16 日受监督正式采集 | **未运行（未授权）** |
| F−D-Geo 比较 / 训练 / L2 / L3 | **未授权** |

## 5. 状态与 Claim limits

```
P0_A = CLOSED_ON_COUNTEREXAMPLE_GATE
P0_B = PASSED_ON_PLAN_B_GATE
P0_C = IMPLEMENTED_AND_ACCEPTED_ON_SYNTHETIC_COUNTEREXAMPLES   # 正式采集仍未授权
FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED
```
未采集任何正式状态（`states_collected_for_L1 = 0`）；未重跑旧诊断样本与服务器计时样本；
`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、
I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
P0-C 通过**不**解除 G0/D8/E2，也不追认任何 A1/A2/A3 增益层。
