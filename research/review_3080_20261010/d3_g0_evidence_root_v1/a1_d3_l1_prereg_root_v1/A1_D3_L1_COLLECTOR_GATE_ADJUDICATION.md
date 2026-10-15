# A1-D3-L1 采集器与恢复门裁决（F-free 采集器 + 单次计时资格运行）

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（用户授权的采集器实现 + 单次计时资格运行）
- 证据：`l1_gates/A1_D3_L1_COLLECTOR_GATES.json`(+stdout/stderr)
- 采集器：`scripts/simulation/d3_l1_state_collector.py`；门套件：`scripts/tests/test_d3_l1_collector_gates.py`
- 结果：**14 行 / 13 PASS / 1 FAIL**；四项状态**分别**报告，**不**升级 L1 执行资格
- **未**执行 64 槽位正式采集，**未**做 F−D-Geo 比较，**未**改动任何冻结源码

## 1. 用户点出的两个接口约束：均已实测确认并内建

**① `reveal_idx` 不能单独判"这是揭示事件"——已用实测反例证实**

本次采样的第 1 个槽位：`event_id=1`，**`reveal_idx=0`**（正是用户警告的情形：t=0 与首次真实揭示都可能带 `reveal_idx=0`）。采集器改用**可见集合增长**判别并交叉核对 `sorted(set(reveal_time))`，实测五项断言全部成立：

| 断言 | 结果 |
|---|---|
| t=0 不占槽位 | true |
| 非揭示物理事件不占槽位 | true |
| 同一揭示时刻只占一个槽位 | true |
| 每个槽位恰好采集一次 | true |
| 槽位与声明的揭示时刻一致（前缀） | true |

采集到的槽位：`(slot1, clock=0.14656, event_id=1, reveal_idx=0)`、`(slot2, clock=0.49480, event_id=2, reveal_idx=1)`。

**② `resume_from_snapshot` 丢失 `booking_horizon`——已运行时确认**

对每个槽位直接调用冻结辅助函数并测量：`frozen_helper_booking_horizon = null`（**全部**槽位）。
采集器**不改冻结源码**，而是自行以完全相同显式配置构造恢复环境后再调用 `env.run_resumed(snapshot)`：
`original_env.booking_horizon == restored_env.booking_horizon == 24.0` → `horizon_chain_ok = true`（全部槽位）。

## 2. 四项状态

| 项 | 状态 | 依据 |
|---|---|---|
| `IDENTITY_CLOSED` | **PARTIAL（阻断项仍在）** | 配置指纹 `78129840…` 已记录（data/profile/collector/strict_online_env/jf1h_repair/recourse_snapshot 的 sha256）；**contract 未提供**（`coldchain_contract_sha256 = null`）；`booking_horizon` **不在** profile/contract 中，只能作为协议常数冻结 → 用户指出的身份缺口**被显式记录而非关闭** |
| `COLLECTOR_CAUSALITY` | **PASS（单实例滚动级）** | 完整 rollout 扰动（删除首个揭示时刻仍不可见的全部订单、已揭示前缀逐位不变）→ 该揭示时刻**及之前**的所有决策指纹**完全一致**；`horizon=None` 反例探针在该实例上亦一致（已记录：本日内泄漏未显形，决定性证据仍是上一轮的接口级翻转） |
| `SNAPSHOT_RECOVERY` | **3/4 PASS + 1 项我的断言缺陷（未收束）** | `hash_stable=true`（含重复计算）；sanctioned horizon 链成立；`run_resumed` 在每个槽位成功执行（返回 `(list, ndarray)`）。**FAIL 项**：我断言 `restore_recourse_snapshot` 返回 `dict`，实测**不是** dict（`restore_returns_dict=false`）。这是**我的测试断言缺陷**（未先核验 API 返回类型），不是实现缺陷；本次未修好，**不得**据此宣告 SNAPSHOT_RECOVERY 通过 |
| `COLLECTION_TIMING` | **MEASURED（单次授权样本，本机）** | baseline rollout **0.0619 s**；capture **≈1.0 ms/槽**；hash **<1 ms**；restore **≈1.0/0.0 ms**；resume-parity **49.0/41.0 ms**；总墙钟 **0.6 s**；上限在运行前登记（K=2、总 ≤900 s、单状态 ≤300 s）且全部达标；`T2` 证明本进程**从未**导入任何 F/D-Geo 模块 |

## 3. 必须保留的限制

1. 计时是**本机**（Windows，无 GPU：`nvidia-smi` 不可用）测得，**不是**共享服务器上的正式资源门；
   正式门必须在目标机器上重测并记录机器负载。本文件**不使用**未经测量的耗时数值。
2. 因果性在**一个实例**上完成滚动级验证；接口级翻转证据来自上一轮，二者不可互相替代。
3. `IDENTITY_CLOSED` 未闭合：缺 contract 身份 + `booking_horizon` 未进入任何既有合同/画像。
4. `SNAPSHOT_RECOVERY` 有一项未通过（见上），故本批**不**申请任何 L1 执行资格。
5. 采集到的 2 个槽位样本**仅用于计时与门测试**，`states_collected_for_L1 = 0`。

## 4. 我自己的缺陷（保留原文，不改写为通过）

1. `sha256_file(None)` 未做空值保护 → 崩溃（已修）。
2. rollout 入口名发现表过窄（已扩充）。
3. R3 断言假定 `restore_recourse_snapshot` 返回 `dict`（未先核验）→ 唯一 FAIL；两次修补尝试中一次因 PowerShell 转义失败而未生效（`SyntaxError: unexpected character after line continuation`），已如实记录、未伪造通过。

## 5. 下一步（待授权）

1. 修好 R3 断言（记录真实返回类型而非假定 dict），重跑门套件；
2. 提供并冻结 contract 身份，把 `booking_horizon=24.0` 写入 L1 身份清单；
3. 在**服务器**上重测计时（记录机器负载/异常）并提交正式资源门；
4. 之后才申请 dev_check 64 槽位正式采集；F−D-Geo L1 比较仍未授权。

## 6. Claim limits

工程资格，不是模型增益。`algorithmic_gain_claim: none`。I3 正式负结果、D2 `not_qualified`、
D3 G0、D8、E2 与 `stable_online_gain=False` 全部不变。
