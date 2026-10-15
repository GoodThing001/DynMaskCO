# A1-D3-L1 服务器 F-free 计时资格运行（唯一一次授权运行）

- 授权：用户批准的一次限定服务器 F-free 计时资格（预注册预算 + 运行身份先行登记）
- Runner：`scripts/tests/d3_l1_server_timing_qualification.py`（F-free；已上传服务器执行）
- 服务器产物（已取回本地留档）：`l1_gates/A1_D3_L1_SERVER_TIMING_{BUDGET,QUALIFICATION}.json`
  与 `A1_D3_L1_SERVER_TIMING_SLOTS.jsonl`（**逐槽位增量落盘，含 fsync**）
- **修正后结论：`Q_server = PASS`**（条件式；见 §3 的记录器缺陷）

## 1. 预注册（运行前写盘，事后未放宽）

`K=2`、`TOTAL_CAP_S=1800`、`PER_SLOT_CAP_S=600`、调用次数：1 次 rollout + 每槽 1 capture/1 restore/1 hash/1 parity；
`no_retry_loop`、`no_budget_relaxation_after_results` 均写入预算文件。
运行机器：服务器 `user`，**96 CPU**，loadavg `0.19/0.09/0.08`，可见 GPU 两张 **RTX 5090**（利用率 0%），
`CUDA_VISIBLE_DEVICES=''`（F-free 链路不使用加速器，GPU 仅作记录）。
数据：`cal` 实例 0 × R1 前 2 个揭示事件（**计时专用样本，不进入正式 L1 状态集合**）。

## 2. 五门结果（`Q_server = Q_identity ∧ Q_contract ∧ Q_recovery ∧ Q_causal ∧ Q_budget`）

| 门 | 结果 | 证据 |
|---|---|---|
| `Q_identity` | **PASS**（见 §3 更正） | 服务器实算 5 个源码 sha + 数据 sha + profile sha，与冻结合同协议身份**逐项一致**：`coldchain_contract 0fb9e6e8…`、`coldchain_state a2ee98dc…`、`d3_l1_state_collector 849667e4…`、`recourse_snapshot 0af3535a…`、`strict_online_env fd17fe77…`、dataset `c35fa683…`、profile `63e56212…` |
| `Q_contract` | **PASS** | effective `edc1b2ea…4833`；快照 `schema_version=recourse_snapshot_v3_cc`；快照 `coldchain_contract_hash == effective` |
| `Q_recovery` | **PASS** | 两槽 `state_identity=true`（再捕获哈希一致）、`restore_arity=3`、`served_mask_equal=true`、semantic parity `complete=true, diffs=0`、`canon_truncated=false` |
| `Q_causal` | **PASS** | 五项 R1 槽位断言全成立；`restored_horizon=24.0`（原始与恢复环境一致） |
| `Q_budget` | **PASS** | 总墙钟 **2.39 s** ≤1800；单槽 **0.73 / 0.82 s** ≤600；两槽全部落盘 |

分段耗时（服务器实测）：rollout **0.58 s**；capture **2.3 / 2.9 ms**；restore **1.8 / 2.5 ms**；
hash **0.8 / 1.2 ms**；完整 parity（含 resume）**0.67 / 0.75 s**。

## 3. 记录器缺陷（保留原文，不放宽结论）

Runner **记录的** `Q_identity=false` 是**我方的装配缺陷**，不是身份不符：我把冻结身份 JSON
上传到服务器时失败（远端目录不存在 → `FileNotFoundError: [Errno 2] No such file`），
故 runner 读不到该文件、`frozen=None` → 该门在运行时被判 false（`mismatches` 为空即其征兆）。
更正方式：把**服务器实算的 SHA** 取回，与**已提交的冻结身份文件**本地逐项比对 → 全部一致（§2）。
- 原始产物**保留**：`A1_D3_L1_SERVER_TIMING_QUALIFICATION.json` 中 `Q_server=NOT_PASSED` 与
  `gates.Q_identity=false` 原样保留、不改写；本文件给出更正与依据。
- 我的比较脚本另有一处键集不一致（冻结文件含 `this_tool`、服务器含 `dataset/profile`），
  已按字段归类重比，非身份失败。

## 4. 用户指出的两处措辞限制（已记录，留待后续资格工具）

1. **I5 门过宽**：`CROSS_VERSION_OBJECT_INCOMPATIBLE_CONFIRMED` 与 `CLAIM_NOT_CONFIRMED` 都被计为 PASS，
   因此 I5 通过本身不证明不兼容；后续工具须把两态拆开（本批源码证据仍成立：当前 `coldchain_state.py`
   访问旧合同缺失的 `precool_energy_per_unit_per_c`）。
2. **I7 范围有限**：只证明**采集器模块**不直接计算成本，不能保证整条 rollout（环境/基线规划器内部）
   没有成本计算。准确表述：**L1 采集产物不执行 F−D-Geo 效果评价**；采集器及其依赖仍需遵守既定 C0、
   可见信息与成本语义。

## 5. 未做与未授权 / Claim limits

- **未**执行正式状态采集（`dev_check × K=4`、≤64 槽位仍**未授权**）；**未**执行 F−D-Geo 比较；
  **未**修改历史合同、profile、冻结源码或归档报告。
- **数据身份不得自动复用**：本次计时用 `cal` 实例 0；正式采集预登记必须**分别记录**
  `dev_check` 的 `day_id`、位置序号、数据 SHA、资源门、失败分母与快照清单。
- tmux 不被视为持久化；本次采用逐槽位 JSONL + fsync 增量落盘。
- `states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
  `stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部不变。

## 6. 建议下一步

按你的既定顺序：**正式状态采集预登记签署**（`dev_check × K=4`，≤64 槽位；冻结 day_id/位置序号/数据 SHA/
资源门/失败分母/快照清单，并以本次服务器实测成本为资源门依据）→ L1 状态资产 → 真实 F−D-Geo 机制诊断
（仍需各自授权）。
