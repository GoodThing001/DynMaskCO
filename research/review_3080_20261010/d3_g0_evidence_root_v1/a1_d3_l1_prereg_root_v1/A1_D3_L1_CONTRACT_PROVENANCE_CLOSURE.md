# D3-L1 合同来源身份：溯源结果 + A3 判据勘误

- 轮次：用户授权的唯一目标——**合同来源身份闭合**（只读溯源；不改历史合同/冻结树/profile）
- 工具：`scripts/tests/d3_l1_contract_provenance_audit.py`（新增，只读）
- 证据：`l1_gates/A1_D3_L1_CONTRACT_PROVENANCE.{json,_stdout.txt,_stderr.txt}`
- 结果：**`CONTRACT_RECOVERED`**（历史校准合同已独立恢复并哈希验证）；`IDENTITY_PARTIAL` **未**升级为闭合（须由你裁定 L1 采用哪份基础合同）

## 1. 勘误：A3 的判据本身有语义错误（原始 FAIL 保留）

你的纠正成立。A3 比较的是：

- `profile.provenance.contract_hash` = **校准来源合同**（`freeze_coldchain_scale.py:200-201` 载入合同、
  `:251` 把 `contract.contract_hash` 写入 provenance）；
- vs 我实际构造的 **profile 应用后**的生效合同。

这两个是**不同阶段**，定义上可以合法不同，所以 `C_calibration == C_effective_current` 不是身份判据。
`l1_gates/A1_D3_L1_RESUME_PARITY_ATTRIBUTION.json` 中该行的 `FAIL` **原样保留、不覆盖**；
本文件为其勘误与正确判据。正确判据（已机械记录在审计 JSON 的 `corrected_criterion`）：

1. 校准来源合同必须**可独立恢复且哈希可验证**；
2. profile 应用关系必须**可重算**：`C_base + profile → C_effective`（记录哈希）；
3. L1 协议必须**点名**它使用哪份基础合同，不得把"校准来源"与"当前默认"静默混用。

## 2. 溯源结果：历史校准合同 `0f00af5d…` **确实找到并验证**

恢复路径（R2）：provenance 记录校准期源码 `coldchain_contract.py` 的 SHA = `abf97d97…`；
按该 SHA 在归档中检索到**三份**完全一致的归档源码副本：

| 归档路径（相对扩展目录） | 隔离导入后 `default_pilot_contract()` 哈希 |
|---|---|
| `archive/正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4/source/.../coldchain_contract.py` | **`0f00af5d…`（匹配）** |
| `archive/诊断与作废冻结包/frozen_version_v1/source/.../coldchain_contract.py` | **`0f00af5d…`（匹配）** |
| `archive/诊断与作废冻结包/o0cc_devgate_2dfff7ae_20260908/source/.../coldchain_contract.py` | **`0f00af5d…`（匹配）** |

即：**从冻结归档源码重建的合同哈希 == profile 声明的校准合同哈希**（逐位一致），
且 `0f00af5d…` 亦出现在 `results/o0cc/scale_v2/manifest.json`、`service_audit/manifest.json`
与 DEV-GATE v4 九个 cell 的 `summary.json` 中（作为该线的合同身份）。

（首次尝试因未把模块注册进 `sys.modules` 而 `AttributeError`（dataclass 解析 `cls.__module__` 依赖它），
已修正；失败痕迹保留在 stderr 与提交历史中。）

## 3. 三个身份与完整链条（同一 profile 下）

| 身份 | 哈希 | 说明 |
|---|---|---|
| `C_calibration` | **`0f00af5d197cf35368b1b2e807bcaf53d9b58a1cd425a7f53ec66091f6acce28`** | 已从归档源码恢复并验证 |
| `C_base_current` | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0` | 当前源码 `0fb9e6e8fcff11c5…`（≠ 校准期 `abf97d97…`，版本身份差异确认） |
| `C_effective(C_base_current)` | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` | = 上一轮实际生效合同 |
| `C_effective(C_calibration)` | `67e0fc7bab65e19d6c5c56bd0b3d6f4a173961a3e0cae9b9cb3a3487ae5ec607` | 校准合同 + 同一 profile |
| `profile_hash` | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` | 与声明一致 |

**`effective_hashes_equal = false`** → 基础合同的选择**会改变生效目标函数**，不是无关紧要的记账差异。

## 4. 两份基础合同的差异：恰好 8 个字段，全在 `thermal`

逐字段对照（`C_calibration` → `C_base_current`）**仅 8 处**，且全部是当前合同**新增**的字段：

`thermal.dispatch_delay_hour`、`thermal.diurnal_amplitude_c`、`thermal.diurnal_phase_hour`、
`thermal.precool_energy_per_unit_per_c`、`thermal.precool_min_temp_class`、
`thermal.precool_station_discount`、`thermal.precooled`、`thermal.use_station`。

这正是项目状态文档记载的 **2026-09-23 研究扩展字段**（diurnal 与 precooling），文档亦明确它们**会改变
`contract_hash`**。物理约束、单位、品质配置与 objective 其余部分**逐字段相同**。

## 5. 我的建议（待你裁定）

1. **优先采用历史校准合同路线**（你此前的建议一致）：L1 基础合同取 `0f00af5d…`，
   其身份由**已归档冻结源码**（上表任一，建议取"正式冻结包"那一份并记录其文件 SHA）重建而来，
   而不是猜测配置；
2. 若采用该路线，L1 身份清单必须写明：`contract_reconstruction_route = archived_frozen_source`、
   归档文件路径 + SHA、重建命令，以及 `C_calibration → profile → C_effective(67e0fc7b…)` 链条；
3. 若你选择继续使用当前默认合同，则必须**明确标记为新的协议身份**，不得声称与历史 DEV-CAL /
   旧正式实验天然可比（二者生效哈希不同：`edc1b2ea…` vs `67e0fc7b…`）；
4. 在裁定前维持 **`L1_COLDCHAIN_CONTRACT_CANDIDATE / IDENTITY_PARTIAL`**，不升级为
   `L1_COLLECTOR_CONTRACT_FROZEN`，也不升级为 `IDENTITY_CLOSED`。

## 6. P1b 的覆盖边界（按你的裁定保留）

接受本轮两槽位的语义恢复结论，归档 `FULL_PARITY` 标签**保留不变**；同时明确记录覆盖边界：
`canon()` 归一化 NumPy 标量，跨类型数值比较对正负零位模式有例外，因此"IEEE-754 位模式一致"**不**等于
所有浮点叶节点均已逐位验证。正负零/NaN/NumPy 标量/原始对象类型的**最小反例**将加入后续资格审计工具，
**不作为**推进主模型研究的长期阻断，本轮不重跑大测试。

## 7. 未做与未授权 / Claim limits

未执行：服务器正式资源门、64 槽位采集、F−D-Geo 比较、模型重训练；未修改历史合同、冻结树或 profile。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部原样保留。
本回复不等于合同身份闭合：**闭合由你的裁定完成**。
