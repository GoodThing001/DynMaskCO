# A1_D3_L1_..._V1.3 追加记录：P0-B 被 `dist_mat` 合同阻断（fail-closed），三个候选修法

- 追加记录，**V1/V1.1/V1.2 与既有产物全部不改写**
- 证据：`scripts/simulation/d3_l1_causal_perturbation.py`、
  `l1_gates/A1_D3_L1_P0B_CAUSAL_PERTURBATION.{json,_stdout.txt,_stderr.txt}`
- 本轮结果：**5 行 / 2 PASS / 3 FAIL**；`P0_B = BLOCKED_BY_DIST_MAT_CONTRACT`；`FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED`

## 1. 你指出的字段级寻址：**已修正并生效**

新的 `perturb(dataset, snapshot, day_id, position)` 按字段 schema 显式寻址
（`coords[day, c, :]`、`demands/tw_start/tw_end/service_time[day, c]`），**不再使用 `arr.shape[-1]`**。
实测（cal 实例 0、槽位 1）：

- 正确识别**不可见客户 24 个**；
- 实际发生**位级变化**的字段：`coords`、`demands`、`tw_start`、`tw_end`、`service_time`；
- `n_invisible_modified` **按实际字节变化计数**（不是赋值次数）；
- 两次相同扰动输出**逐位一致**，输入 dataset/snapshot **未被修改**（B1、B2 PASS）。
- `temp_class`（分类型）**不扰动**并在报告中标注。

## 2. 你指出的第二个风险：**实测成立，且阻断当前扰动族**

对真实数据检查派生距离一致性：

| 检查 | 结果 |
|---|---|
| dataset 是否含 `dist_mat` | **含**，形状 `[16, 51, 51]` |
| `dist_mat` 是否等于 `coords` 的欧氏距离 | **否**（`equals_euclidean_of_coords = false`） |
| 我的处理 | **`NOT_VERIFIABLE`**，并给出原因，**未**用欧氏距离冒充该矩阵 |

因此 B3/B4/B5 FAIL：不产生"扰动后的反事实数据"，也就无法做决策投影比较。
按你的规则（"若无法从坐标合法重建，必须采用预先声明的合法扰动方式，不能假装欧氏≈路网"），
当前扰动族**必须重新设计**后才能具备检测能力。

## 3. 三个候选修法（**需你择一批准后我实现并重跑门**）

| 方案 | 内容 | 优点 | 风险/代价 |
|---|---|---|---|
| **A（推荐）扰动真正的决策输入 `dist_mat`** | 在扰动副本中只改**涉及不可见客户**的距离项（按预声明规则，如 `d' = d·(1+u)`，`u` 由既有 hash 规则导出、限定合法范围），**可见×可见项必须逐位不变**；`demands/tw/service_time` 照旧扰动；**coords 不扰动**（避免与 dist_mat 的合同冲突） | 扰动的正是 JF1-H-F 实际评分用的输入（`joint_fleet` 直接读 `env.dist_mat`），检测能力最强；无"合同未知"假设 | 需明确 coords 不再是扰动族成员（其"物理属性"角色本轮不参与因果检验） |
| **B 溯源 dist_mat 生成合同** | 查明 frozen 数据里 `dist_mat` 的构造（生成器/预处理，可能含 `distance_scale 18.33` 或合成路网），在扰动副本中**用同一合同**重算受影响项 | 保留 coords 扰动族，最完整 | 需先做一次只读溯源；若为外部路网数据则**不可**重建，方案作废 |
| **C 缩小扰动族** | 只扰动 `demands/tw_start/tw_end/service_time`，**排除 coords 与 dist_mat** | 立即可实现、自洽 | 检测能力最弱（不覆盖空间信息）；结论边界更窄 |

**我的建议**：先做 **B 的只读溯源**（成本低、结论决定性强）；若合同可恢复则用 B，否则用 **A**。
无论选哪个，都必须：不可见订单规则不变、可见-可见距离逐位不变、`n_invisible_modified` 按位级变化计数、
两侧投影独立重建并登记调用、无法完整比较返回 `NOT_VERIFIABLE`。

## 4. 其余保持

B1/B2 已通过的性质（字段级寻址、位级计数、确定性、输入不变）保留；
P0-A 的逐位比较器维持 `ESTABLISHED`（16/16）；
P0-C（正式 worker/supervisor 与真实入口闭包）仍 `PENDING`；
120/60/3600 三硬截止、原子快照发布、manifest、失败不可重采、64 槽位分母、合同与数据身份**全部不变**。

## 5. F 侧边界（按你的要求明确写入）

**F-free 投影通过只证明"该无模型决策信息边界在所声明的扰动族下不变"**；
真正的 F 特征提取器仍需在**后续 F−D-Geo 机制诊断资格门**内单独核验，
本轮**不得**声称"模型特征因果安全已证明"。

## 6. 状态与 Claim limits

```
P0_A = CLOSED_ON_COUNTEREXAMPLE_GATE
P0_B = BLOCKED_BY_DIST_MAT_CONTRACT (SPEC_CORRECTED / NOT_PASSED)
P0_C = PENDING
```
本轮未采集状态、未调用 F/D-Geo、未改历史合同/profile/冻结源码/归档报告、未重跑服务器计时样本。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部不变。
