# A1_D3_L1_STATE_COLLECTION_EXECUTION_PREREGISTRATION_V1

- 状态：**`DRAFT_FOR_SIGNATURE`**（数据身份、资源上限、因果校验与持久化规则已在本文件冻结；**实际采集仍需用户签署**）
- 依据：用户对 `951dbc1` 的初步裁决（接受服务器计时尚证据用于预算设计；批准编制本预登记）
- 未授权：实际采集、F−D-Geo 比较、L2/L3、模型重训练
- `states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`

## 0. 上一轮计时结果的资格切分（不改写原件）

| 字段 | 取值 |
|---|---|
| `recorded_Q_server` | **`NOT_PASSED`**（原件 `951dbc1` 的 JSON 与 JSONL 不变：`Q_identity=false`、`gates.Q_identity=false`） |
| `posthoc_identity_verification` | **`MATCH`**（限已核对字段：服务器实算 5 源码 + 数据 + profile 与冻结身份逐项一致） |
| `server_timing_evidence` | **`ACCEPTED_FOR_BUDGET_PLANNING`**（总 2.386 s；两槽 0.728 / 0.822 s） |
| `formal_collection_authorization` | **`NOT_AUTHORIZED`** |
| 是否重跑该计时样本 | **否**（不因上传故障重跑） |

## 1. 用户指出的三项边界缺口 → 本预登记中的硬性修法

| 缺口（源码证据） | 本预登记规定 |
|---|---|
| 身份门未真正执行冻结对照（runner `:119–130`：冻结文件缺失时 `frozen=None` 仍继续） | 采集程序**启动即强制读取冻结身份文件**；文件缺失、不可读或任一字段不符 → **立即停止并记 `identity_mismatch`**，不得进入 rollout（对应 `Q_identity` 为硬前置，非事后补核） |
| 因果门范围有限（`:154–158` 只查 R1 槽位规则） | 除槽位规则外，**新增特征层因果校验**：对每个槽位，用一个"删除当前不可见订单"的扰动弹，验证决策可见特征、候选枚举与 planned suffix **不变**（不可见订单不得进入 F/D-Geo 特征构造）；该检查按槽位记录 |
| 单槽计时未覆盖 capture（`:161–165` `slot_start` 晚于 rollout 阶段 capture） | 正式计时**从 capture 调用开始**计到该槽 parity 结束；总墙钟同样从 rollout 开始计时 |

另：`resume_parity` 的**精度判据按既有预登记为逐位一致**，不得降级为近似相等。采集前必须先用
**最小反例**（正负零、NaN、NumPy 标量、原始对象类型）证明比较器能落实该精度要求；反例未过则
**不进入本批次**（记为 `comparator_not_qualified`）。

## 2. 采集主方案（唯一主方案，不再就 K 值做筛选实验）

| 要素 | 冻结内容 |
|---|---|
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz`（**dev_check 全部 16 个开发日**） |
| 位置规则 | **R1**：每个开发日**前 4 个真实揭示事件**（按 `reveal_idx`/可见集合增长判定，**不得**用 `event_id`） |
| 候选状态数 | **≤64**（16×4）；未产生的槽位**按原因记录并保留在分母** |
| 基线策略 | 登记身份的 **JF1-H-F**，**独立 F-free 进程**（不导入任何 F/D-Geo 模块，运行时断言） |
| 合同 | L1 冻结协议：base `32124bfd…30b0` + profile `df1f7a6e…25b6` → effective **`edc1b2ea…4833`**；`booking_horizon=24.0`；`CROSS_VERSION_EXPLICIT` |
| 效果评价 | **禁止**（采集阶段不调用 F 或 D-Geo，不计算任何效用） |

### 2.1 不可变 `day_id × position` 清单

**枚举规则（冻结）**：`day_id ∈ {0,1,…,15}`（dev_check 16 个实例，顺序即 NPZ 内实例顺序，不重排）；
`position ∈ {1,2,3,4}`（该日的第 1…4 个真实揭示事件）。清单为笛卡尔积：

```
(0,1) (0,2) (0,3) (0,4)   (1,1) (1,2) (1,3) (1,4)   (2,1) (2,2) (2,3) (2,4)   (3,1) (3,2) (3,3) (3,4)
(4,1) (4,2) (4,3) (4,4)   (5,1) (5,2) (5,3) (5,4)   (6,1) (6,2) (6,3) (6,4)   (7,1) (7,2) (7,3) (7,4)
(8,1) (8,2) (8,3) (8,4)   (9,1) (9,2) (9,3) (9,4)  (10,1)(10,2)(10,3)(10,4)  (11,1)(11,2)(11,3)(11,4)
(12,1)(12,2)(12,3)(12,4) (13,1)(13,2)(13,3)(13,4) (14,1)(14,2)(14,3)(14,4)  (15,1)(15,2)(15,3)(15,4)
```

共 **64** 个预登记槽位（**预算槽位，不是合格快照保证**）。
物化说明：`(day_id, position)` 在运行前冻结；**实际 `event_id`、`reveal_idx`、`clock`、`reveal_time`
在采集时按上述规则物化登记**；不得声称完整 `event_id` 清单在运行前已生成。
本地对该 NPZ 的普查（与服务器同 sha，见 §3）：每实例 **25** 个不同揭示事件、`insufficient_events=0`
⇒ 4 ≤ 25，规则可满足；该普查须在采集时于服务器**复算并记录**。

## 3. 数据与源码身份（**服务器独立实算**，2026-10-15）

| 对象 | sha256 |
|---|---|
| `dev_check` NPZ | **`66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89`** |
| `M0_TEACHER_MANIFEST.json` | `b84e171676f8afc02e5ac416538965dda224671d719863a1412f659464bcaf04` |
| `cal` NPZ（**仅对照，不得复制为 dev_check 身份**） | `c35fa6834b8007922160f7979680c1fb8c6dda4d93309cd5ef509b57d83f20f1` |
| `train` NPZ（仅登记） | `c70258c4fde86e5ebfb08c17e4f8ccacac5f1b18319dc180eaa048e8ae771d01` |
| `d3_l1_state_collector.py` | `849667e4d572e8250041913eee8d59ce9d376b8119f777179ab5a99f9b71e86f` |
| `strict_online_env.py` | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6` |
| `jf1h_repair.py` | `233f8439536a2af676fe389b012b74c4fa8cf458fde7aef226b084342ca089ee` |
| `recourse_snapshot.py` | `0af3535acaf6a84956104c0f1c8788c04e02499a087b6faa3d264239a19f27f8` |
| `coldchain_contract.py` | `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| `coldchain_state.py` | `a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e` |
| 冻结身份文件 | `f44e3456db5fac67c85829e5f28648174b4c50b81d1af85a288dc78a7795172c` |
| 目标 profile | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |

**身份规则**：采集启动时逐项核对上表；任一不符 → 停止（`identity_mismatch`），
并保留已完成槽位与停止原因。**`cal` 的 `c35fa683…` 不得复用为 dev_check 身份**。

## 4. 每槽位状态校验（全部通过才算合格槽位）

1. `state_hash`：计算一次并**重复计算一次**要求逐位一致（`state_hash_unstable` 为失败原因）；
2. **capture → restore → 再捕获**：同一 `clock` 重捕获的 `state_hash` 必须与原始相同；
3. 恢复载荷三对象齐全（arity=3）且关键决策字段存在（`replanner_state`、`served_mask`、冷链嵌套字段）；
4. **完整后缀 parity**：从该槽恢复后继续 rollout 的**终局轨迹**与原始 rollout 终局
   **逐位/逐字段一致**（按 §1 的精度判据；不一致须输出**首个差异路径 + 值 + 位模式**）；
5. **strict-online 信息边界**：决策可见特征、归一化（`coord_normalize_visible`）与候选枚举不得读取
   未揭示订单；快照可为复核保存完整环境，但**必须与决策可见信息隔离存放**，确保后续 F/D-Geo
   特征构造不会经由快照引入未来信息；
6. `schema_version = recourse_snapshot_v3_cc` 且快照 `coldchain_contract_hash = edc1b2ea…4833`。

## 5. 失败语义（fail-closed，全部保留在分母）

原因分类固定：`capture_failed`、`restore_mismatch`、`state_hash_unstable`、`resume_divergence`、
`causal_violation`、`empty_pool`、`insufficient_events`、`timeout`、`identity_mismatch`、
`comparator_not_qualified`。
**禁止**：补采、替换、追加运行直到通过、把失败槽位记作"两臂一致"、把未取得的结果记作落后。

## 6. 资源上限（三类，运行前冻结；以服务器实测为基线并留保守系数）

| 类别 | 上限 | 依据 |
|---|---|---|
| 完整 rollout（每开发日） | **≤ 120 s** | 实测 cal 实例 0 = 0.58 s（×≈200） |
| 单槽（**capture 起算** → restore → hash ×2 → 完整 parity） | **≤ 60 s** | 实测最大 0.822 s（×≈70） |
| 整批（64 槽 + 16 次 rollout + 全部校验） | **≤ 3600 s** | 实测总 2.386 s（×≈1500） |

计时起点：整批自首个 rollout 起；单槽自 capture 调用起。
超时 → 该槽记 `timeout` 并**保留在分母**，不放宽上限、不重跑。
**注意**：本次样本仅为 1 个 cal 实例的 2 个早期事件，不足以单独证明 64 槽位资源门通过；
`cal` 的耗时不得直接外推为 dev_check 全部日期/事件/冷链状态的成本。

## 7. 运行次数与持久化协议

- **运行次数**：恰好 1 次采集 pass；每槽恰好 1 次 capture、1 次 restore、2 次 hash、1 次 parity；无重试。
- **增量落盘**（`tmux` 不作为持久化）：每个槽位完成即写
  ① 快照文件（独立目录，按 `(day_id, position)` 命名）② `STATE_MANIFEST.jsonl` 追加一行
  ③ `RUN_STATUS.json`（已完成槽位、当前进度、停止原因），均 `flush + fsync`。
- **中断规则**：进程中断或机器重启后，**保留已完成槽位、停止原因与未完成记录**；
  续跑必须使用**新的 run_id**，且两个 run 的记录同时保留，不得只提交成功结果或静默覆盖。

## 8. 本文件**不**授权的事项

实际采集（需用户签署）、F−D-Geo 机制比较、L2/L3、模型重训练、服务器正式资源门的"已通过"表述
（本文件只提供预算设计与上限冻结）。D3 G0、D8、E2、`stable_online_gain=False`、I3 正式负结果与
旧 D2 `not_qualified` 全部不变。
