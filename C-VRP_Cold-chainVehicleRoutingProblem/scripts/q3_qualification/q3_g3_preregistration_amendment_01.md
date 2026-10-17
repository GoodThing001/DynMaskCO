# G-3 预声明 · 勘误 01：跨字段拼接歧义 + G-08 降级 + 输入契约更正（2026-10-17，追加式）

> **性质：勘误（amendment），追加式。** 对象：`q3_g3_preregistration_2026-10-17.md`。
> **本勘误先于任何新增控制的运行而登记**（用户明确要求），且**不改写**原 9/9 历史记录、
> 不改写 G-3 报告、不修改任何生产 helper。
> 基准远端 HEAD：`a71a0c3cede64b15e20f34f7833134d6236bb7a1`。

---

## 1. 新增缺陷类别：`plan_hash` 的**跨字段拼接歧义**（确定性，非概率）

### 1.1 源码机制

`action_contract.py:301-309` 在**同一条字节流**上连续 `h.update()`，**字段之间没有长度前缀或分隔符**：

```python
h.update(str(vid).encode())
h.update(str(p.anchor_node).encode())
h.update(repr(tuple(p.suffix)).encode())
```

### 1.2 已实测的确定性冲突（合法 suffix `(5,6)`，符合生产契约）

| 字段 | 计划 A | 计划 B |
|---|---|---|
| `vehicle_id` | `1` | `12` |
| `anchor_node` | `23` | `3` |
| `suffix` | `(5, 6)` | `(5, 6)` |
| **拼接后字节** | `b'123(5, 6)'` | `b'123(5, 6)'` |
| **完整 SHA256** | `58137566f9309ff7eaa8726199c7323e76b6b7f233dd367a800709be008cea6e` | **同左** |
| `plan_hash`（前 16 hex） | `58137566f9309ff7` | **`58137566f9309ff7`** |

同一现象在 `suffix=(3,5)` 下为 `f5c20e00860e9e87`（与独立审查给出的表格逐字一致）。

### 1.3 为什么这**不是**同一种缺陷

| 缺陷 | 性质 | 能否靠"用完整 256 位摘要"修好 |
|---|---|---|
| 字段遗漏（`anchor_time`/`anchor_load`） | **信息未被纳入编码** | **不能**（信息根本不在输入里） |
| 摘要截短 16 hex | 概率碰撞预算降到 2^32 | **部分**（用完整摘要可缓解） |
| **跨字段拼接歧义** | **序列化非单射**：两个不同计划映射到**同一字节流** | **完全不能**（前像相同 ⇒ 摘要必然相同） |

⇒ 三者必须**分别**登记、**分别**计数、**分别**给负控。G-3 的身份规范必须包含
**无歧义编码**（长度前缀或显式分隔 + 位型编码），而不仅是"把两个字段加进去"。

---

## 2. G-08 结论降级：样本级观察，**不是**结构性保证

原 G-08 写"`plan_hash` 能分辨 suffix / `anchor_node` / 车辆集合变化"。**该表述过度推广**，就地更正：

| 原句 | 更正后 |
|---|---|
| "plan_hash 能分辨 `anchor_node` 变化" | 仅在**其余字段同时不变**时成立；若 `vid` 同时变化可以**恰好补偿**（§1.2），此时不分辨 |
| "plan_hash 能分辨车辆集合变化" | 仅在样本上成立；`vid` 的数字串与 `anchor_node` 的数字串相邻，存在**跨字段补偿**空间 |
| "缺陷只涉及两个被遗漏字段" | **撤回**：共有**三类**缺陷（遗漏 / 截短 / 拼接歧义） |

**处置**：G-08 保留为**该测试样本上的观察**（其 9/9 记录不改写），但**不得**被引用为旧哈希的
结构性保证。任何后续文档若引用"旧哈希能分辨 `anchor_node`/车辆集合"，必须先声明"样本级、非一般性"。

---

## 3. 输入契约更正：生产计划的 `suffix` **不含尾部 0**

`action_contract.py:107`：

```python
suffix=tuple(int(x) for x in v.mutable_suffix if int(x) != 0)
```

`VehiclePlan` 的 docstring（`:68`）亦写明"客户序列（不含尾部 0）"。

**因此我原主反例使用的 `suffix=(3,5,0)` 不符合生产输入契约**。更正：

- 主反例（字段遗漏）改用**合法** `suffix=(3,5)` 重做（见 M-3），结论不变：
  `anchor_time 24.0→30.0`、`anchor_load 12.5→25.0` ⇒ `plan_hash` **相同**，`plan_identity_v2` **不同**；
- 原 9/9 记录**保留不改**（历史），其"字段遗漏"结论由合法 suffix 的新控制**独立复现**；
- 后续所有涉及计划身份的测试输入**必须**满足该契约（无尾部 `0`、无 `0` 元素）。

---

## 4. 新增判据与负控

| 编号 | 内容 |
|---|---|
| **C13**（新） | 候选身份必须**区分** §1.2 的歧义对（`vid=1,anchor=23` 与 `vid=12,anchor=3`，同 suffix）；这是"无歧义编码"的机械判据 |
| **C14**（新） | 候选身份必须**区分**合法 suffix 下的字段遗漏对（`anchor_time` 1 ulp、`anchor_load` 1 ulp 各一次） |
| **N-15**（新） | 跨字段拼接歧义对 ⇒ 候选必须 `IDENTITY_MISMATCH`；旧身份为 `IDENTICAL`（已实测） |
| **N-16**（新） | 尾部 `0` 或含 `0` 的 suffix ⇒ `REFUSED: SUFFIX_CONTRACT_VIOLATION`（生产不产生此形状；出现即为非法输入） |

**注意**：**N-1（时间 1 ulp）与 N-15（拼接歧义）不是同一种缺陷**，不得合并计数。
N-2（载货）与 N-1 同属"字段遗漏"，可合并为一条家族但须分别给出用例。

---

## 5. S-11…S-16 的裁决意见（按独立审查，逐条落地）

| 裁决 | 意见 | 落地方式 |
|---|---|---|
| **S-11** 16 hex 截短列为缺陷 | 同意，**但须与拼接歧义分开** | §1.3 三缺陷分类；C13 专门覆盖歧义 |
| **S-12** 唯一权威来源 | **有条件**同意：`VehiclePlan` 是待认证计划的字段来源，但锚点**必须**与冻结执行状态**独立核对** | §6 的信任链；新增 `ANCHOR_CROSS_CHECK` 要求 |
| **S-13** 取代 sampler 自报哈希 | 同意独立证据，**不同意简单覆盖** | §7：认证器对**实际审核的冻结计划**产出身份；安装方**独立复算**；`certified_plan_hash` 仅作交叉核对 |
| **S-14** 四标签正式登记 | 原则同意，名称待核对 | 四个**原始字符串**抄录如下，供逐字核对：<br>`ANALYTIC_REAL_BOUND_CONDITIONAL`<br>`FLOAT_EXECUTOR_OUTWARD_BOUND_NOT_ESTABLISHED`<br>`K_HAS_STRUCTURAL_BOUND / NO_CERTIFIED_TIGHT_BOUND`（同一行内一对）<br>`PRODUCTION_NOT_QUALIFIED` |
| **S-15** 两条边界行为复现 | 同意，**仅限受控合成** | 在隔离候选中用 mock 安装回调验证；不接触生产安装器 |
| **S-16** 生产替换独立授权 | 同意为**强制前置** | 新版本须：单独预声明 → 回归 → 冻结 → 身份比对 → 用户授权；**不得**改历史 helper 或既有冻结证据 |

---

## 6. S-12 的信任链（不可合并处理）

```
冻结的执行器状态
  └─ 已执行前缀 / 承诺 / 车辆与冷链状态            （唯一权威：env 的冻结快照）
       └─ VehiclePlan 来源核验                     （build_vehicle_plans 的输入必须正是该快照）
            └─ anchor_node / anchor_time / anchor_load / suffix   （VehiclePlan 字段为**待认证计划**来源）
                 └─ 认证器对冻结候选生成完整身份   （对**实际审核的对象**计算）
                      └─ G3_PLAN_IDENTITY_V1 独立证据（完整 64 hex + 逐字段）
                           └─ 安装前独立复算并比较
                                └─ 不一致 / 状态变化 / 资格缺失 ⇒ REFUSED
```

**新增要求 `ANCHOR_CROSS_CHECK`**：`anchor_node` / `anchor_time` / `anchor_load` 除了来自
`VehiclePlan`，**必须**与冻结执行状态中的对应量独立比对（`current_node`/`committed_next`、
`ready_time`/`committed_finish`、冷链总载）。**不匹配 ⇒ `REFUSED: ANCHOR_NOT_FROM_FROZEN_STATE`**。

---

## 7. S-13 的设计要求：独立证据替代，而非改名

当前 `d3_geo_adapter.py` 在 `certify_fn(P)` 返回后由 sampler 自行计算 `_plan_identity(P)`；
`InstallLedger.finish()` 会重算安装计划的旧哈希，但**两侧使用同一个有缺陷的编码器**，
故无法检测 §1.2 与 §3 的物理字段变化。**这不是"换名字"能解决的**。要求：

1. 认证器必须对**实际审核的冻结对象**产出身份（并回报该对象的**逐字段**值，不只回报摘要）；
2. 安装方必须**独立**读取候选计划并**重算**身份（不得采信 `certified_plan_hash`）；
3. 比对必须覆盖**完整 64 hex + 逐字段**；摘要不匹配或逐字段不匹配 ⇒ `REFUSED`；
4. 认证与安装之间计划对象**不可变**（值拷贝 + 只读视图）；可变共享引用 ⇒ `REFUSED: PLAN_MUTATED_BETWEEN`；
5. 任何 `REFUSED` ⇒ `n_installs = 0` 且安装回调未被调用。

---

## 8. 影响与不主张

- **不改写**：原 9/9 记录、原 G-3 报告、G-2/G-1/S-8 任一文档；本勘误只追加。
- **不改变**任何科研状态（见 §9）。
- **明确不主张**：旧哈希能分辨 `anchor_node`/车辆集合（样本级而已）；截短是唯一或最严重缺陷；
  拼接歧义与截短是同一缺陷；候选身份一致即资格成立。

---

## 9. 状态（不变）

```
GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED（永久，不追认）
THERMAL_CONTINUOUS_NOT_ESTABLISHED / Q3_FORMAL_FREEZE_NOT_AUTHORISED / M2_NOT_AUTHORISED
L1 BLOCKED / L2_NOT_ESTABLISHED / A3_NOT_ESTABLISHED
stable_online_gain = False / HISTORICAL_BACKBONE_UNVERIFIED
C0 AWAITING_USER_BUDGET_RULING / t_min 恒 0
最终测试种子 SEALED
PRODUCTION_NOT_QUALIFIED
```
