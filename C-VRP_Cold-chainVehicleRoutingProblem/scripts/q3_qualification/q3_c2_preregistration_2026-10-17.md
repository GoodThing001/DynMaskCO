# C2 共同身份契约 · 七要素预声明（`C2_SHARED_IDENTITY_V1`，2026-10-17）

> **性质：预声明（pre-registration）。本轮只做规格 + 只读接口审计，不做实现、不接线、
> 不做 G6 持久化、不做 C3 联合门实验。**
> 不改任何生产 helper、不真实安装、无训练 / GPU / 正式 rollout / 最终种子；
> 若必须字段只能从**隐藏未来**取得，**立即拒绝**，不放宽 strict-online。
> 基准 HEAD：`da8a93a2029e18b2e19be143706ec31979fcf335`。

---

## 0. 源码身份（执行前登记，全部取**提交 blob**摘要）

| 项 | 值 |
|---|---|
| `scripts/simulation/action_contract.py`（M1-a） | `9d64273aa3df98cb931c0c8c4b0d0112c134c13e080d3b4f1615b8ae06d2b53b` |
| `scripts/m1b_adapter/m1b_q2_certifier.py`（M1-b） | `7674205b34db5fc5ec3acf226acfc972158dcdd3a0b8c61bb2c5ea607a0e1a97` |
| `scripts/evaluation/run_exp_energy_c0.py`（C0 `certify_plan`） | `9f0e9dd321abebe05a549dc98c14778251eb2583b71749adb9b39ba13a877bbf` |
| `scripts/simulation/d3_geo_adapter.py`（生产安装入口） | `099c3517472d886ece4ea3e929df2ec54d0e459b2a2e2e07da5b658acc07adc5` |
| `scripts/m1b_adapter/m1b_immutable_snapshot.py`（状态身份） | `38acab48996211191c60ca88edc9190c327815381fe2b287ec2d7f8b06432c21` |
| 已交付的隔离候选 | G-1 `q3_g1_time_identity_candidate.py`（26/26）、G-3 `q3_g3_plan_identity_candidate.py`（17/17） |
| 只读接口审计（本轮） | `scripts/tests/test_q3_c2_interface_audit.py` → **7/7 观测** |

---

## 1. 假设（由本轮的只读接口审计**实测**得出，不是设计假设）

| 编号 | 事实（AST 静态分析，`interface_audit.json`） |
|---|---|
| **H-1** | **三方都不接受任何 state/plan 身份参数**：M1-a 取 4 个裸锚点标量，M1-b 取 snapshot 对象，`certify_plan` 取 env/vehicles/plan。⇒ "共享一个身份"**不可能**靠透传一个 digest 满足 |
| **H-2** | **只有 M1-b 复算身份**（`verify_state_unchanged`、`validate_contract_for_quality_certificate`）；M1-a 与 `certify_plan` **不复算任何身份** ⇒ 若由调用方给 digest，这两个会**无条件接受** |
| **H-3** | **没有任何一个状态字段被三方同时读取**（交点 = 空，并集 = 14 个字段）⇒ 共享状态必须**定义**，不能**发现** |
| **H-4** | M1-a 把 `manifest_consistent: True` 作为**字面量**返回，且 `projected_energy/quality/thermal` 恒为 `None` ⇒ 消费方无法区分"算出来的通过"与"写死的通过" |
| **H-5** | 三方认证的是**互不重叠**的东西（可行性重放 / 质量+热+身份门 / C0 能耗预算），**没有一方**证明自己看到的状态与另两方相同 |
| **H-6** | 生产安装入口 `d3_geo_adapter.py` **从不构造 M1-b 认证器、从不调用 `certify_plan`** ⇒ 今天不存在任何"三方比对"的落点 |

---

## 2. 唯一主对照（the single primary control）

> **主对照 = 四方独立复算（quadruple independent recomputation）。**
> 对同一冻结状态 `S`，由**四个责任方各自独立**计算 `C2_SHARED_IDENTITY_V1`：
> M1-a、M1-b、C0 `certify_plan` 的消费方、以及安装方；
> **判据只有一条**：四份摘要必须**逐字相同**；对 `S` 的任一被绑定字段做 1 ulp 变更，
> 四份摘要必须**同时改变**（不得有三份跟不上）。

**为什么只有一个主对照**：C2 的问题是"不同认证层是否在证明同一个对象"，
这只能用一个**同一性/差异性**判据回答。任何第二个对照（例如"摘要是否更快/更短"）都会把资格
问题变成工程口味问题。G-1/G-3 的隔离候选取用的编码约定是**被复用**的基线，不是对照。

---

## 3. 判据（C1–C12，全部机械化、可否决）

| # | 判据 | 通过条件 |
|---|---|---|
| **C1**（主） | 四方独立复算一致 | 同一 `S` ⇒ 四份摘要逐字相同；任一绑定字段 1 ulp 变更 ⇒ 四份**同时**改变 |
| **C2** | 统一身份内容 | 摘要覆盖 `contract / state / plan / projection` 四元组，且绑定：执行前缀、**不可取消的接单承诺**、`instance_id`、`event_id`、源码身份 |
| **C3** | 无歧义、无损编码 | 逐字段定界 + 位型浮点；**禁止** `round`/十进制格式化；完整 64 hex；`null` 与 `0.0` 区分 |
| **C4** | 责任声明（三方） | M1-a / M1-b / C0 各自**声明实际读取了什么、认证了什么**；**禁止**三方仅复用同一个未经核验的调用方摘要 |
| **C5** | 常量声明可见 | 任何恒为真的字段（如 `manifest_consistent`）必须标为 `declared_constant`，消费方据此不得计为计算通过 |
| **C6** | 状态单一来源 | `state` 部分必须由**冻结执行状态**导出；调用方自报的状态字段一律拒绝（沿用 G-3 L-5 教训） |
| **C7** | 计划身份 | 复用 G-3 的完整计划身份（含 `anchor_time`/`anchor_load`、定界编码、深度冻结） |
| **C8** | 时间身份 | 复用 G-1 的无损时间身份；**1 ulp 必须被识别**（不得用未预声明的绝对容差） |
| **C9** | 跨事件/跨实例 | 旧收据跨事件复用 ⇒ `STALE_CERTIFICATE`；跨实例 ⇒ `CROSS_INSTANCE_CONFUSION` |
| **C10** | G-2 未合格即阻断 | G-2 预算/距离模型/浮点外扩未合格时，**不得**签发可安装的联合认证（`JOINT_NOT_CERTIFIABLE`） |
| **C11** | 隐藏未来即拒绝 | 若某必要字段只能从未揭示订单取得 ⇒ **立即 REFUSED**，不放宽 strict-online |
| **C12** | 身份 ≠ 资格 | 四方摘要一致**不**等于 G-2 预算合格、**不**等于 Q3 连续温度成立、**不**等于可安装 |

---

## 4. 参数与次数 + 负控（实现前固定）

| 项 | 值 |
|---|---|
| 合成状态 | ≤ 8 个（3 车 × suffix ≤ 4 节点），不读 `data/` |
| 责任方 | **4**（M1-a / M1-b / C0 消费方 / 安装方），各自独立复算 |
| 重复次数 | 纯函数，每项 1 次 |
| 随机种子 | 不使用（确定性构造） |

| 编号 | 注入 | 预期 |
|---|---|---|
| **N-1** | 时间漂移 1 ulp（`anchor_time`） | 四份摘要**同时**改变；任一不变 ⇒ FAIL |
| **N-2** | 计划变更（suffix 交换两项） | 同上 |
| **N-3** | 合同变更（`hpu` 1.0 → 2.0） | 同上；且 G-1 侧 `NOT_CERTIFIABLE` |
| **N-4** | 状态漂移（认证后改 `committed_next`） | `POST_CERTIFICATION_STATE_CHANGED` |
| **N-5** | 旧收据跨事件复用（`event_id` 变） | `STALE_CERTIFICATE` |
| **N-6** | G-2 资格缺失（`budget_qualified=False`） | `JOINT_NOT_CERTIFIABLE` |
| **N-7** | 调用方自报 `state_identity` 与冻结状态不符 | 拒绝；**不得**采信自报摘要 |
| **N-8** | 某字段只能从隐藏未来取得 | **REFUSED: HIDDEN_FUTURE_FIELD_REQUIRED**（不放宽 strict-online） |
| **N-9** | 三方共用一份未核验的调用方摘要 | `SHARED_DIGEST_UNVERIFIED`（判据 C4 的机械形式） |
| **N-10** | 恒真常量被计为通过 | `DECLARED_CONSTANT_NOT_A_PASS`（判据 C5） |

---

## 5. 预期方向

| 量 | 预期 |
|---|---|
| 四方摘要 | 同一状态**逐字相同**；1 ulp 变更**四方同时变** |
| 现有三方 | **做不到**（H-1/H-2：两个不复算、三个签名不收身份）⇒ C2 必须**新增契约**，不是接线 |
| 状态同一性 | 必须由**冻结状态**导出，不接受调用方自报 |
| G-2 未合格 | 一律 `JOINT_NOT_CERTIFIABLE` |
| 生产路径 | **不变**（本轮不实现、不接线） |

---

## 6. 失败条件（任一成立即 C2 规格不合格）

| 编号 | 条件 |
|---|---|
| **C2-F1** | 主对照出现"四方不一致"或"1 ulp 只动了一部分" |
| **C2-F2** | 出现三方复用同一份**未核验**的调用方摘要 |
| **C2-F3** | 任何身份字段用绝对容差代替 1 ulp 判据（且未预声明） |
| **C2-F4** | 为凑齐字段而读取未揭示订单（**必须改为拒绝**） |
| **C2-F5** | 把"四方摘要一致"写成"可安装"或"G-2 合格" |
| **C2-F6** | 恒真常量被当作计算通过 |
| **C2-F7** | 改生产 helper / 冻结树 / 数据 / `MASKCO_code/` |
| **C2-F8** | 未经授权 commit/push（本轮按常设要求推送，仅限规格与审计） |

---

## 7. 资源门

只读审计 **< 30 s** 纯 CPU；无 `data/` 读取；无 GPU/训练/正式 rollout/最终种子；
不改生产件；无网络。

---

## 8. `C2_SHARED_IDENTITY_V1` 草案

### 8.1 统一身份内容（四元组 + 五项绑定）

```
C2_SHARED_IDENTITY_V1 = {
  contract:   {contract_hash, hours_per_time_unit, distance_km_per_unit,
               travel_domain{models, multiplier_ceiling, k_structural_bound/k_certified_tight_bound}}
  state:      {state_identity_from_FROZEN_STATE, clock, coldchain_anchor,
               per_vehicle{anchor_node, anchor_time, anchor_load, status, current_node, committed_next}}
  plan:       {G3 计划身份（深度冻结 + 定界编码 + 逐字段证据）}
  projection: {projection_identity, projection_challenge}
  bindings:   {executed_prefix, committed_order_ids(不可取消承诺), instance_id, event_id,
               source_identity(blob 摘要), c2_contract_version}
  digest:     sha256(逐字段定界编码)   # 完整 64 hex
}
```

### 8.2 统一验证责任（三方各自声明，禁止共用未核验摘要）

| 责任方 | 必须声明"我读了什么" | 必须声明"我认证了什么" | 禁止 |
|---|---|---|---|
| **M1-a** `certify_route` | env 的实例数组 + 调用方给的 4 个锚点标量 | 该 suffix 的 TW/容量/返仓/温区可行性 | 把 `manifest_consistent` 当作已计算结论；**不可**仅凭调用方锚点就宣称绑定了状态 |
| **M1-b** `certify` | snapshot / projection / contract / per-class 常量 | 合同身份、质量下界、热采样、状态未变、投影身份稳定 | 接受外部 `per_class_constants` 覆盖（已拒绝）；不得以自报 `state_identity` 取代重算 |
| **C0** `certify_plan` | env 数组 + 车辆冷链状态 + `plan` + `served_mask` + `B` | **仅**总能耗 ≤ B | 不得被读作可行性/质量/热/身份认证；不得静默丢弃已服务订单而不登记 |

**新增硬要求（对应 H-1）**：三方**都必须**接受同一个 `C2_SHARED_IDENTITY_V1` 参数并**各自重算**，
不接受任何一方透传的摘要作为他人结论的依据。

### 8.3 与 G-1 / G-2 / G-3 的职责分离

```
G1 时间身份（无损）        → 提供 state/plan 里的时间与锚点编码
G2 执行预算资格           → 提供 T_clock_sup / n_max / 距离模型域；未合格则 C2 不得签发可安装结论
G3 计划身份 + 交接对象     → 提供 plan 部分与"审核对象=使用对象"
C2 共同身份契约（本文件）  → 只统一"四方是否在证明同一个对象"
```

**禁止的合流**：C2 一致 ⇒ 不得推出 G-2 合格 / Q3 成立 / 可安装。

---

## 9. 保留的阻断项

`B-1…B-9` 全部保留（B-9a 有条件闭合、B-9b 划出 G-3）；另保留 G-2 未合格、G-2 浮点外扩界未建立、
G-TMIN 正式 FAIL（不追认）、V-10 干净 checkout FAIL、Q3 连续温度未建立、B-7 证据本地性缺口。

---

## 10. 状态（不变）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、
`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0、
最终测试种子 `SEALED`、`PRODUCTION_NOT_QUALIFIED`。
