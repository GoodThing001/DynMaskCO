# `P_after_install` 四字段前瞻采集：执行版预登记草案（**不执行**）

- 轮次：`A1-D3-L1` / prospective four-field install-evidence collection
- 状态：`PREREG_DRAFT_FOR_USER_SIGNATURE` / **`NOT_AUTHORISED`**
- 本文件**只交付可签署规格**。它**不授权、也未执行**任何前瞻采集、任何真实安装、任何批次。
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**
- 上位：`D3_PROSPECTIVE_PREREGISTRATION.md`（四字段前瞻采集的原始要求）、
  `D3_G0_RESIDUAL_RISK_AUDIT.md`（A-4b / A-5c / A-5d）

---

## 0. 要解决的问题

D3 前瞻采集要求每个决策点记录四个**互不替代**的字段：

```
P_before          决策前的完整车队计划
P_proposed        控制器/评分臂提出的计划
P_certified       通过认证的计划（认证对象）
P_after_install   环境**实际安装**后的计划
```

硬约束（原始要求，**原样保留**）：**`P_after_install` 必须由真实安装或事务回执证明，不得由
`P_certified` 推算**。此前该字段**不存在**，适配器只有内存事件、无落盘回执。

## 1. 本轮已闭合的部分（**可执行证据**），以及**仍未闭合**的部分

| 能力 | 状态 | 证据 |
|---|---|---|
| 落盘安装回执（immutable 新目标名 + 写后读回校验） | **已实现并验收** | `B3A_receipt_evidence/D3_INSTALL_RECEIPT_ACCEPTANCE.json` **7/7** |
| 独立**进程**重读回执并重算计划哈希 | **已实现并验收** | 同上 R1 |
| 回执只在 ledger 确实安装**恰一个**计划时写出 | **已实现并验收** | 同上 R2/R3 |
| 篡改字节 / 篡改计划值都被检出 | **已实现并验收** | 同上 R4/R5 |
| `A-5b` 重复 `finish` 不再重复安装 | **已实现并验收** | `D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json` |
| `A-5d` 无回滚函数不再被计成"已恢复" | **已实现并验收** | 同上 |
| `A-4b` 认证计划与安装计划同身份 + 台账存值拷贝 | **已实现并验收** | 同上 |
| **`A-3b` 适配器自身不校验已执行前缀/保护车** | **经执行确认仍开放** | 同上（`RISK_CONFIRMED_STILL_OPEN`） |
| **环境自身 `_plan_and_commit` 的提交不可观测** | **仍未闭合** | 回执记录的是**被请求的安装**及其绑定证据 |

**因此**：回执把"`P_after_install` 完全不存在"变成"**可读、可篡改检出、可独立重读的落盘事实**"，
但**不**把 `P_after_install` 变成"环境确认提交"的证据。后者是本预登记**不主张**的部分。

## 2. 四字段的采集与身份（每个决策点）

| 字段 | 来源 | 必带身份 |
|---|---|---|
| `P_before` | 决策前快照的计划（`build_vehicle_plans`） | `state_hash`、`clock`、`inst`、`event` |
| `P_proposed` | 评分臂返回的计划（含 `failure_reason`） | `plan_hash`、臂 `scorer_id`、`legal_sha256`、`mask_sha256` |
| `P_certified` | 认证器输入并返回 ok 的那个对象 | `plan_hash`、**certifier 身份**、认证细节 |
| `P_after_install` | **仅**来自落盘回执的 `installed_plan_hash` + 回执值拷贝 | 回执 `sha256`、`receipt_id`、`sequence`、**独立重读结果** |

规则（**fail-closed**）：

1. 四字段**各自独立落盘**；任一缺失 ⇒ 该决策点 `INCOMPLETE_INSTALL_EVIDENCE`，**留在分母**；
2. `P_after_install` **禁止**由 `P_certified` 或 `P_proposed` 推算/填充；
3. 回执未通过独立重读（摘要不符 / 计划值不再复现记录哈希）⇒ `INSTALL_EVIDENCE_INVALID`，
   **不重写回执**、**不重跑安装**；
4. `P_certified ≠ P_proposed` ⇒ 记 `certified_plan_identity_mismatch` 并按拒绝处理；
5. 安装**未发生**时必须**如实留空**并记 `receipt_missing`；**不得**用 `None` 冒充"未安装"，
   也不得用空值冒充"已安装"（空值与缺失必须可区分：见 `read_receipt` 返回的
   `absent` vs `schema_ok=False`）。

## 3. 前瞻采集的运行规格

- **采集者是独立的只读观察者**：它驱动环境、消费回执，**不参与**决策；
- **不改变**被测决策路径的任何字节；身份清单（§4）在采集前后各查一次；
- 采集位置（inst × event）**运行前完整冻结**并写进清单文件（含 sha256）；不足则记
  `insufficient_events` 并保留在分母；
- 一次采集**只消费一次**；失败原件保留，**不补采、不挑状态、不追加运行**。

## 4. 身份冻结

| 类别 | 内容 |
|---|---|
| 采集器自身 | 逐文件 sha256（起止各一次） |
| 环境与决策路径 | `strict_online_env.py`、`action_contract.py`、`jf1h_repair.py`、`d3_geo_adapter.py`、`mpre_policy.py`、`coldchain_contract.py`、`coldchain_state.py` |
| 回执通道 | `d3_install_receipt.py`、`d3_install_boundary.py` + 回执目录路径 |
| 认证器 | 认证器可调用身份 + 其依赖文件的逐文件 sha256 |
| 数据 / 合同 / profile | NPZ sha256、`effective_contract_hash`、`profile_hash`（**显式记录**，不沿用"缺口"表述） |
| 环境 | python / jax / flax / numpy 版本、`jax.default_backend()`、设备 |
| 明确排除 | 并行 `CC_Compare` 提交不属于 D3 身份 |

## 5. 预算（执行前按实测冻结）

- 状态数量上限、每状态墙钟上限、安装次数上限、回执条数上限：**首次运行前冻结**；
- `PENDING` / `None` / 无限预算**不得**被解释为无上限；
- 先做一次**极小规模计时资格运行**（不计入正式集合）取得单位成本，再冻结总预算；
- 超时按 `timeout` 记账并保留在分母。

## 6. 判定与主张边界

**允许**声明：
1. 在登记身份下，四个字段的产生率、缺失率与失败原因分布；
2. 回执在采集期内是否**始终**通过独立重读（含篡改检出）；
3. `P_certified` 与 `P_after_install` 的一致性/不一致率（**作为事实**，不作为质量结论）。

**不允许**声明：
1. "环境已提交"（`_plan_and_commit` 不可观测）；
2. 状态级回滚已被证明（`A-3b` 仍开放，且回执不含回滚证明）；
3. 安装计划的品质 / C0 能耗性质；
4. 任何效用、显著性、稳定增益结论。

## 7. 执行前置门

- [ ] 用户签署本预登记（含 §1 的"**未闭合**"三项如实保留）
- [ ] 回执通道的独立重读验收已通过（**本轮已完成，7/7**）
- [ ] 采集位置清单冻结（含 sha256）
- [ ] 身份清单冻结（§4 全表）
- [ ] 预算按 §5 冻结
- [ ] 独立只读复核：规格可机械执行
- [ ] `A-3b` 的归属与处置已获用户裁定（"适配器不校验前缀"是**保留**还是**先修**）

## 8. Claim limits

1. 本文件是**规格**，不是结果；产出时前瞻采集次数 = **0**、真实安装次数 = **0**。
2. 回执只证明"**被请求的**安装及其绑定证据"，不证明环境提交，也不证明计划正确。
3. `A-3b` 经执行确认**仍开放**，不得因为回执已落盘而视作已解决。
4. D3 G0 **仍未通过**；D8 仍阻断于状态认证（G-L）、真实写入唯一性（G-C）与品质保守保证。
5. I3 正式负结果、D2 `not_qualified`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
