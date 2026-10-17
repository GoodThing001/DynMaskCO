# G-3 预声明 · 勘误 02：认证—安装 TOCTOU 规格 + S-14 标签修订（2026-10-17，追加式）

> **性质：勘误（amendment），追加式。本文件先于任何候选实现与任何新增控制的运行而登记。**
> 对象：`q3_g3_preregistration_2026-10-17.md`（下称"预声明"）与其
> `q3_g3_preregistration_amendment_01.md`（下称"勘误 01"）。
> 不改写预声明、勘误 01、9/9 与 6/6 的既有记录；不改任何生产 helper。
> **本轮只做隔离候选的实施准备与必要的规格补充。**

---

## 1. 触发：两处已核实源码事实（独立审查提出）

### 1.1 安装回调收到的**就是**被核对的那个可变对象，归档拷贝在回调**之后**

`d3_geo_adapter.py:666-696` 的实际执行顺序：

```python
installed_hash = _plan_identity(plan)     # :666  先算身份
if why is not None: ... return False      # :668-688  门控
self.install_fn(plan)                     # :689  把【原始可变对象】交给安装回调
self.n_installs += 1                      # :690
stored = _copy_plan(plan)                 # :691  回调【之后】才做值拷贝
self.installed_plans.append(stored)       # :692
self.installs.append({... 'plan_hash': installed_hash, 'plan': stored})   # :693-696
```

**两个直接后果**：

| # | 后果 |
|---|---|
| **C-1** | `install_fn` 收到的是**原始 `plan`**，不是调用前建立的不可变快照；它可以在安装"过程中"修改计划 |
| **C-2** | 账本条目把**回调前**算出的 `plan_hash` 与**回调后**拷贝出的 `plan` 配对存档 |

`_plan_identity` = `action_contract.plan_hash`（旧哈希，**不含** `anchor_time` / `anchor_load`）；
`_copy_plan` 是值拷贝，但**时点在回调之后**。

### 1.2 持久收据存了物理字段，读回却只用旧哈希校验

`d3_install_receipt.py`：

- `:114-119` 收据**确实存储** `anchor_time`、`anchor_load`；
- `:203-230` `_reproduce_plan_hash()` 把收据的 plan **重建**为 `VehiclePlan`（含两个物理字段，
  `:223-224`），随后调用的是 `plan_hash(rebuilt)`（`:228`）——**旧的、不含这两个字段的哈希**。

因此：**读回收据并"成功复现哈希"，只证明 `vid / anchor_node / suffix` 一致，
不证明 `anchor_time` / `anchor_load` 未被改动。**

> **归类**：这是 **D-1（字段遗漏）的下游证据影响**，**不**算第四类缺陷、**不**重复计数。

---

## 2. 结论：这不是第四种哈希缺陷，而是一个**对象生命周期**问题

D-1/D-2/D-3 都是"身份编码不够严"。本节的两点是**"被核对的对象"与"被使用/被归档的对象"
在时间上可以不是同一个**。所以规格必须补的**不是更强的哈希**，而是：

```
冻结执行状态  --(唯一来源)-->  VehiclePlan 字段
        |
        +-- 认证器：对【实际审核的那个对象】计算完整身份，并回报逐字段值
        |
        +-- 安装方：读取【实际收到的那个对象】独立复算身份，逐字段比对
        |
        +-- 归档：值拷贝必须发生在【比对之前】，且归档对象必须是【被认证的那个快照】
```

**新增要求（写入候选实现约束）**：

| 编号 | 要求 |
|---|---|
| **L-1** | 安装方在调用 `install_fn` **之前**取得不可变快照（值拷贝 + 只读视图），并**对该快照**复算身份 |
| **L-2** | `install_fn` 只接收该快照的只读视图；回调结束后**再次**复算快照身份，与安装前不一致 ⇒ `REFUSED: PLAN_MUTATED_DURING_INSTALL` |
| **L-3** | 归档条目必须绑定**安装前快照**的身份，不得把回调后的状态与回调前的哈希配对 |
| **L-4** | 认证器回报的身份必须覆盖"它实际读到的对象"的**逐字段值**，而不只是摘要 |

---

## 3. S-14：标签的机器可读修订（第三组拆成两个独立状态）

独立审查指出：`K_HAS_STRUCTURAL_BOUND / NO_CERTIFIED_TIGHT_BOUND` 写在同一个标签里是错的，
因为它们是**可以同时成立的两个事实**，不是互斥结论。**接受该意见**，机器可读记录改为：

| 字段 | 取值 | 含义 |
|---|---|---|
| `k_structural_bound` | `true`（附 `k_max_source = n_edge_perturb`） | 源码层面存在有限的结构上界 |
| `k_certified_tight_bound` | `false`（附 `why = 无已认证且足够紧的界`） | 不存在可用于生产的紧界 |

四个语义保留，记录形态调整：

```
ANALYTIC_REAL_BOUND_CONDITIONAL            = true
FLOAT_EXECUTOR_OUTWARD_BOUND_NOT_ESTABLISHED = true
k_structural_bound                          = true   (k <= n_edge_perturb)
k_certified_tight_bound                     = false
PRODUCTION_NOT_QUALIFIED                    = true
```

**不得**再把第三组当作一个整体标签引用；任何后续文档应写两个字段。

---

## 4. S-15 增补：TOCTOU 负控规格（**先登记，后运行**）

在**受控合成环境**中注入以下四类，逐项规定预期。**旧路径只作诊断对照**：
本轮只**观察**旧路径的行为并记录，**不**声称旧系统发生过生产事故。

| 编号 | 注入 | 独立候选的预期 | 旧路径（仅诊断对照） |
|---|---|---|---|
| **T-1** | `certify_fn` 在**审核前/审核后**修改传入的**可变**计划 | 候选必须识别"审核对象 ≠ 回报对象" ⇒ `REFUSED: AUDIT_OBJECT_CHANGED` | 观察认证回报的身份是否与实际审核对象一致 |
| **T-2** | `install_fn` 在**执行期间**修改 `anchor_time` 或 `anchor_load` | 安装后复算快照身份与安装前不一致 ⇒ `REFUSED: PLAN_MUTATED_DURING_INSTALL`；且**不得**归档被修改的版本 | 观察账本条目中"哈希"与"归档计划"是否指向同一物理计划 |
| **T-3** | 安装方使用与认证方**不同的计划对象**（内容相同但非同一对象） | 候选以**内容身份**（逐字段）比对，不依赖对象同一性；内容一致则通过、不一致则 `REFUSED: CERTIFIED_INSTALLED_MISMATCH` | 观察旧门控能否区分"不同对象同内容"与"同对象被改" |
| **T-4** | 持久收据含**已变化的物理字段**，但旧 `plan_hash` 仍一致 | 候选的收据校验必须覆盖 `anchor_time`/`anchor_load` ⇒ `REFUSED: RECEIPT_PHYSICAL_FIELD_MISMATCH` | 观察 `_reproduce_plan_hash()` 是否**看不见**这两个字段的变化 |

**每项都必须明确预期的 `IDENTITY_MISMATCH` / `REFUSED` / `NOT_CERTIFIABLE`，并说明必要字段的唯一来源。**

**禁止的写法**（本轮与后续都适用）：

- 不得把旧系统**可能**存在的 TOCTOU 风险写成"已发生的未认证安装事故"；
- 不得把"旧路径在合成注入下表现如此"写成"生产环境曾如此"；
- 不得因为某个注入在当前生产代码里**无法触发**就省略该负控（规格要求的是**边界闭合**）。

---

## 5. 资格边界（本轮）

| 允许 | 禁止 |
|---|---|
| 新建**隔离**模块与**只读**诊断探针 | 替换 `action_contract.py` / `d3_geo_adapter.py` / `m1b_immutable_snapshot.py` / `d3_install_receipt.py` |
| 读取生产 helper 以复现机制 | 修改冻结源码、数据、`MASKCO_code/` 上游 |
| 轻量 CPU 合成 | 正式实验、GPU 训练、最终测试种子 |
| 记录旧路径的诊断行为 | 生成**生产安装证书** |

**G-3 只可形成"隔离候选工程诊断"**；生产替换必须另行取得用户明确授权。

---

## 6. 候选实现的最终报告必须列出（四类，缺一不可）

1. **新旧身份能区分的字段清单**（逐字段：旧 `plan_hash` 能否区分 / 新身份能否区分）；
2. **不能证明的物理资格**（例如：无法证明进程外篡改、无法证明最终种子、无法证明 G-2 预算）；
3. **全部 SKIP / FAIL 及其判定规则**（含 `PL-12` 这类未声明失败行的处置）；
4. **剩余阻断项**（沿用预声明 §10 的 B-1…B-8，逐条给出状态）。

---

## 7. 状态（不变）

`GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED`（永久，不追认）、
`THERMAL_CONTINUOUS_NOT_ESTABLISHED`、`Q3_FORMAL_FREEZE_NOT_AUTHORISED`、`M2_NOT_AUTHORISED`、
L1 `BLOCKED`、`L2/A3_NOT_ESTABLISHED`、`stable_online_gain=False`、
`HISTORICAL_BACKBONE_UNVERIFIED`、C0 `AWAITING_USER_BUDGET_RULING`、`t_min` 恒 0、
最终测试种子 `SEALED`、`PRODUCTION_NOT_QUALIFIED`。
