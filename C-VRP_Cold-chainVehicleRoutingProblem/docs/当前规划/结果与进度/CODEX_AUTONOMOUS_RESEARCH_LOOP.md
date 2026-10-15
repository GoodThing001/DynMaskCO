# CODEX_AUTONOMOUS_RESEARCH_LOOP

> 长期自主科研迭代账本。**新会话从此文件 + `项目当前状态.md` + 进度表顶部恢复**，不重新盘点项目。

- 最终使命：以严谨、可证伪、可复现的持续研究，探索并**验证** MaskCO 在 strict-online 动态冷链取货—回仓问题上的稳定正向在线效用增益；
  若证据不支持，则交付完整、可复核的负结果或机制发现。
- 当前 HEAD（最近一次本账本更新时）：见 §1「已消费正式实验」与最后一次 commit。
- 权威顺序：最新用户复核 > `项目当前状态.md` > 进度表顶部 > 本账本 > 历史报告。

## 0. 最近收束（新会话从这里续接）

### 0.000000 本轮第七阶段：L1 正式机制比较**签署包**已就绪（未执行）


用户确认下一步优先级为「完成 L1 的 F−D-Geo 正式机制比较签署包」。本阶段只做**签署包准备**，**未执行**任何正式比较。F 资格批次**未重跑**（继承 V1.22 的 64/64 PASS 与独立裁决）。


**新增实现**：`scripts/simulation/d3_l1_real_arm.py`（**真实** MaskCO scorer 驱动两臂协议 + 与部署一致的逐步 extractor + 逐步账本）、`scripts/tests/d3_l1_real_arm_acceptance.py`、`scripts/tests/d3_l1_l1_signing_package.py`；协议扩展 `BudgetGate`（两臂**相同**上限、逐次评分与逐秒计费、越限 **fail-closed** 回退）与 `on_score_step` 钩子（校验 logits 属于**本步**）。既有行为保持：两臂验收 **10/10**、独立审计 **17/17** 仍全绿。


**真实臂验收 6/6**（`l1_gates/L1_REAL_ARM_ACCEPTANCE.json`）：83 个训练叶子**逐位恢复**；每次调用返回的 logits 长度**恰等于该步**有序 legal 长度；两臂收到**逐字节相同**的初始输入；两臂**都被**账本记录（故跨步复用可被检出）；D-Geo 仅读登记距离向量。**过程中发现并修复两处缺陷**：①JIT 形状必须取**数据导出的最大值**（用单状态自身长度会使 pad 宽度为负）；②只给 F 臂加仪表会使"两臂逐步指纹都被记录"这一断言**先天为假**。


**预算由 V1.22 实测推导，不编造乘数**：0.422 s 是**单状态两次首步**评分的跨度，**不是** repair 成本；单次调用成本由墙钟反推（最坏 0.211 s）；步数会计固定 `m = min(2, |B 池|)` ⇒ 两臂各 ≤128 次调用、**上限完全相同**、**不因 F 更慢而给额外额度**；沿用既有 10 s 单决策 / 60 s 初始化 / 180 s 整批，**不放宽**。**明确记录两处缺口**：第 2…m 步单位成本**从未实测**；**不为填补缺口开未授权试跑、不猜乘数**（唯一合法收紧手段是**降低 m**）。离线诊断**不得**冒称通过在线门。


**七要素预登记** `A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md`；**签署包** `l1_gates/L1_FORMAL_EXECUTION_SIGNING_PACKAGE.json`（run_id `d3l1-l1-two-arm-20261015-l1v1`、V3 = `NOT_RUN`、不安装、一次性消费、九项机械验收 M1–M9 **全真**）；**V3 = NOT_RUN 的直接后果**已在预登记中写明：本轮**只允许**讨论候选动作分歧与**未认证**的完整修复差异，**不得**称为"已认证的可执行计划差异"。C0 预算分类为 **`NON_BLOCKING_UNRESOLVED`**，保留原始记录，**不伪装成 PASS**。


**F−D-Geo 比较本身未启动**；下一步唯一人工决策点 = 对具体 `run_id`、源码身份、预算与预登记 SHA 的**正式执行签署**。

### 0.00000 本轮第六阶段：**真实 F 前向已执行并通过（V1.22，64/64）**


用户在会话中**明确批准** `run_id = d3l1-real-f-forward-20261015-v122` 与预登记 SHA `f5a0dd00ddf39dd7d8576d97a463a272cf94331a7c7583a5d93e071c3053ed5e`。批次经**唯一合法路径**（`--supervise --authorization --preregistration`）执行，一次性凭据消费**恰好一次**。


**结果：64/64 状态合格、`ROUND_PASS`、verdict `REAL_F_FORWARD_ENGINEERING_PASS`。**


| 量 | 实测 | 冻结上限 |
|---|---|---|
| 前向调用 | **129**（1 warm-up + 64×2） | ≤129 |
| 单槽跨度 max | 0.422 s | 10 s |
| 初始化+编译 | 6.688 s | 60 s |
| worker 墙钟 / 监督器墙钟 | 33.0 s / 35.1 s | 180 s |
| watch 路径 | 34/34 一致，起止身份记录相符 | — |

**独立裁决**（`scripts/tests/d3_l1_v122_real_f_adjudication.py`）**6/6 PASS**：A 整轮 PASS、64 行合格无撤销；B 逐行评分契约（长度=有序 legal 长度、dtype `<f4`、两次前向**逐字节相同**、原始 pickle 复现记录十六进制）；C 逐行**输入身份**（input/legal 指纹与 state_hash 全等于冻结目标）；D 34 条 watch 全部仍哈希到冻结值；E 全程在**冻结 60/10/180 s** 内且前向数 ≤129；F **恰好一条**一次性回执，且凭据文件仍哈希到被消费的摘要。


**独立只读复核器**：`VERIFIED`，64 行全部重算通过、**独立判定器与生产判定器 64/64 逐行一致**、0 分歧。


**明确不主张**（随裁决一并记录）：无 F−D-Geo 动作分歧、无决策质量或在线效用增益、无显著性、无学习贡献、无 MaskCO 结论；backbone 标签维持 `HISTORICAL_BACKBONE_UNVERIFIED`；`stable_online_gain = False`。


**本次机会已消费**：不重跑、不补齐缺失状态、**禁止拼接 V1.11 缺失的 42 条**。零真实 F 审计现已如实**拒绝**该零声明（`ZERO_F_CLAIM_REJECTED`，5 PASS / 2 FAIL），作为「审计确实承重」的前后对照保留。


**L1 下一步**（`v122_evidence/L1_NEXT_STEP_PACKAGE.json`）：七项**可机械化**前置条件在当前树上**全部为真**（F 前向通过、watch 完好、64 状态冻结并已评分、合同/profile 身份、两臂协议 10/10、独立 L1 审计 17/17、backbone 标签裁定）。唯一**需签署人决定**而非阻断的项：**C0 认证预算** —— L1 输出三个独立判定且 V3 可以 `NOT_RUN`（并记录认证器身份），回答 L1 机制问题**不需要**正式 C0 预算；只有在签署人希望 V3 表示**真实 C0 认证**时才需要先裁定预算。


L1 包的共享输入规则（同状态、B 池采样前构建、环境有序 legal 原序、`eps = 0`、`legal_order_first`、同合同与同信息边界、唯一声明差异 = `scorer_id`）、两臂（恢复的 F logits vs 登记的 `-incremental_distance`）、三判定、以及**资源与身份冻结基准**均已冻结；后者现在可以由**已实测的 F 成本**推导，而不再依赖 V1.9 采集计时。


**F−D-Geo 比较本身未被本阶段授权，也未启动。** 下一步唯一的人工决策点 = L1 正式比较的独立签署。

### 0.0000 本轮第五阶段：运行期源码依赖冻结闭环（V1.21，7/7）

审查发现 `IDENTITY_TARGETS` 的 14 条 watch 路径**漏登记**真实执行路径实际 import 的模块：
`d3_l1_v118_authorization.py`、`d3_l1_v120_execution_context.py`、`d3_l1_v116_forward_control.py`、`d3_l1_v112_phase_journal.py`；追查 `typed_bytes`（**逐位重复**判定的定义者）指向 `d3_l1_v18_readonly_audit.py`，**同类漏登记**。实测闭包为 **30 个源码文件**，不是 9 个。

**加固**：watch 集**不再手写，而是导出**。`scripts/data/d3_l1_v121_dependency_closure.py` 以执行路径的 seed 模块为起点，按 `import`/`from`（**含函数体内导入**）在本项目 `scripts/` 树内迭代到不动点求传递闭包；被排除的工具**逐条记录理由**；所有路径以**同一规范形式**报告。identity 模块再追加 4 项非源码资产（权重/数据/profile 以扩展根为基准、backbone 以仓库根为基准，**逐项显式指定根**）。`--readiness` 新增 **`runtime_watch_covers_closure`** 交叉检查：冻结集若漏掉闭包中任一文件即 `NOT_READY` 并列出缺项。**冻结与冻结校验来自同一处。**

**反例（`v121_evidence/D3_V121_FREEZE_CLOSURE_ACCEPTANCE.json` = 7/7 PASS）**：F1 四个漏登记模块**与 `typed_bytes` 定义者**均在导出 watch 集内且闭包干净（30 源码路径、无未解析 seed、无意外排除）；F2 V1.21 身份携带**完整 34 条** watch 集、逐路径 SHA 一致；F3 readiness `READY_FOR_USER_SIGNATURE` + `runtime_watch_covers_closure=true` + `real_F_forward_executed=false`；**F4/F5 每个漏登记模块各自隔离篡改 ⇒ NOT_READY、还原 ⇒ READY**（含 `readonly_audit`）；**F6 全量扫掠：34 条路径逐条篡改、逐条检出**（无路径未被检验）；F7 **仅传递可达**的文件也在冻结集内且其篡改被检出；F8 V1.18/V1.19/V1.20 证据目录**仍在且未被改写**。

**旧身份在新门禁下的正确行为**：V1.20 运行目录现报 `NOT_READY`、`runtime_watch_covers_closure=false`（冻结 14 / 导出 30）—— 正是「运行期源码变更必须产生新执行身份」，而不是静默通过。

**新执行身份** `d3l1-real-f-forward-20261015-v121` + 预登记 `…_V1_21.md`（**取代 V1.20 作为签署对象**）+ 授权模板；V1.18–V1.20 全部原件保留。`--readiness = READY_FOR_USER_SIGNATURE`（34/34 路径、0 漂移），模板与直接 `--worker` 均被真实入口拒绝，`ZERO_REAL_F_WORK_AUDIT` = **7/7**。回归：V1.16 6/6、V1.18 e2e 9/9、V1.19 25/25、V1.20 12/12。预算 60/10/180 s、64 状态、129 次调用不变。

**本阶段为冻结闭环的最后一轮**：不再新增非必要工程任务，交付可签署的 V1.21 执行证据包。

### 0.000 本轮第四阶段：执行入口绕过闭合（V1.20，12/12）

独立审查发现 `main()` 接受直接 `--worker`（`if args.worker: return worker(...)`），该路径**未过人类授权、未过一次性消费、未过监督器**，却能加载真实 F 权重并评分 —— blocker 1 的整套授权装置可被**一个参数**绕过；`--synthetic-command` 是第二个开口。

**加固**：监督器成为**唯一**可启动真实 F 工作的组件。人类批准凭据经**逐字段比对**并**一次性消费之后**，监督器才发布**执行上下文**（私有 256-bit secret + 一次性 nonce + run_id/目录绑定 + 已消费凭据摘要与回执路径 + HMAC(secret, 载荷)）。真实 worker 的**第一条动作**即校验该上下文，位于**一切加载之前**（不 import JAX、不读 checkpoint、不调用 scorer）：C1 文件可解析 / C2 secret 存在且私有 / C3 HMAC 通过 / C4 run 绑定 / C5 **一次性消费回执**存在且属本 run / C6 凭据**在消费后**仍哈希到同一摘要。**C5/C6 是关键**：单有 nonce 不能证明任何事（谁都能编），但只有监督器 `authorize()` 能造出、且绑定人类已批准凭据的回执，**无法由被闸门保护的进程自己伪造**。另加：**授权 + `--synthetic-command` 组合直接拒绝**，一次性凭据不得消耗在被替换进程上。

**反例（`v120_evidence/D3_V120_BYPASS_ACCEPTANCE.json` = 12/12 PASS）**：N1 直接 `--worker` REFUSED 且 0 MODEL_LOAD/0 WARMUP/0 槽行/0 评分；N2 上下文指向不存在文件 REFUSED；N3 guard 探针无上下文 REFUSED；N4 **手写上下文+自造 HMAC** REFUSED（`consumption_receipt_missing`）；N5/N6 签名后编辑（run_dir / 凭据摘要）REFUSED（`hmac_mismatch`）；N7 回执改成另一 run id REFUSED；N8 凭据在消费后被编辑 REFUSED；N9 `--synthetic-command` **指向真实 worker** ⇒ 0 真实 F 工件 + 监督器记录 `published=false` + guard 探针 REFUSED；N10 合法授权 + synthetic ⇒ REFUSED 且 **0 条消费回执**；**P1** 上下文**无法先于消费**发布，且 guard 在**独立进程**中接受监督器实际产出的上下文（什么都不加载）；**P2** 合成路径端到端 64 行 `ROUND_PASS` 且**未发布任何执行上下文**。

**如实记录平台限制**：`os.chmod` 在 Windows 上无法表达「仅所有者」（stat mode 由只读位合成，普通文件恒为 `0o666`），故 0600 检查**仅在 POSIX 生效**；Windows 侧私有性来自运行目录 ACL，以及 secret **既不进入命令行也不写入任何记录**。该限制在 P1 产物中显式标注。

**新执行身份** `d3l1-real-f-forward-20261015-v120` + 预登记 `…_V1_20.md`（**取代 V1.19 作为签署对象**）+ 授权模板（人工字段留空，被真实入口拒绝）；V1.18/V1.19 预登记与证据目录**未改写**。`--readiness = READY_FOR_USER_SIGNATURE`，`runtime_watch_closed = true`，`real_F_forward_executed = false`；`ZERO_REAL_F_WORK_AUDIT` = **7/7** `ZERO_REAL_F_WORK_CONFIRMED`。回归：V1.16 6/6、V1.18 e2e 9/9、V1.19 25/25。

### 0.00 本轮第三阶段：三个正式执行阻断闭合（V1.19，新执行身份）

独立审查在 V1.18 执行路径上发现三个阻断。**全部由可执行反例闭合**，证据
`v119_evidence/D3_V119_BLOCKER_ACCEPTANCE.json` = **25/25 PASS**；V1.18 预登记与
`v118_evidence/` 归档**未改写**。

| 阻断 | 问题 | 修复 | 反例 |
|---|---|---|---|
| **1 授权未绑定身份** | `supervise()` 只要求 JSON 可解析、`authorization_sha256` **可选**且只是文件自身摘要 ⇒ `{}` 即"合法授权"，且无法表达批准了什么 | 新增 `d3_l1_v118_authorization.py`：人类撰写的凭据须逐项声明 run_id / 预登记 / 权重 / 骨干 / 数据 / profile / 有效合同 / 状态绑定 / 状态顺序 / 前向目标摘要 / 依赖身份 / 三种调用次数 / 三个预算上限 / 输出目录 / `approved_by`+`approved_utc`+`approval_statement`，并与冻结身份**逐字段比对**；`credential_sha256` 覆盖**载荷**（不是整文件自哈希这种不可计算的定点）；**一次性消费**且回执落在**运行目录之外** | 13 条全部 REFUSED：缺失、空 `{}`、未填人批准的模板、错 run_id、错权重/骨干/数据/合同/状态绑定/目标摘要、错调用次数、放宽预算、错输出目录、封装后篡改、**重放**、预登记已变更 |
| **2 READY 语义不一致** | 监督器查**固定名** `READY.json`，worker 发布**唯一名** READY ⇒ 监督器**永远看不到就绪** | `run_contained` 新增可注入 `ready_probe`；**历史默认（`None`）= 固定名存在**，V1.16 接口与语义**逐字节不变**（6/6）；V1.19 传入绑定本运行目录、校验不可变 READY 事件（kind/run_id/warm-up 次数/评分摘要/时钟）的探针 | **B2-a** 批次运行 ~70 s（> 冻结 60 s 初始化上限、≪ 180 s 批上限）**未被误杀**且 `ready=true`、`ROUND_PASS`；**B2-b** 真实 61 s 初始化超限**被强杀**（0 READY、0 槽记录、`ROUND_REVOKED`）；**B2-c** 该路径**无**固定 `READY.json` |
| **3 FAILED_CHECKS 不终止** | 失败槽只记状态后**继续循环**，与 V1.18 预登记 §5 不符 | 首个失败槽**立即停止**：先落盘该槽记录（含 `failed_checks` + 冻结词表内 `failure_reason`），再发布 `FIRST_FAILURE`，再发布**尾部身份记录**（使 V1.16 撤销可裁决），以**非零码**退出 | **B3-a..d**（finite / shape / repeat / len）：**恰好 1 条行记录**、**63 条 `NOT_ATTEMPTED_AFTER_TERMINATION`**、`ROUND_REVOKED`、verdict `BLOCKED`、`rows_qualified=0`，且由**生产 worker 自身检查集**判定 |

**另修一处会让上述修复被绕过的门禁缺口**：`--readiness` 原先只比对**资产哈希**（权重/数据/profile/
合同），**改代码不会失败**。新增 **runtime watch closure**：从运行目录 `IDENTITY_TARGETS` 逐路径重算，
漂移即 `NOT_READY` 并把该运行身份标为**过期**。因此 V1.18 运行目录在修复后正确报 `NOT_READY`
（漂移 2 条：runner + run_identity）——**这正是"运行期源码修改必须产生新执行身份"的行为**。

**自身发现并修正的两处缺陷（留档）**：① 首版 `publish_new` 未给 JSON 记录盖 `published_mono`，
读者排序退化为**文件名序**，而运行目录**合法地**存有同族多条不可变记录（从不覆盖）⇒
`zero-F` 审计一度读到**错误的** readiness 记录；已让 `publish_new` 统一盖时间戳。
② 该块修复改动了受 watch 的运行时路径，故 **V1.19 身份重新冻结**（新 RUN_IDENTITY /
FORWARD_TARGETS / IDENTITY_TARGETS / 授权模板）。

**新执行身份与签署对象**：`run_id = d3l1-real-f-forward-20261015-v119`；
预登记 `A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_19.md`（**取代 V1.18 作为签署对象**）；
授权模板 `v119_evidence/AUTHORIZATION_TEMPLATE_V119.json`（三个 `approved_*` **留空**，
由**真实入口**拒绝：`AUTHORIZATION REFUSED: no_human_approval`）。
**`--readiness` = `READY_FOR_USER_SIGNATURE`**，`runtime_watch_closed = true`（14/14），
`real_F_forward_executed = false`。

**真实 F 执行次数仍为 0 的独立证明**：`v119_evidence/ZERO_REAL_F_WORK_AUDIT.json` = **7/7 PASS** →
`ZERO_REAL_F_WORK_CONFIRMED`。该审计**不导入** runner / 监督器 / scorer，只数"只有真实前向才会留下的工件"：
槽行记录 0、原始评分 0、相位记录 0、MODEL_LOAD 0、WARMUP 0、尾部身份 0、监督结果 0、ROUND 0，
且本 run_id 的**授权回执为 0**、模板人工字段为空、readiness 记录被**重判**而非采信。

### 0.01 本轮第二阶段（G0 反例 + 三份预登记草案）

在 §0.1 之后继续，全部已 push：

| commit | 内容 |
|---|---|
| `3b6afe75` | **G0 残余风险反例可执行化**：8 个最小反例跑在**当前代码**上，区分 `FIX_CONFIRMED`（风险已关闭）与 `RISK_CONFIRMED_OPEN`（风险经执行确认**仍开放**）⇒ **7 FIX_CONFIRMED / 1 RISK_CONFIRMED_OPEN** |
| `3993e34e` | **E2 训练预登记草案**（`A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md`，**不执行**） |
| `b336f6e6` | **`P_after_install` 四字段前瞻采集预登记草案**（**不执行**） |
| `4c3fb5bb` | **L2/L3 层级升级预登记草案**（**不执行**） |

**G0 反例结论（逐条，可复核 `D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json`）**

| 风险 | 反例 | 判定 |
|---|---|---|
| A-2 | 无认证器的事务 | **FIX_CONFIRMED**：`no_certifier` 拒绝、0 次安装 |
| A-2 | 认证 A 安装 B | **FIX_CONFIRMED**：`certified_plan_identity_mismatch` 拒绝 |
| A-4b | 安装后改动调用方计划对象 | **FIX_CONFIRMED**：台账存**值拷贝**，记录不被改写 |
| A-4b | 计划身份不可计算 | **FIX_CONFIRMED**：拒绝，不以 `None` 身份安装 |
| A-5d | 无 `rollback_fn` | **FIX_CONFIRMED**：记 `rollback_unverified`，**不**记"已恢复" |
| A-5d | 有 `rollback_fn` | **FIX_CONFIRMED**：`committed_state_restored=True`，恰好调用 1 次 |
| A-5b | 重复 `finish` | **FIX_CONFIRMED**：第二次 `already_finished` 拒绝，总数 1 次安装 |
| **A-3b** | 掩码含**已提交**客户 | **RISK_CONFIRMED_OPEN**：采样器**照做**（计划中已提交客户被移动），适配器**无**已执行前缀守卫；真正的守卫在 `VehicleState` + B 池 |

**如实留档的自身缺陷**：A-3b 第一版反例把探针绕经安装门，产生了**无关失败导致的假"detected"**；
该次运行**不作为证据保留**，已改写为**直接机制检查**，并在产物里把 A-3b 单列为
`RISK_CONFIRMED_STILL_OPEN`，**不计入已关闭项**。

**三份草案的实质内容**

- **E2**：单因素 = backbone 初始权重来源。新增 **E2-7（已对源码核实，不是假设）**：
  `mpre_trained.py` 的 `c_adapter` / `delta_head` 都是 `MLP(..., zero_last=True)`，末层
  kernel/bias 为 `zeros_init()`、隐藏层取自 rngs 流 ⇒ 若 E2-rand 只随机化 backbone 而两层初始化
  不同，则"预训练 vs 随机"会与"零末层 vs 随机初始化""seed A vs seed B"**同时**变化。规格**强制**
  两层在两侧使用**同一初始化器与同一 seed**，并要求**参数名/形状/dtype 逐项相等**的机械证据。
  同时重述 `--seed` **不是** E2（backbone 仍载入预训练）。
- **`P_after_install`**：回执通道已闭合（7/7），`P_after_install` **只**允许来自通过独立重读的回执，
  **禁止**由 `P_certified`/`P_proposed` 推算；并如实列出**未闭合**三项：环境自身 `_plan_and_commit`
  不可观测、`A-3b` 仍开放、状态级回滚未被回执证明。
- **L2/L3**：三层不可跳级逐层写死前置门；质量口径**三列分离**（`candidate`/`accepted`/`closed_loop`，
  `candidate` **不得**称"部署"）；L2 完整 C0 认证**依赖用户裁定 B**；L3 需**新鲜开发日** + 等预算
  + 冻结统计计划。

**仍未完成、且本轮未做**：2×2 统一场景输入适配器**设计**（最小清单第 7 项，纯设计）；
32 状态确定性重采集（最小清单第 3 项，本身是一次新执行，需另行授权）。

### 0.1 本轮第一阶段（V1.18 + L1 + D8 + C0 搜索）**全部完成**

自 `f73d2c88` 续接，**V1.18 达到 `READY_FOR_USER_SIGNATURE`，L1 工程协议闭合**。三项已 push：

| commit | 内容 |
|---|---|
| `e8907eb` | **V1.18**：隔离 run_id、真实 F 证据推导的预算冻结、V1.14 相位日志 + V1.16 整轮撤销接入 F worker、9 场景合成端到端验收（9/9）、独立只读复核器（9/9 + 监督自检 9/9）、七要素执行版预登记、`--readiness` 动态计算 |
| `8de9a3e` | **L1 两臂协议**：共享输入单因素机器可检验、三判定分离、六个合成反例（10/10）、可重放验收器、独立只读审计 17/17、七要素执行版预登记 |
| `eeb8dbec` | **D8 持久安装回执 + 独立重读**（G0 A-5c 关闭，7/7）；不动 `d3_geo_adapter.py` 一个字节 |

**V1.18 现状（可直接复核）**

- `run_id = d3l1-real-f-forward-20261015-v118`，与 V1.9 / V1.11 文本不相交；
  绑定 64 状态（台账发布顺序，不重排）+ 模型 / 骨干 / 数据 / profile / 基础+有效合同 /
  14 依赖模块身份；运行期 watch 集 = 10 代码文件 + ckpt + backbone + npz + profile = **14 路径**。
- 调用次数 **129**（1 warm-up + 64×2，沿用 V1.11 协议）；预算**由 V1.11 真实 F 实测推导**：
  槽跨度 max/median = 0.344 / 0.234 s（n=22）、模型初始化 6.344 s、首次编译 6.781 s、
  22 状态批墙钟 19.562 s ⇒ 全 64 状态外推 **70.03 s**；冻结上限 **60 / 10 / 180 s**
  （余量 4.57× / 29.07× / 2.57×，**不放宽**）。V1.9 的 F-free 计时**只作溯源**，明确排除。
- V1.11 根因（固定目标名 `STAGE.json` 被读者持有 ⇒ `WinError 5` ⇒ 22/64）**结构性消除**：
  每次发布到**从未存在过**的新目标名并**写后读回校验**；读取有界，持久拒绝与损坏记录均为**硬错误**。
- 验收：`v118_evidence/SYNTHETIC_END_TO_END.json` **9/9 PASS**（读者全持句柄、瞬时共享拒绝、
  损坏记录、真实截止强杀、真实进程树强杀、尾部身份篡改、尾部缺失、63/64 行、评分长度坏行）；
  V1.16 回归 **6/6**。
- 独立复核器：`v118_evidence/INDEPENDENT_READONLY_VERIFIER.json` = `VERIFIED_NOT_EXECUTED` 9/9
  （真实 F 未跑，如实不是 PASS），
  `..._SUPERVISED_SELFTEST.json` = 64 合成行**独立判定器 64/64 与生产判定器逐行一致**。
- `--readiness`：**`READY_FOR_USER_SIGNATURE`，`missing_before_signature = []`**，
  同时如实记录 `real_F_forward_executed = false` —— 前置条件已备齐，待签署的就是那一次执行。
- 唯一待批科学执行：**V1.18 真实 F 64 状态前向**（新 run_id，**禁止拼接 V1.11 缺失的 42 条**）。

**L1 现状（可直接复核）**

- `scripts/simulation/d3_l1_two_arm_protocol.py`：共享输入（单因素三重机械检查：同一 state 对象、
  运行前后 state 摘要三处相等、`DECLARED_VARYING_FIELDS = ('scorer_id',)`）；
  `legal_fingerprint` 同时用于共享身份与每臂逐步轨迹（避免两套编码互相比较）；
  V1 / V2 / V3 **三个独立判定**；空池 = `NOT_APPLICABLE_EMPTY_MASK`（**不是**"一致"）；
  认证器缺失时 V3 = `NOT_RUN`（既非 True 也非 False）。
- 验收 `l1_gates/A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json` **10/10**；空池、`score_len`、
  `score_shape`、分叉后 legal、认证失败、超时六类反例全部命中；
  `no_legal` 如实记为 **`NOT_EXERCISED`**（64 状态中每个可见订单都至少 1 个可行插入，
  该分支在本数据集不可自然触发，**不冒充通过**）。
- 独立审计 `l1_gates/A1_D3_L1_TWO_ARM_INDEPENDENT_AUDIT.json` **17/17**：
  **不导入**协议 / geo_adapter / 验收器，只读 trace JSON，独立重算 V1/V2/V3 并与记录值对拍。
- L1 正式比较仍 `BLOCKED`，**唯一硬前置 = V1.18 真实 F 前向**（不是协议缺失）。

**D8 / C0 本轮新增**

- D8：`scripts/simulation/d3_install_receipt.py` + `d3_install_boundary.py`（**再导出**冻结的
  `pre_install_reverify` 并以对象身份证明未被重实现）+ 7 场景验收 7/7。
  回执**只在 ledger 确实安装恰一个计划时**写出，存全部 `plan_hash` 所需字段以便读者**自行重算**
  （不靠声明）；R1 由**独立进程**重读。**仍不证明**：环境自身 `_plan_and_commit` 的提交、
  状态级回滚、已安装计划的品质/能耗性质。
- C0 预算：`C0_BUDGET_PROVENANCE_SEARCH.json` —— 有界穷举（脚本 / D3 证据 / data / docs /
  research / CC_Compare 审计分列），**184 命中：DOC_MENTION 165 / TEST_LOCAL 12 /
  NON_PORTABLE_FIXTURE 7 / 协议可达候选 0**，判定
  **`SEARCH_EXHAUSTIVE_NO_REGISTERED_PROTOCOL_BUDGET_FOUND`**（搜索半边已闭合）。
  **勘误并留档**：第一版分类器按"上下文含 protocol 一词"判类，把 29 处**文档提及**误升为
  `REGISTERED_PROTOCOL_BUDGET`；已在报告 `corrected_from` 中原样保留该误判并给出更正后的分类器。
  预算裁定仍 **`AWAITING_USER_BUDGET_RULING`**（代理不得自定）。

### 0.2 上一轮收束（V1.18(A)，commit `52362db`）

- V1.18(A) 归档资产冻结 **4/4 PASS**：`STATE_LEDGER.jsonl` 的 `PUBLISHED` 行 64 条按发布顺序、
  64/64 带 `state_hash`、128/128 snapshot+validation 原件 SHA 现场复算一致、完整性位全真、
  `run_id` 唯一。证据 `v118_evidence/D3_V118_FREEZE_CHECK.json`。
- 同时实测采集预算（**仅溯源**）：`slot_span_wall_s` 0.625/1.422/3.328 s（n=64）、
  `day_rollout_wall_s` max 0.519 s（n=16）。
- 事故与恢复（如实记录）：一次工作树补丁损坏了该脚本；已从 `1b2bc9b` **逐字节恢复**并只重放定位器改写。
- 本轮已把该节列出的五项缺失**全部闭合**（见 §0.1）。

## 1. 已消费的正式实验次数（**不得复用、不得重跑**）

| 实验 | 次数 | 状态 | 证据 |
|---|---|---|---|
| F-free 一次性状态采集（64 槽位） | **1（已用尽）** | 64/64 发布，独立验收 1498/1498，未补采 | `v19_evidence/`、`l1_gates/A1_D3_L1_FORMAL_64_INDEPENDENT_AUDIT_V19.json` |
| 真实 F 前向批次 V1.11 | **1（失败，已用尽）** | 22/64 完成、42 未完成、整轮 `BLOCKED`；**禁止拼接补跑** | `v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json` |
| 服务器 F-free 计时样本 | 1（已用尽） | `recorded_Q_server=NOT_PASSED`，事后身份 `MATCH` | `l1_gates/A1_D3_L1_SERVER_TIMING_*.json` |
| 真实 F−D-Geo 机制比较（L1） | **0（未授权）** | 协议已闭合，**待 V1.18 通过后另行签署** | `l1_gates/A1_D3_L1_TWO_ARM_*.json` |
| E2 训练 | **0（未授权）** | — | — |
| L2/L3、40 日批次、最终测试 | **0（未授权）** | — | — |

## 2. 当前真实阻断（按优先级）

1. **V1.18 真实 F 64 状态前向**：工程资格已 `READY_FOR_USER_SIGNATURE`，**唯一待批的科学执行**。
   未执行前 L1 正式比较不可启动。
2. **L1 正式机制比较**：协议与反例已闭合；硬前置 = 第 1 项。
3. **C0 正式预算**：搜索半边已穷举闭合（无注册协议预算），裁定仍 `AWAITING_USER_BUDGET_RULING`。
4. **D8**：持久回执已实现（A-5c 关闭）；**仍阻断**于 G-L 状态认证、G-C 真实写入唯一性、
   品质保守保证，以及"环境自身提交不可观测"。
5. **E2**：前置条件未冻结，未训练。
6. **L2/L3 与 A1/A2/A3**：未启动。

## 3. 已完成并可独立复核的工作包

| 工作包 | 结论 | 证据 |
|---|---|---|
| P0-A 逐位比较器 | 16/16 反例门通过 | `l1_gates/A1_D3_L1_BITWISE_PARITY_GATE.json` |
| P0-B 因果扰动（方案 B） | `DIST_MAT_GENERATION_CONTRACT = VERIFIED`（16/16 天逐位） | `l1_gates/A1_D3_L1_P0B_CAUSAL_PERTURBATION.json` |
| P0-C 采集工程 | 64/64 发布 + 1498/1498；V1.8 起 6 套门 | `l1_gates/A1_D3_L1_P0C_*.json` |
| V1.16 监督器整轮撤销 | 6/6 合成验收 | `v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json` |
| V1.19 G0 可执行差分 | 8/8；G0-01–G0-05 接口层闭合 | `v119_evidence/D3_V119_G0_INVARIANT_DIFFERENTIAL.json` |
| V1.20 组合认证器**真实执行** | `installable=false`；品质缺口归属**未隔离** | `v120_evidence/D3_V120_READINESS.json` |
| **V1.18 F 前向工程资格** | `READY_FOR_USER_SIGNATURE`；9/9 + 复核 9/9 | `v118_evidence/SYNTHETIC_END_TO_END.json`、`INDEPENDENT_READONLY_VERIFIER*.json` |
| **L1 两臂协议** | 10/10 + 独立审计 17/17 | `l1_gates/A1_D3_L1_TWO_ARM_*.json` |
| **D8 持久安装回执** | A-5c 关闭；7/7 | `B3A_receipt_evidence/D3_INSTALL_RECEIPT_ACCEPTANCE.json` |
| **C0 预算溯源穷举** | `SEARCH_EXHAUSTIVE_NO_REGISTERED_PROTOCOL_BUDGET_FOUND` | `C0_BUDGET_PROVENANCE_SEARCH.json` |
| **G0 残余风险反例可执行化** | 7 FIX_CONFIRMED / 1 RISK_CONFIRMED_OPEN（A-3b 仍开放） | `D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json` |
| **E2 预登记草案** | 规格已交付（含 E2-7 零末层强制项，源码已核）；**未训练** | `A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md` |
| **`P_after_install` 采集预登记草案** | 规格已交付；回执通道 7/7，**未采集** | `A1_D3_L1_P_AFTER_INSTALL_COLLECTION_PREREGISTRATION_V1_18.md` |
| **L2/L3 升级预登记草案** | 规格已交付；**未执行** | `A1_D3_L1_L2_L3_ESCALATION_PREREGISTRATION_V1_18.md` |

## 4. 重要失败与真实分母（原件保留，未改写）

- V1.20 = 3 行 2 PASS 1 FAIL：漏必填 kwargs → `sys.path` 缺 `scripts/evaluation` → 归属未隔离。
- V1.19 = 8/8，但含两处**解释**缺陷 + `B_BUDGET` 匹配错误 → V1.20 勘误；本轮 C0 搜索在报告
  `corrected_from` 中再次留档一版**分类器**误判（29 处文档提及被误升为注册预算）。
- V1.11 F 前向 22/64：Windows 进度文件替换 `PermissionError`；V1.18 起以"新目标名 + 写后校验 +
  有界读取"结构性修复（9/9 场景 + V1.16 6/6 回归）。
- `reconcile_rows` 本轮新增可选 `rows_dir` / `row_glob`（默认值复现历史行为，V1.16 6/6 通过）；
  **一次实现错误（默认 `rows_dir` 从 `'scores'` 误改为 `out`）已被 V1.16 回归当场抓住并修正**，
  未隐藏。
- **A-3b 反例第一版**把探针绕经安装门，得出**无关失败导致的假"detected"**；该次运行**不作为证据保留**，
  已改写为直接机制检查，并把 A-3b 在产物中单列为 `RISK_CONFIRMED_STILL_OPEN`。
- **C0 搜索第一版分类器**按"上下文含 protocol 一词"判类，把 29 处**文档提及**误升为注册预算；
  已在报告 `corrected_from` 中原样保留并给出更正分类器。
- 较早：P0-B 首版把 float64 与 float32 计算路径差异误判为"外部路网矩阵"（V1.4 已更正）。

## 5. 下一个可自动执行任务（无需用户授权）

1. **V1.18 真实 F 前向**（唯一待批；未获签署前不得执行）——签署后：
   `python scripts/simulation/d3_l1_v118_forward_runner.py --supervise --out-dir <run_v118> --authorization <file>`，
   随后 `--readiness` 与 `d3_l1_v118_independent_verifier.py` 复验。
2. **L1 正式机制比较**（V1.18 通过 + L1 预登记签署后）；运行预算须按 L1 预登记 §6 用**真实 F 实测**
   冻结，不得引用 V1.9 计时。
3. **D8 受限三问**：G-L 状态认证；G-C 真实写入唯一性；品质保守保证（回执已不再是缺口）。
4. **E2/L2/L3 与 A1/A2/A3 预登记草案**（不训练、不跑批次）。
   —— **已完成**：E2 → `A1_D3_L1_E2_TRAINING_PREREGISTRATION_V1_18.md`；
   `P_after_install` → `A1_D3_L1_P_AFTER_INSTALL_COLLECTION_PREREGISTRATION_V1_18.md`；
   L2/L3 → `A1_D3_L1_L2_L3_ESCALATION_PREREGISTRATION_V1_18.md`（**均未执行**）。
5. **2×2 统一场景输入适配器设计**（最小清单第 7 项，纯设计；Role C 现绕 cond_hist 场景流，
   接 MaskCO 须走**一条**决策代码路径）。
6. **32 状态确定性重采集**（最小清单第 3 项）：**本身是一次新执行**，须先获授权并冻结清单/预算；
   合同/profile 身份须在该预登记内显式记录（本轮已把 `effective_contract_hash` 与 `profile_hash`
   写入 V1.18/L1/E2 身份块，不再以"缺口"表述）。
7. 若需扩充 `no_legal` 覆盖：需先有**非活跃**且无可行插入的订单来源，并保持 strict-online 因果
   （本数据集不可自然触发，已如实记录）。

## 6. 需要用户签署的事项（唯一清单）

1. **V1.19 真实 F 前向批次**（64 状态、`run_id=d3l1-real-f-forward-20261015-v119`、与 V1.11 完全隔离、
   禁止拼接 42 条）——**唯一待批的科学执行**。签署方式：
   ① 填写 `v119_evidence/AUTHORIZATION_TEMPLATE_V119.json` 的 `approved_by` / `approved_utc` /
   `approval_statement`；② 用 `d3_l1_v118_authorization.content_digest` 重新封装 `credential_sha256`；
   ③ 执行 `--supervise --authorization <填好的凭据> --preregistration <V1.19 预登记>`（**一次性消费**）。
   **机器不能自我批准**：模板本身会被拒绝。
2. **C0 正式预算**（`AWAITING_USER_BUDGET_RULING`，代理不得自定；搜索半边已穷举闭合）。
3. **L1 正式机制比较**（在 V1.19 通过、L1 预登记签署后）。
4. E2 训练、L2/L3、40 日批次与最终测试（各自独立授权）。

## 7. 硬约束（每次执行前自检）

`MASKCO_code/` 永不修改；不改冻结数据/冻结树/历史 FAIL 原件/用户正式裁决；扩展代码不加 `__init__.py`；
strict-online 只用已揭示信息；已执行前缀与承诺冻结、每单恰好服务一次；C0 硬认证超预算不提交；
`distance_cost` 与 `coldchain_cost` 严格分离；效用 = 收入 − 燃油 − 拒绝损失（品质不货币化）；
开发证据与最终测试隔离；正式运行须先有七要素预登记 + 身份冻结 + 资源门 + 授权；
不得事后调判据/加次数/挑结果；不得用 `PENDING`/`None`/无限预算/历史 CAL 夹具/synthetic-only 冒充正式值；
不得把合成 64 行 PASS 冒充真实 F 64 状态 PASS；**未测试的分支记 `NOT_EXERCISED`，不得记 PASS**。

## 8. 汇报单位

以**完整工作包**或**真正需要人工决定的科学关口**为汇报单位；不以单个脚本/测试/commit 为单位；
不为形成汇报而中断仍可自主完成的工作。汇报固定回答六问（见总指令 §10）。
