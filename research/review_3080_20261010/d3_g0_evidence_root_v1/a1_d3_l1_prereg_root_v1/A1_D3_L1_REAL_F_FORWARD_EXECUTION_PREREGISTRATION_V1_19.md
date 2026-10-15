# V1.19：真实输入 F 前向执行版预登记（七要素 · 阻断修复版）

- 轮次：`A1-D3-L1` / `V1.19 real-F forward execution qualification (blocker-fixed)`
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行任何真实 F 前向**）
- 绑定运行：`run_id = d3l1-real-f-forward-20261015-v119`
- 取代关系：本文件**取代** `A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_18.md` **作为签署对象**。
  **V1.18 预登记与 `v118_evidence/` 全部工件原样保留**（`4a6d207` 归档证据，未改写、未删除）。
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 为什么需要 V1.19：三个正式执行阻断

独立审查在 V1.18 的执行路径上发现三个阻断，全部由**可执行反例**确认并修复
（证据 `v119_evidence/D3_V119_BLOCKER_ACCEPTANCE.json`，**25/25 PASS**）：

| 阻断 | 审查位置 | 问题 | 修复 |
|---|---|---|---|
| **1 授权凭据未绑定执行身份** | `d3_l1_v118_forward_runner.py:359–366` | 只要求 JSON 可解析；`authorization_sha256` 是**可选**且只是文件自身摘要 ⇒ 一个 `{}` 就是合法"授权"，且**无法表达它批准了什么** | 新增 `d3_l1_v118_authorization.py`：凭据必须逐项声明 run_id / 预登记 / 权重 / 数据 / 合同 / 64 状态顺序 / 调用次数 / 预算 / 输出目录 / 人类批准，并与冻结身份**逐字段比对**；一次性消费并留可独立审计回执 |
| **2 READY 语义与监督器截止不一致** | `d3_l1_v116_forward_control.py:107` vs `d3_l1_v118_forward_runner.py:241` | 监督器查**固定名** `READY.json`，worker 却发布**唯一名**的 READY ⇒ 监督器**永远看不到就绪** | 监督器新增可注入 `ready_probe`（V1.19 传入校验不可变 READY 事件的探针）；**V1.16 历史默认接口与语义逐字节不变**（6/6 回归通过） |
| **3 FAILED_CHECKS 不终止后续评分** | `d3_l1_v118_forward_runner.py:266–319` | 失败槽只记状态后**继续循环** ⇒ 与预登记 §5"失败即终止、后续 NOT_ATTEMPTED"不符 | worker 在首个失败槽**立即停止**，发布 `FIRST_FAILURE` 与尾部身份后以非零码退出；后续 63 槽明确 `NOT_ATTEMPTED_AFTER_TERMINATION` |

另修一处**使上述修复可被绕过**的门禁缺口：`--readiness` 原先只比对**资产哈希**（权重/数据/profile/合同），
代码编辑**不会**使其失败。现新增 **runtime watch closure** 检查：从运行目录的 `IDENTITY_TARGETS`
逐路径重算，任一漂移即 `NOT_READY`（并把该运行身份标为过期、要求新版本）。

**结果**：V1.18 运行目录在修复后的源码下 `--readiness` 正确变为 `NOT_READY`（漂移 2 条：
runner + run_identity），**这正是"运行期源码修改必须产生新执行身份"的行为**，而不是静默通过。

---

## 1. 冻结身份（与 V1.18 相同，逐条实测）

| 类别 | 值 | 实测 |
|---|---|---|
| F 权重 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` | `f5acf83f…903a7f5` ✅ |
| backbone | `MASKCO_code/ckpts/cvrp100.ckpt` | `03615b08…` ✅ |
| 标签 | `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED` | 强制保留 |
| backbone 标签 | `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / HISTORICAL_BACKBONE_SHA_UNVERIFIED` | 强制保留 |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` | `66573ddc…` ✅ |
| profile / 合同 | `63e56212…` / 有效合同 `edc1b2ea…`（基础 `32124bfd…`） | ✅ |
| 64 状态 | 台账**发布顺序**，逐状态 `state_hash`，与 manifest 零分歧 | ✅ |
| 前向目标 | `FORWARD_TARGETS` `targets_sha256` | `37e23f7567e3391a…` ✅ |
| 运行期 watch | 代码 + ckpt + backbone + npz + profile | 逐路径起止 SHA 闭合（新增 **runtime watch closure** 前置检查） |

## 2. 调用次数与预算（**与 V1.18 完全相同，未放宽**）

```
warmup_calls = 1 ; calls_per_state = 2 ; n_states = 64 ; max_forward_calls = 129
initialize_compile_s = 60.0 ; slot_span_s = 10.0 ; batch_s = 180.0
```

推导依据仍是 **V1.11 真实 F 实测**（槽跨度 max 0.344 s、初始化 6.344 s、首次编译 6.781 s、
22 状态批墙钟 19.562 s ⇒ 全 64 状态外推 70.03 s）。**V1.19 不放宽任何一项**；
B2-a 反例即用**冻结的 60/180 s** 证明"批次超过初始化上限但未超批上限"**不会**被误杀。

## 3. 授权凭据规格（阻断 1 的签署载体）

凭据是一份**人类撰写**的 JSON，必须逐项声明（缺一即拒绝）：

```
schema, run_id, preregistration_sha256,
f_ckpt_sha256, backbone_sha256, data_sha256,
profile_file_sha256, profile_hash, effective_contract_hash,
state_binding_sha256, state_order_sha256, forward_targets_sha256,
dependency_identity_sha256,
warmup_calls, calls_per_state, max_forward_calls,
budget_initialize_compile_s, budget_slot_span_s, budget_batch_s,
output_dir,
approved_by, approved_utc, approval_statement, scope,
credential_sha256            # 覆盖上述全部内容（不含自身）的载荷摘要
```

生成流程（**机器不能自我批准**）：

1. `--write-authorization-template` 产出**仅有绑定字段**的模板，三个 `approved_*` 字段**留空**；
2. **人类**填写 `approved_by` / `approved_utc` / `approval_statement`；
3. 用 `d3_l1_v118_authorization.content_digest` 重新封装 `credential_sha256`（该摘要覆盖载荷，
   编辑一个字节即失配；**不使用**"整文件自哈希"这种无法计算也无法校验的结构）；
4. 执行时 runner 逐字段比对冻结身份，然后**一次性消费**并在**运行目录之外**留下
   `authorization_consumption_<content_digest>.json`（含 run_id、run_dir、批准人、消费时刻、pid）。

**fail-closed 覆盖（反例逐条）**：缺失文件、空 JSON `{}`、仅有模板（无人批准）、错 run_id、
错权重/骨干/数据/合同/状态绑定/前向目标摘要、错调用次数、放宽预算、错输出目录、
封装后篡改、**重放同一凭据**、预登记文件已变更 —— **13 条全部 REFUSED**。

## 4. READY 语义（阻断 2 的签署载体）

- worker 在初始化与首次编译完成后发布**不可变、唯一名**的 READY 记录
  （`kind=initialization_and_compilation_finished`、`run_id`、`n_warmup_calls=1`、
  `warmup_scores_sha256`、`mono`）；
- 监督器通过 **`ready_probe`** 读取并**校验**该记录（缺字段即视为未就绪），
  probe **绑定到本运行目录**，不会读到别的运行的 READY；
- **V1.16 历史默认接口不变**：`ready_probe=None` 时仍是"固定 `READY.json` 是否存在"，
  V1.16 验收 **6/6** 通过；
- 反例：**B2-a** 批次运行 ~70 s（> 60 s 初始化上限、≪ 180 s 批上限）**未被误杀**且
  `ready_state.ready = true`、`ROUND_PASS`；**B2-b** 真实初始化 61 s 超限**被强杀**，
  `reason = initialization_compile_deadline`、`tree_active_after = 0`、0 条 READY、0 条槽记录、
  整轮 `ROUND_REVOKED`；**B2-c** 该路径下**不存在**固定 `READY.json`（证明固定名探针会永远"未就绪"）。

## 5. 逐槽失败终止语义（阻断 3 的签署载体）

任一槽的 `finite` / `shape` / `score_len`（长度≠有序 legal 长度）/ `bitwise-repeat` 等检查失败
⇒ **该槽为终止条件**：

1. 失败槽记录**先落盘**（含 `failed_checks` 与冻结词表内的 `failure_reason`）；
2. 立即发布 `FIRST_FAILURE`（含 `row_index` 与"终止、后续不得评分"的策略声明）；
3. **不再调用任何后续状态的 F scorer**，循环退出；
4. 仍发布**尾部身份记录**（`ended_because=slot_failure`），使 V1.16 撤销可裁决；
5. worker 以**非零码**退出；监督器把整轮判为 `ROUND_REVOKED`，verdict `REAL_F_FORWARD_ENGINEERING_BLOCKED`；
6. 后续 63 槽明确记 `NOT_ATTEMPTED_AFTER_TERMINATION`。

反例（**B3-a..d**，四类缺陷各一）：**恰好 1 条行记录**、**63 条 NOT_ATTEMPTED**、
`ROUND_REVOKED`、`verdict=BLOCKED`、`rows_qualified=0`，且**由生产 worker 自身的检查集**（而非
测试断言）判定失败。**不改变任何已冻结的科学判据**。

## 6. 执行边界（签署后仍受约束）

**授权**：以 `run_id = d3l1-real-f-forward-20261015-v119` 在上述冻结身份下执行**一次**
64 状态真实 F 评分接口检查（最多 129 次前向），结论仅限"真实输入下 F 评分接口工程可用性"。

**不授权**：F−D-Geo 机制比较、E2 训练、L2/L3、40 日批次、最终测试、任何安装/写入真实状态、
任何效用或显著性主张。**执行一次即消费**：PASS/FAIL 都保留原件，不重跑、不补样；
**禁止拼接 V1.11 缺失的 42 条**。

## 7. Claim limits

1. 本文件是**规格 + 工程资格**；产出时真实 F 前向调用数 = **0**。
2. 通过只支持"真实输入下 F 评分接口工程可用"，**不**支持算法优势、学习增益或在线效用。
3. backbone 训练期身份未证；即使通过也只能声明"工程验证"一类结论。
4. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
5. 工程上限（60/10/180 s）**不是** L3 在线发布门，**不得**被引用为满足 8+2 秒。
6. 授权凭据是**人类批准**的载体；`--write-authorization-template` 产出的模板**本身不是授权**
   （三个 `approved_*` 字段留空且被拒绝）。
7. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
