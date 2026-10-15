# V1.20：真实输入 F 前向执行版预登记（七要素 · 执行入口加固版）

- 轮次：`A1-D3-L1` / `V1.20 real-F forward execution qualification (entry-hardened)`
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行任何真实 F 前向**）
- 绑定运行：`run_id = d3l1-real-f-forward-20261015-v120`
- 取代关系：本文件**取代** `…_V1_19.md` **作为签署对象**。
  **V1.19 与 V1.18 的预登记、`v119_evidence/`、`v118_evidence/` 全部原样保留**（未改写、未删除）。
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 本次闭合的第四个阻断：执行入口绕过

独立审查发现 `main()` 接受直接 `--worker`：

```python
if args.worker:
    return worker(args.out_dir)          # 未过授权 / 未过一次性消费 / 未过监督器
```

这条路径能进入**真实 F 权重加载与评分**，而 blocker 1 安装的整套授权装置可被**一个参数**绕过。
`--synthetic-command` 是第二个开口：它替换 worker 进程，因此把它指向真实 worker 就等于在
合成harness（故意不带授权）下启动真实 F 工作。

### 加固：执行上下文（`scripts/simulation/d3_l1_v120_execution_context.py`）

监督器是**唯一**可授权真实 F 工作的组件。人类批准的凭据经**逐字段比对**并**一次性消费之后**，
监督器才发布**执行上下文**：

```
执行上下文 = 私有 256-bit secret（0600 文件，永不进入命令行或任何记录）
           + 一次性 nonce
           + run_id、运行目录、监督器 pid、创建时钟
           + 已消费凭据的内容摘要与消费回执路径
           + HMAC(secret, 规范化载荷)
```

真实 worker 的**第一条动作**就是 `validate_execution_context(...)`，它位于**一切加载之前**
（不 import JAX、不读取 checkpoint、不调用任何 scorer）：

| 条件 | 内容 |
|---|---|
| C1 | 上下文文件存在且可解析 |
| C2 | secret 文件存在且为监督器签名所用的那一个（POSIX 下另查 0600；Windows 无该位） |
| C3 | HMAC 校验通过 ⇒ 上下文非手写、非事后编辑 |
| C4 | run_id 与运行目录绑定 |
| C5 | **一次性消费回执**存在、属于本 run、且记录的凭据摘要与上下文一致 |
| C6 | 凭据文件**在消费之后**仍哈希到同一摘要（未被编辑） |

**C5/C6 是关键**：单有 nonce 不能证明任何事（谁都能编一个）；但只有监督器的 `authorize()`
才能造出、且绑定到人类已批准凭据的回执，**无法由被闸门保护的进程自己伪造**。因此得到真正要的性质：
**真实 F 工作只在"存在一份针对本运行的、人类批准的、身份绑定的、已一次性消费的凭据"时运行。**

另加：**授权与 `--synthetic-command` 组合被直接拒绝**——一次性凭据不得消耗在被替换的进程上。

### 反例（`v120_evidence/D3_V120_BYPASS_ACCEPTANCE.json` = **12/12 PASS**）

| # | 场景 | 结果 |
|---|---|---|
| N1 | 直接 `--worker`（无上下文） | **REFUSED**，0 MODEL_LOAD / 0 WARMUP / 0 槽行 / 0 评分 |
| N2 | 上下文变量指向不存在的文件 | **REFUSED**，0 加载 |
| N3 | `--guard-probe` 无上下文 | **REFUSED** |
| N4 | **手写**上下文 + 自造 secret/HMAC | **REFUSED**（`consumption_receipt_missing`：回执无法伪造） |
| N5 | 签名后编辑 run_dir | **REFUSED**（`hmac_mismatch`） |
| N6 | 签名后编辑凭据摘要 | **REFUSED**（`hmac_mismatch`） |
| N7 | 回执被改成**另一个 run id** | **REFUSED**（`consumption_run_id_mismatch`） |
| N8 | 凭据在消费**之后**被编辑 | **REFUSED**（`credential_digest_changed_after_consumption`） |
| N9 | `--synthetic-command` **指向真实 worker** | 0 真实 F 工件；监督器记录 `published=false`；guard 探针 **REFUSED** |
| N10 | 合法授权 **+** `--synthetic-command` | **REFUSED**，**且 0 条消费回执**（不浪费一次性凭据） |
| **P1** | 监督器**在消费前无法**发布上下文；guard 在**独立进程**中接受监督器实际产出的上下文（什么都不加载） | **PASS** |
| **P2** | 合成路径端到端 64 行 `ROUND_PASS`，且**未发布任何执行上下文** | **PASS** |

**如实记录平台限制**：`os.chmod` 在 Windows 上无法表达"仅所有者"（stat mode 由只读位合成，
普通文件恒为 `0o666`），故 C2 的 0600 检查**仅在 POSIX 生效**；Windows 侧的私有性来自
运行目录的 ACL，以及 secret **既不进入命令行也不写入任何记录**。此项在 P1 的产物中明确标注。

---

## 1. 冻结身份（与前版相同，逐条实测）

| 类别 | 值 | 实测 |
|---|---|---|
| F 权重 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` | `f5acf83f…903a7f5` ✅ |
| backbone | `MASKCO_code/ckpts/cvrp100.ckpt` | `03615b08…` ✅ |
| 标签 | `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED` | 强制保留 |
| backbone 标签 | `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / HISTORICAL_BACKBONE_SHA_UNVERIFIED` | 强制保留 |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` | `66573ddc…` ✅ |
| profile / 合同 | `63e56212…` / 有效合同 `edc1b2ea…`（基础 `32124bfd…`） | ✅ |
| 64 状态 | 台账**发布顺序**，逐状态 `state_hash`，与 manifest 零分歧 | ✅ |
| 前向目标 | `targets_sha256` | `37e23f7567e3391a…` ✅ |
| 运行期 watch | 代码 + ckpt + backbone + npz + profile | 逐路径起止 SHA 闭合（`runtime_watch_closed`） |

## 2. 调用次数与预算（**未放宽**）

```
warmup_calls = 1 ; calls_per_state = 2 ; n_states = 64 ; max_forward_calls = 129
initialize_compile_s = 60.0 ; slot_span_s = 10.0 ; batch_s = 180.0
```

推导依据仍是 **V1.11 真实 F 实测**（槽跨度 max 0.344 s、初始化 6.344 s、首次编译 6.781 s、
22 状态批墙钟 19.562 s ⇒ 全 64 状态外推 70.03 s）。V1.19 的 B2-a/B2-b 反例已用**冻结的
60/180 s** 证明"批次超初始化上限但未超批上限不被误杀"与"真实初始化超限被强杀"，**本轮沿用**。

## 3. 授权凭据规格（与 V1.19 相同）

人类撰写的 JSON，逐项声明：`run_id` / `preregistration_sha256` / `f_ckpt_sha256` /
`backbone_sha256` / `data_sha256` / `profile_file_sha256` / `profile_hash` /
`effective_contract_hash` / `state_binding_sha256` / `state_order_sha256` /
`forward_targets_sha256` / `dependency_identity_sha256` / 三种调用次数 / 三个预算上限 /
`output_dir` / `approved_by` / `approved_utc` / `approval_statement` / `scope` /
`credential_sha256`（覆盖载荷，非整文件自哈希）。

生成流程（**机器不能自我批准**）：`--write-authorization-template` 产出**留空人工字段**的模板 →
**人类**填写三个 `approved_*` → 重新封装 `credential_sha256` → 执行时逐字段比对并**一次性消费**。

**注意**：在授权 JSON 里填一个非空 `approved_by` **不等于**已获得人类授权；人工批准必须有
**用户明确决策**作为来源，本文件的措辞不得被理解为"填字段即授权"。

## 4. 逐槽失败终止语义（V1.19 已闭合，本轮沿用）

首个失败槽（finite / shape / score_len / bitwise-repeat）立即终止：先落盘该槽记录 →
`FIRST_FAILURE` → 尾部身份记录 → 非零码退出；后续 63 槽 `NOT_ATTEMPTED_AFTER_TERMINATION`；
整轮 `ROUND_REVOKED`、verdict `BLOCKED`。**未改变已冻结的科学判据。**

## 5. 执行边界（签署后仍受约束）

**授权**：以 `run_id = d3l1-real-f-forward-20261015-v120` 在上述冻结身份下执行**一次**
64 状态真实 F 评分接口检查（最多 129 次前向），结论仅限"真实输入下 F 评分接口工程可用性"。

**不授权**：F−D-Geo 机制比较、E2 训练、L2/L3、40 日批次、最终测试、任何安装/写入真实状态、
任何效用或显著性主张。**执行一次即消费**；PASS/FAIL 都保留原件，不重跑、不补样；
**禁止拼接 V1.11 缺失的 42 条**。

## 6. Claim limits

1. 本文件是**规格 + 工程资格**；产出时真实 F 前向调用数 = **0**（见 `ZERO_REAL_F_WORK_AUDIT`）。
2. 通过只支持"真实输入下 F 评分接口工程可用"，**不**支持算法优势、学习增益或在线效用。
3. backbone 训练期身份未证；即使通过也只能声明"工程验证"一类结论。
4. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
5. 工程上限（60/10/180 s）**不是** L3 在线发布门，**不得**被引用为满足 8+2 秒。
6. 执行入口加固不等于"授权机制已被人类使用"：在拿不到用户明确决策前，真实 F 批次**仍为 0 次**。
7. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
