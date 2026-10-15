# V1.21：真实输入 F 前向执行版预登记（七要素 · 运行期依赖冻结闭环版）

- 轮次：`A1-D3-L1` / `V1.21 real-F forward execution qualification (dependency-freeze closed)`
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行任何真实 F 前向**）
- 绑定运行：`run_id = d3l1-real-f-forward-20261015-v121`
- 取代关系：本文件**取代** `…_V1_20.md` **作为签署对象**。
  **V1.18/V1.19/V1.20 的预登记与 `v118_evidence/`、`v119_evidence/`、`v120_evidence/` 全部原样保留**
  （未改写、未删除；本轮反例脚本只在临时目录内篡改并立即还原）。
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 本次闭合的第五个阻断：运行期源码依赖漏登记

审查发现 `IDENTITY_TARGETS` 的 **14 条** watch 路径**未包含**真实执行路径实际 import 的模块：

```
scripts/simulation/d3_l1_v118_authorization.py      （授权器）
scripts/simulation/d3_l1_v120_execution_context.py  （执行上下文闸门）
scripts/tests/d3_l1_v116_forward_control.py         （监督器）
scripts/tests/d3_l1_v112_phase_journal.py           （相位日志）
```

并追查真实 F worker 调用的 **`typed_bytes`**（`scripts/tests/d3_l1_v18_readonly_audit.py` 定义，
决定**逐位重复**判定）——**同类漏登记，共 5 个已知文件**。

**后果**：改动其中任何一个都会改变执行、身份校验、计时或结果判定，而 `--readiness` 仍报 READY。

### 加固：watch 集不再手写，而是**导出**

`scripts/data/d3_l1_v121_dependency_closure.py`：

```
seeds   = 真实执行路径的入口模块
          （runner / run_identity / persistence / targets / authorization /
            execution_context / v116_forward_control / v112_phase_journal）
edges   = 从 seeds 出发、按 `import x` / `from x import y`（含函数体内导入）在本项目 `scripts/`
          树内解析出的本地边
closure = 可达的**全部**本地 `.py`，迭代到不动点
watch   = closure 去掉**故意排除**的工具（每个都记录理由），再由 identity 模块追加
          非源码资产（权重 / backbone / 数据 / profile，各自显式指定相对根）
```

实测：**源码闭包 30 个文件**（原 9 个源码 + 权重等共 14 条 → 现 **34 条**）。
`--readiness` 新增 **`runtime_watch_covers_closure`** 交叉检查：冻结集若**漏掉**闭包中任一文件，
直接 `NOT_READY` 并把缺项列出——**冻结与冻结校验来自同一处，无法各自漂移**。

### 反例（`v121_evidence/D3_V121_FREEZE_CLOSURE_ACCEPTANCE.json`）

| # | 检查 | 内容 |
|---|---|---|
| F1 | 闭包完整性 | 四个"漏登记"模块 + `typed_bytes` 定义者**全部在**导出 watch 集中；闭包干净（无未解析 seed、无意外可达的被排除项） |
| F2 | 冻结完整 | V1.21 身份携带**完整** watch 集（34 条），逐路径 SHA 现场复算一致 |
| F3 | 动态 readiness | `READY_FOR_USER_SIGNATURE`，`runtime_watch_covers_closure = true`，`real_F_forward_executed = false` |
| F4/F5 | **逐模块篡改** | 四个漏登记模块**各自**隔离篡改 ⇒ readiness **NOT_READY**；还原后回到 READY。`typed_bytes` 定义者同样单独验证 |
| F6 | **全量扫掠** | 冻结集中**每一条**路径**逐一**篡改、**逐一**被检出（无路径未被检验） |
| F7 | 闭包承重 | **仅传递可达**的文件也在冻结集内，且其篡改同样被检出 |
| F8 | 原件保留 | V1.18/V1.19/V1.20 证据目录仍在且未被本轮改写 |

---

## 1. 冻结身份（与前版相同，逐条实测）

| 类别 | 值 |
|---|---|
| F 权重 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` `f5acf83f…903a7f5` |
| backbone | `MASKCO_code/ckpts/cvrp100.ckpt` `03615b08…` |
| 标签 | `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`（强制保留） |
| backbone 标签 | `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / HISTORICAL_BACKBONE_SHA_UNVERIFIED` |
| 数据 / profile / 合同 | `66573ddc…` / `63e56212…` / 有效合同 `edc1b2ea…`（基础 `32124bfd…`） |
| 64 状态 | 台账**发布顺序**，逐状态 `state_hash`，与 manifest 零分歧 |
| 前向目标 | `targets_sha256 = 37e23f7567e3391a…` |
| **运行期 watch** | **导出闭包：30 个源码文件 + 4 项非源码资产 = 34 条**，逐文件 SHA 冻结，起止闭合 |

## 2. 调用次数与预算（**未放宽**）

```
warmup_calls = 1 ; calls_per_state = 2 ; n_states = 64 ; max_forward_calls = 129
initialize_compile_s = 60.0 ; slot_span_s = 10.0 ; batch_s = 180.0
```

## 3. 授权凭据与执行上下文（V1.19/V1.20 已闭合，本轮沿用）

- 凭据逐项绑定 run_id / 预登记 / 权重 / 骨干 / 数据 / profile / 有效合同 / 状态绑定 / 状态顺序 /
  前向目标摘要 / 依赖身份 / 三种调用次数 / 三个预算上限 / 输出目录 / 人类批准 /
  `credential_sha256`（覆盖载荷）；**一次性消费**且回执落在运行目录之外。
- 真实 worker 的**第一条动作**校验执行上下文（可解析 / 私有 secret / HMAC / run 绑定 /
  消费回执属本 run / 凭据消费后仍哈希一致），位于**一切加载之前**；授权 + `--synthetic-command`
  组合直接拒绝。
- **在授权 JSON 里填非空 `approved_by` 不等于已获人类授权**：人工批准必须有**用户明确决策**作为来源。

## 4. 逐槽失败终止语义（V1.19 已闭合，本轮沿用）

首个失败槽（finite / shape / score_len / bitwise-repeat）立即终止（槽记录 → `FIRST_FAILURE` →
尾部身份 → 非零码），后续 63 槽 `NOT_ATTEMPTED_AFTER_TERMINATION`，整轮 `ROUND_REVOKED`、
verdict `BLOCKED`。**未改变任何已冻结科学判据。**

## 5. 执行边界（签署后仍受约束）

**授权**：以 `run_id = d3l1-real-f-forward-20261015-v121` 在上述冻结身份下执行**一次**
64 状态真实 F 评分接口检查（最多 129 次前向）。**不授权**：F−D-Geo、E2、L2/L3、40 日、最终测试、
任何安装或效用/显著性主张。执行一次即消费；**禁止拼接 V1.11 缺失的 42 条**。

## 6. Claim limits

1. 本文件是**规格 + 工程资格**；产出时真实 F 前向调用数 = **0**。
2. 通过只支持"真实输入下 F 评分接口工程可用"，**不**支持算法优势、学习增益或在线效用。
3. 闭包冻结只覆盖**本项目本地源码**；第三方库版本以运行身份的 `packages` 记录，不以文件 SHA 表达。
4. backbone 训练期身份未证；即使通过也只能声明"工程验证"一类结论。
5. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
6. 工程上限（60/10/180 s）**不是** L3 在线发布门。
7. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
