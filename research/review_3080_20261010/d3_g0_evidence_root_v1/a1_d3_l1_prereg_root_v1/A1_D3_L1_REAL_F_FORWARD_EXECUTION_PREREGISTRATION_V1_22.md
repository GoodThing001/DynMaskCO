# V1.22：真实输入 F 前向执行版预登记（七要素 · 唯一性修复版）

- 轮次：`A1-D3-L1` / `V1.22 real-F forward execution qualification (single-shot receipt fix)`
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行任何真实 F 前向**）
- 绑定运行：`run_id = d3l1-real-f-forward-20261015-v122`
- 取代关系：本文件**取代** `…_V1_21.md` **作为签署对象**。
  **V1.18/V1.19/V1.20/V1.21 的预登记与 `v118_evidence/`–`v121_evidence/` 全部原样保留。**
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. 为什么需要 V1.22：一次性消费回执的路径缺陷（**未消耗任何执行机会**）

用户已批准 V1.21 的 `run_id` 与预登记 SHA 后，按合法路径
`--supervise --authorization --preregistration` 发起执行。批次在**启动 worker 之前**中止：

```
FileNotFoundError: '...\v121_evidence\consumption\authorization_consumption_5e9b9d0f...json'
  在 d3_l1_v118_authorization.consume() 的 O_EXCL 独占创建处
```

**根因**：`consumption_dir()` 对传入的**相对** `--consumption-root` 直接返回 `Path(root)`，
其解析依赖**当时的进程工作目录**；调用方与使用方的基准目录不同，导致 `mkdir` 建在了别处，
而独占创建落在不存在的目录上。

**为什么这次中止不算消耗机会（已机械核验，非声明）**：

- 消费记录写入使用 `open(path, 'x')`（**O_EXCL 独占创建**）⇒ 该次调用**未写出任何回执**；
- `v121_evidence/consumption/` **记录数 = 0**；
- `run_v121/` 中 **MODEL_LOAD = 0、槽记录 = 0**（worker 从未启动，F 权重从未加载）；
- 中止点在 `authorize()` 内、`run_contained()` 之前 ⇒ 无进程、无计时、无评分。

因此这是一次**被自身门禁挡下的工程中止**，真实 F 前向调用数仍为 **0**。

**修复**：`consumption_dir()` 现在把路径 `expanduser().resolve()` 为**绝对路径**（含
`D3_AUTH_CONSUMPTION_DIR` 环境变量分支），使一次性回执在整个运行生命周期内可被确定性定位；
并明确区分"回执目录无法创建"与"凭据已被消费"这两件事。

**为何必须新建身份**：`d3_l1_v118_authorization.py` **在 34 条冻结 watch 集之内**，修改它即改变
被冻结 SHA ⇒ 按既定规则必须建立**新的执行身份与预登记**，并保留 V1.21 全部原件。
核验结果与规则一致：V1.21 运行目录现报 **`NOT_READY`**、`n_drifted = 1`（正是该文件）。

---

## 1. 冻结身份

| 类别 | 值 |
|---|---|
| F 权重 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` `f5acf83fa2519020…903a7f5`（89,592,616 B，实测一致） |
| backbone | `MASKCO_code/ckpts/cvrp100.ckpt` `03615b08a55125a8…`（305,061,150 B，实测一致） |
| 标签 | `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`（强制保留） |
| backbone 标签 | `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / HISTORICAL_BACKBONE_SHA_UNVERIFIED` |
| 数据 | `data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz` `66573ddc653c863f…` |
| profile / 合同 | `63e56212174c9ce3…` / 有效合同 `edc1b2ea3914ddba…`（基础 `32124bfd…`） |
| 64 状态 | 台账**发布顺序**，逐状态 `state_hash`，与 manifest 零分歧 |
| 前向目标 | `targets_sha256 = 37e23f7567e3391a…` |
| **运行期 watch** | **导出闭包：30 个源码文件 + 4 项非源码资产 = 34 条**，逐文件 SHA 冻结（含授权器、执行上下文、V1.16 监督器、V1.14 相位日志、`typed_bytes` 定义者） |
| 运行环境 | python 3.12 / jax 0.5.0 / flax 0.10.4 / numpy 1.26.4，CPU 后端（`JAX_PLATFORMS=cpu`） |

## 2. 调用次数与预算（**未放宽**）

```
warmup_calls = 1 ; calls_per_state = 2 ; n_states = 64 ; max_forward_calls = 129
initialize_compile_s = 60.0 ; slot_span_s = 10.0 ; batch_s = 180.0
```

（依据仍是 V1.11 真实 F 实测：槽跨度 max 0.344 s、初始化 6.344 s、首次编译 6.781 s、
22 状态批墙钟 19.562 s ⇒ 全 64 状态外推 70.03 s。）

## 3. 授权、执行上下文与入口加固（V1.19–V1.21 已闭合，本轮沿用）

- 凭据逐项绑定 run_id / 预登记 / 权重 / 骨干 / 数据 / profile / 有效合同 / 状态绑定 / 状态顺序 /
  前向目标摘要 / 依赖身份 / 三种调用次数 / 三个预算上限 / 输出目录 / 人类批准 / 载荷摘要；
  **一次性消费**，回执落在**运行目录之外**且现在解析为绝对路径。
- 真实 worker 的**第一条动作**校验执行上下文（可解析 / 私有 secret / HMAC / run 绑定 /
  消费回执属本 run / 凭据消费后仍哈希一致），位于**一切加载之前**。
- 授权 + `--synthetic-command` 组合直接拒绝；直接 `--worker` 在无上下文时于加载前拒绝。
- **填充 `approved_by` 不等于人类授权**：人工批准必须有**用户明确决策**作为来源。

## 4. 逐槽失败终止语义（V1.19 已闭合，本轮沿用）

首个失败槽立即终止（槽记录 → `FIRST_FAILURE` → 尾部身份 → 非零码），后续 63 槽
`NOT_ATTEMPTED_AFTER_TERMINATION`，整轮 `ROUND_REVOKED`、verdict `BLOCKED`。
**未改变任何已冻结科学判据。**

## 5. 执行边界与消费规则

**授权**：以 `run_id = d3l1-real-f-forward-20261015-v122` 执行**一次** 64 状态真实 F 评分接口检查
（最多 129 次前向），全程冻结预算与严格身份核验；不调用 D-Geo、不训练、不安装计划、不做新 rollout。

**不授权**：F−D-Geo L1 比较（另一份独立授权）、E2 训练、L2/L3、40 日批次、最终测试、
任何效用或显著性主张。**C0 正式预算在本阶段不构成阻断**（只验证评分接口，不安装计划）。

**消费规则**：PASS / FAIL / 超时 / 中途强杀**均消费本次唯一机会**；不重跑、不补齐缺失状态、
**禁止拼接 V1.11 缺失的 42 条**；崩溃时保留原件、不静默重启整批。

## 6. Claim limits

1. 本文件是**规格 + 工程资格**；产出时真实 F 前向调用数 = **0**。
2. 通过只支持"真实输入下 F 评分接口工程可用"，**不**支持算法优势、学习增益或在线效用。
3. 闭包冻结只覆盖本项目本地源码；第三方库版本以运行身份 `packages` 记录。
4. backbone 训练期身份未证；即使通过也只能声明"工程验证"一类结论。
5. 复用已用于开发的状态**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日。
6. 工程上限（60/10/180 s）**不是** L3 在线发布门。
7. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
