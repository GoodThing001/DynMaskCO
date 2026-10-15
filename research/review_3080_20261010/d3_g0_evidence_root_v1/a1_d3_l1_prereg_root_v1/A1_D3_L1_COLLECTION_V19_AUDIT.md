# V1.9 独立裁决：一次 F-free 64 槽采集完成，科学增益未建立

2026-10-15。启动前源码提交 `806efd559eea66571dee0ad4894c751a5d46175a`；前置 V1.8 已推送 `50721fb6dee348c5ff37b6319d80060c40577fe1`。本条只追加，不改 V1–V1.8 原文和旧 FAIL。

## 授权与实际执行

用户在明确的一次 16 日 × 4 槽签署申请之后直接答复“你自己判定核查 通过则继续执行”，已按 V1.9 预登记记录为人类条件授权。前置独立复核通过后，授权文件绑定运行身份、输出目录、冻结参数与通过的审核 SHA。supervisor 在首个 worker 前独占消费一次 token；本次已消费，不能通过新 run_id 继续采集。未替用户签名，未将授权扩展到 F−D-Geo、训练、L2/L3、40 日或最终种子。

实际运行：本地 Windows、Anaconda Python 3.12.7，numpy 1.26.4，jax/jaxlib 0.5.0，flax 0.10.4；无 GPU 科学任务。唯一 run_id `d3l1-formal-20261015-v19-once`，输出 `v19_evidence/formal_collection/`。16 天原顺序，每日前四个真实 reveal，固定分母 64；64 个均为真实 `PUBLISHED`，无失败、未到达、补采或筛天。模型身份 `F_FREE_NO_MODEL`。

| 验证 | 实际结果 | 原件 |
|---|---|---|
| 修改前 V1.8 独立正控 | 101 PASS / 0 FAIL | `v19_evidence/V18_INDEPENDENT_REVIEW_BEFORE_CHANGES.json` |
| V1.8 历史原件字节 | 1856 份零变化 | `v19_evidence/V18_ORIGINAL_BYTES_REVIEW.json` |
| 最终真实 worker 工程矩阵 | 44 PASS / 0 FAIL | `v19_evidence/engineering_release/ACCEPTANCE.json` |
| 真进程树强杀 | 8 场景 / 72 断言通过 | `v19_evidence/kill_release/KILL_E2E.json` |
| 授权接线正反例 | 20 PASS / 0 FAIL | `v19_evidence/authorization_release/AUTHORIZATION_GATE.json` |
| C0/快照共享回归 | 38 PASS / 0 FAIL | `v19_evidence/REGRESSION.json` |
| 正式一次采集 | 64/64，BATCH_COMPLETE | `v19_evidence/formal_collection/SUPERVISOR_RESULT.json` |
| 独立逐槽只读复核 | 1498 PASS / 0 FAIL，64 合格 | `v19_evidence/FORMAL_64_INDEPENDENT_AUDIT.json` |
| 源码起止身份 | 14 模块全一致 | `v19_evidence/COLLECTION_ADJUDICATION_V19.json` |

耗时：完整批次 **85.89100000006147 s / 3600 s**；最大完整 baseline rollout **0.5190811157226562 s / 120 s**；最大 capture 到全部验证完成的槽跨度 **3.327999999979511 s / 60 s**。各时钟起点独立，不以 worker 总时长替代 rollout 或槽窗口。

## 新修改与原来行为

- `d3_l1_collection_runner.py`：新增显式授权入参；新身份写 V19 文件；失败时完整 snapshot、两条轨迹、served masks 与比较结果存 `failed/` 原件，并在 FAILED 行记录 SHA。每槽重置失败上下文，避免引用上一槽结果。
- `d3_l1_p0c_v18.py`：核验人类条件授权、前置审计、参数/环境/目录/源码身份，独占消费全局一次 token；向子 worker 正确继承 formal 16 日与实际 day_id、授权 SHA；正式合格计数实际发布数。未授权入口仍拒绝，formal 不允许合成替身 worker。
- 新增 `d3_l1_v19_{authorization_gate,formal_readonly_audit,prepare_formal,state_coverage,seal_evidence}.py`：授权反例、全部 64 槽独立审计、启动前封存、纯读取覆盖检查和证据封口。各工具实际源 SHA 由原件记录。
- 共享恢复/strict-online/C0/合同/比较器/基线本轮不改。V1.8 恢复修复及首差异详见 [V1.8 审计](A1_D3_L1_P0C_V18_AUDIT.md)。原类型 numpy.float64 与 Python float 的值和 bits 相同但类型不同，不曾以 cast、容差或比较规范化“修绿”。

## 数据、合同和运行身份（完整 SHA256）

| 对象 | SHA256 |
|---|---|
| dev_check NPZ | `66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89` |
| profile 文件 | `63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8` |
| profile 内容哈希 | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| base contract | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0` |
| effective contract | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` |
| runner 原字节 | `08492329b84419958d06011a325cc28fa8314037ca75a862f96128959c34b136` |
| supervisor 模块原字节 | `1d897b8c0d0b65fe09c8230c4e506e157466a24cde157215479b7c10520f50f7` |
| RUNTIME_IDENTITY_V19.json | `7f1b18613c0c59504e24cd14d30dcdbc5fd14e0e231b2c03776528bacbbaf4c5` |
| RUNTIME_SOURCE_V19.zip | `5569153ab4e0904c209a962546c467d4199069668a0294d367f47d19d3ba0bce` |
| 人类条件授权文件 | `42ae1138703e7497dab727d59c0e4dac0089443288d54afc35594eaf8a4c73d7` |
| 独立 64 槽审计原件 | `781eb86aee03c45dce08978d4e1614c8ffd122b35cc0945ee9d67040d998a240` |

其他 14 模块逐文件起止 SHA、包版本、真实 origin、冻结文件身份、全部 64 state_hash/快照 SHA 见运行身份、supervisor 原件和 `STATE_MANIFEST_64_V19.json`；本轮 1497 份资格/采集/裁决原件逐文件 bytes/SHA 见 `v19_evidence/ARTIFACT_SHA256_V19.json`（不递归包含自身）。二进制和 JSON 原件按 `.gitattributes` 保持字节。

## 重放命令与不重采边界

正式 supervisor 的实际完整命令存于 V1.9 启动前准备及各身份原件。**一次 token 已消费，禁止重复执行正式采集命令**。以下是独立验收重放，仅读原件；输出使用新的临时路径，不覆盖已归档 JSON。

Windows PowerShell（仓库根目录）：

```powershell
cd C-VRP_Cold-chainVehicleRoutingProblem
$e = '../research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/v19_evidence'
python scripts/tests/d3_l1_v19_formal_readonly_audit.py --root "$e/formal_collection" --runtime-identity "$e/RUNTIME_IDENTITY_V19.json" --authorization "$e/USER_CONDITIONAL_AUTHORIZATION.json" --out "$env:TEMP/D3_V19_READONLY_REPLAY.json"
```

Linux（同等源码原字节和依赖，可运行只读审计；**服务器运行资格未实测**）：

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem
e=../research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/v19_evidence
python scripts/tests/d3_l1_v19_formal_readonly_audit.py --root "$e/formal_collection" --runtime-identity "$e/RUNTIME_IDENTITY_V19.json" --authorization "$e/USER_CONDITIONAL_AUTHORIZATION.json" --out /tmp/D3_V19_READONLY_REPLAY.json
```

原件中的路径分隔符、真实 origin 和环境绑定属于 Windows 实测身份；跨系统重放若不匹配须如实失败，不能自动改原件。正式运行源码原字节已在 ZIP 中保存。Windows/开发数据轻量 qualification 与真实正式16日采集在文件和分母上分开记账；两次候选工程资格也保留，非正式重采。

## 向稳定增益推进的实际发现与剩余工作

[状态覆盖原件](v19_evidence/STATE_COVERAGE_V19.json)纯读取 64 个已发布快照及保存投影，新增 rollout/基线规划/F 评分/F−D-Geo 比较均为 **0**。原始 pre-plan suffix 的 B 池 0/64 非空；保存的 post-baseline 诊断投影 B 池 **64/64 非空且均至少 2 单**。这是 reveal 时清空可变尾段的正常阶段边界。不得读空 pre-plan 尾段冒充共同基准 P0，也不得因此丢弃或重采任何状态。保存投影尚未成为真实准备后、通过 C0 认证且合同/状态绑定的共同 P0。

后续依次完成：共同 post-baseline P0 与真实可见 F 特征工程资格；E2 与 D8 的真实状态/合同绑定安装回执；另行冻结并获授权的 L1 同掩码归因、L2 合法决策质量、L3 配对日效用和跨条件稳定性。已有单个合成 F 评分接口 8/8 不是这 64 个真实输入的资格；当前骨干仅为 `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`。REV3 中未冻结的预算/阈值保持 PENDING，不自动填数。

`states_collected_for_L1=64/64`；P0-C 工程与状态来源门通过；`stable_online_gain=False`、`algorithmic_gain_claim=none`。D3 G0 总门、D8/E2、旧 D2 not_qualified、I3 正式负结果 **−144.680，95% CI [−191.109,−100.125]** 保持。未触及最终种子、上游、冻结数据或并行 CC_Compare 工作。后续任何正式科学执行仍须符合各门冻结条件和用户范围，不能由采集成功外推。
