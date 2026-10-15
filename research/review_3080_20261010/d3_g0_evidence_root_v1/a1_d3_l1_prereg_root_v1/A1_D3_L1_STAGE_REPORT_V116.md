# A1-D3-L1 阶段总汇报：一次 F-free 正式采集 + 真实输入资格 + F 前向失败 + V1.16 监督器撤销语义

- 汇报范围：**本阶段全部已完成工作**（你执行的部分 + 本轮我补完的部分）
- 基准：`46251e8`（`main` 与 `codex/o0cc-cal-phys` 同步；`2bba9be`/`806efd5` 为并行线提交）
- 状态：**科学执行 HOLD**；不申请升级签署；未启动受禁止批次

## 1. 本阶段已完成（你执行的部分）

| 工作 | 结果 |
|---|---|
| **一次 F-free 正式采集** | **64/64 发布**，独立验收 **1498/1498**，**未补采**（一次 pass 约束成立） |
| **真实基线方案与输入检查** | **64/64 通过**，独立复核 **1619/1619** |
| **真实 F 前向** | **22/64 完成、42 未完成 → 整轮 `BLOCKED`** |
| **持久化修复** | 合成压力测试 **11/11**；修复后**未重跑 F** |
| **负结果独立审计** | **707/707** |

**F 失败的确定原因**：Windows 进度文件替换触发 `PermissionError`。
已完成的 **22 个**状态评分**均满足**长度、有限性与重复逐位一致性；**未**外推到其余 42 个状态。
失败原件与两次未通过的合成修复记录**全部保留**。
此前恢复差异已确认为 **`numpy.float64` → Python `float` 类型漂移**；修复**保留原类型与位模式**，**未放宽比较器**。

提交：`2bba9be`（采集与独立验收）、`46251e8`（输入资格、F 失败、持久化修复）。
`codex/o0cc-cal-phys` 本地与远端一致；`main` 本轮已由我推送同步至 `46251e8`（此前落后于 `4ca034d`）；未混入 `CC_Compare` 并行修改。

## 2. 本轮我补完的部分（V1.16 修订 F 监督器）

### 2.1 你指出的待验证风险：**已实现并验证**

> 写出逐槽 PASS 文件后，如果**尾部身份记录失败**，父进程必须**撤销整轮通过资格**。

**缺口定位**：`d3_l1_v116_forward_control.py` 的 `reconcile_rows()` 逐槽 `qualified` **完全不参考整轮控制结果**，
因此 64 个 PASS 行已落盘时，尾部身份失败**不会**影响合格资格——这正是你担心的假通过路径。

**修复（最小改动）**：

1. `run_contained()` 在结束身份检查后**发布尾部身份记录** `CONTROL_IDENTITY_END.json`（原子发布）；
2. 新增 **`round_verdict(out, control, rows, expected_rows=64)`**：逐槽结果为**临时（provisional）**；
   整轮合格须同时满足 ①尾部身份记录存在且 `ok` ②`control.ok` 为真 ③合格行数 == 64；
   否则**撤销整轮**：把所有行降级为 `qualified=False, revoked=True,
   status='REVOKED_*'` 并**持久化 `ROUND_REVOKED.json`**（合格时写 `ROUND_PASS.json`）；
3. `reconcile_rows()` 文档注明其逐槽判定为临时结果，须经 `round_verdict()` 裁决。

**关键语义**：逐槽 PASS 文件**保留在磁盘作为不可变证据**，但**整轮裁决优先**——尾部失败绝不可能留下一个"通过的轮次"。

### 2.2 独立可重放验收：**6/6 PASS**（真实子进程，不跑 F、不采状态）

| 场景 | 期望 | 实测 |
|---|---|---|
| **S1** 64 行 + 有效尾部 | `ROUND_PASS` | **ROUND_PASS**（64 qualified，真实受控子进程） |
| **S2** 64 行 + **尾部被删除** | 整轮撤销、64 行全部降级 | **ROUND_REVOKED**（64→0，`TAIL_IDENTITY_MISSING_OR_INVALID`） |
| **S3** 64 行 + 尾部**损坏** | 整轮撤销 | **ROUND_REVOKED** |
| **S4** 64 行后 worker **改写被监视文件** | 结束身份失败 → 撤销 | **`HARD_STOP_IDENTITY_END` + ROUND_REVOKED** |
| **S5** 仅 63 行 + 有效尾部 | 撤销（行数不足） | **ROUND_REVOKED**（`INCOMPLETE_ROWS`） |
| **S6** 槽内真实阻塞 → 监督器**真实强杀** | 截止命中 + 撤销 + 终止/未到达状态 | **`slot_span_deadline` + ROUND_REVOKED** |

证据：`v116_evidence/A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json`；
入口：`python scripts/tests/d3_l1_v116_forward_control_acceptance.py --outdir <dir>`（一条命令可独立重放）。

**我自己的两处脚本缺陷（保留原文，未改写为通过）**：S4 首版把期望哈希写错 → 在 **start** 就 `HARD_STOP_IDENTITY`
（未测到 end 路径）；第二版忘了把 `mode` 传成 `mutate_watched` → worker 未改写文件。
两次失败均已记录，第三次通过。

## 3. 剩余工作（未完成，不申请签署）

1. **完整修订 F 流程资格**（修复 Windows 进度文件替换的 `PermissionError`，重跑并覆盖 64/64，仍不合并既有 FAIL 原件）；
2. **C0 预算与状态/合同绑定认证**：现有**组合认证器**与**提交协调器**可复用，但缺口需精确表述为——
   **这 64 个状态尚未绑定具体 C0 预算**、**品质投影尚无保守保证**、**真实安装回执未闭合**；
3. **D8 安装回执**；
4. **E2**（合格贡献证明，须另立预登记）；
5. 后续**独立在线效用与稳定性验证**。

## 4. 边界与 Claim limits

未采集新状态、未重跑 F、未做 F−D-Geo 比较、未训练、未修改冻结资产/`MASKCO_code/`/旧报告；
`CC_Compare` 并行修改未混入。**稳定正向增益仍未建立**；`algorithmic_gain_claim: none`；
D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、I3 正式负结果
（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
