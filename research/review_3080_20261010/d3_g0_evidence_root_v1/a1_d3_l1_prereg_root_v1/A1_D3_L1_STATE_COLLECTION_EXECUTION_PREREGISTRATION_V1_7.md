# A1_D3_L1_..._V1.7：P0-C 阻断修复（部分）+ 正式 worker 成功路径的真实阻断

- 追加记录；**V1–V1.6 与既有产物、冻结资产、`MASKCO_code/`、旧报告不改写**
- 补丁原件（可复现）：`V17_PATCH.py`（逐项 anchor 断言）
- **签署状态：`SIGNATURE = HOLD / REVISION_REQUIRED`**；`FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED`

## 1. 你的独立审查：全部成立

| 你的发现 | 核实结果 |
|---|---|
| `parity_ok` 用 `v == 'EQUAL'` 扫全字典（含后缀列表与布尔） | **成立**：正常路径必然 `bitwise_parity_failed` |
| 13/13 未覆盖 worker 成功路径 | **成立** |
| 预先登记四槽 → 未到达槽被误记、slot 超时从 rollout 起算 | **成立** |
| 身份门未强制 13 模块闭包/runner/profile/生效合同 | **成立**（见 §3 未完成项） |
| 120 s 作用域错误（限 worker 存活而非 rollout） | **成立** |
| 完整恢复轨迹 parity 未做 | **成立** |
| 运行时强杀未真实验证 | **成立** |
| V1.6 引用的两个证据 JSON 不在提交内 | **成立**：上次 `Copy-Item` 发生在一次失败的运行里，故未落盘；本轮已补齐 |

另外更正我的措辞：`acceptance()` 会跑 `dev_check` 第 0 日的**真实 rollout**，应记为
**开发数据上的轻量验收**，不应统称"合成反例"；它与正式采集的运行身份/用途/计数必须分开。

## 2. 本轮已完成的修复（E1–E4、E6）

| 编号 | 修复 | 证据 |
|---|---|---|
| E1 | `parity_ok` 只把**数组逐位判定**当字符串比较，结构项按布尔判定：`served_mask`/`full_resume_served_mask`/`full_resume_traces` == `EQUAL` **且** `suffix_boundary`/`serialization_roundtrip`/`full_resume_parity` 为 True | 补丁 E1；失败记录新增 `bitwise_parity` 明细 |
| E2 | worker 保留原始 rollout 终局，`run_resumed` 后做**完整轨迹逐位比较** | `full_resume_*` 三项明细 |
| E3 | 槽位登记**绑定真实 reveal hook**：采集器新增 `pre_capture` 回调，在 `capture_recourse_snapshot` **之前**写 `ATTEMPT_STARTED` 并 fsync；未到达的槽**不再**被预先标记 | 账本实测顺序：`DAY_ROLLOUT_STARTED → ATTEMPT_STARTED×4 → DAY_ROLLOUT_FINISHED` |
| E3b | 修复 rollout 后读到**旧账本快照**导致四槽被 `continue` 跳过的实现错误（改为 fresh read） | 修复前 PUBLISHED=0 且无 FAILED；修复后进入真实校验 |
| E4 | 120 s 改为**只作用 rollout 窗口**：worker 写 `DAY_ROLLOUT_STARTED/FINISHED`，supervisor 以两者时间戳判定；进程存活仅作独立 `lifetime_guard` | 补丁 E4a/E4b/E4c |
| E6 | `--inject-block-s` 使指定槽阻塞，用于真实强杀验收 | CLI 已就位 |

**未完成（仍阻断签署）**：**E5** —— 身份门尚未强制 13 模块运行时闭包、runner/profile SHA 与生效合同哈希，
也尚无运行后独立对照；supervisor 实际强杀（E6 的端到端使用）本轮**未执行**。

## 3. 正式 worker 成功路径的真实阻断（新发现，必须由你裁定）

修复后 worker 真正执行校验，四槽**全部 `FAILED(bitwise_parity_failed)`**，逐项归因：

| 子项 | 结果 |
|---|---|
| `served_mask` | **EQUAL** |
| `full_resume_served_mask` | **EQUAL** |
| `serialization_roundtrip`（pickle protocol 4 往返） | **True** |
| `suffix_boundary`（车辆边界后缀） | **True** |
| **`full_resume_traces`** | **DIFFERENT** |

这与 P1b 当年的发现一致：恢复分支的 trace 叶子类型为 Python `float`，原始为 `numpy.float64`
（数值相同、**类型不同**）。在**你预登记的逐位/精确类型精度**下，这被判为 **DIFFERENT**，
因此 worker **正确地拒绝发布**——这不是假通过，而是 fail-closed 生效。

**因此正式采集目前不可签署**，需你择一裁定：
1. **修实现（推荐）**：让 `run_resumed` 的 trace 记账与原始 rollout **类型一致**（不放松任何判据），
   再验证 `full_resume_traces == EQUAL`；
2. **改判据（需你明确授权）**：把 trace 对象图定义为按**规范化投影**比较、仅数组按位——
   这属于**降低**已冻结精度，我**不会**自行采用；
3. 若二者皆不接受，则录为 `FORMAL_WORKER_BLOCKED_BY_RESUME_TYPE_PARITY` 并停止采集线。

## 4. 资源与身份纪律

- 异常/强杀验收**只在隔离临时目录**进行，不占用正式 64 槽分母；本轮 worker 探针用 `%TEMP%` 下的
  scratch 目录与独立 `run_id`（`v17-probe`/`v17b`/`v17c`），**未**写入任何正式采集目录；
- 共享服务器仍需注意进程树清理、SSH 中断与写盘故障；`tmux` 不作为持久化保证。

## 5. 状态

```
P0_A = CLOSED_ON_COUNTEREXAMPLE_GATE
P0_B = PASSED_ON_PLAN_B_GATE
P0_C = V1_7_PARTIAL / FORMAL_EXECUTION_REVISION_REQUIRED
       (E1-E4,E6 done; E5 identity gate extension + supervisor real-kill E2E still open;
        worker success path BLOCKED by full_resume_traces type-level DIFFERENT)
SIGNATURE = HOLD
FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED
states_collected_for_L1 = 0
```
未执行正式 64 槽位采集、未做 F−D-Geo 比较、未训练、未修改冻结资产或旧报告。
`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、`stable_online_gain=False`、
I3 正式负结果（ΔJ = −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 全部不变。
