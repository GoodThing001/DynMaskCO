# A1-D3-L1-COLLECTOR-QUALIFICATION-CORRECTION 裁决（P0–P2）

- 轮次：用户授权的限定范围资格修复（P0 运行身份 + R3、P1 真正的 resume_parity、P2 冷链合同身份）
- 唯一新套件：`scripts/tests/test_d3_l1_collector_qualification.py`（旧 `test_d3_l1_collector_gates.py` 已被取代）
- 证据（**同一次执行**）：`l1_gates/A1_D3_L1_COLLECTOR_QUALIFICATION{,_stdout,_stderr}.{json,txt}`，三者 mtime 均 **10:27:21**
- 结果：**10 行 / 9 PASS / 1 FAIL**；未做 64 槽位采集，未做 F−D-Geo 比较，未改冻结源码

## 1. 你的三点核查：全部成立，已逐条处置

### ① 冷链合同（P2）：**已修，且保留标签**

上一轮确实以 `contract=None` 跑通了"严格在线基线"路径，却把 profile 的 SHA 当作已生效身份 —— 你的判断正确。
现在按**本数据线自身的先例约定**构造生效合同（`run_fixed_state_quality.py:244-245`）：

```
base_contract = default_pilot_contract()                 # base hash  32124bfd4ef73169…
eff           = apply_objective_profile(base_contract, profile)   # effective hash edc1b2ea3914ddba…
profile_hash  = df1f7a6e935f00a0…   booking_horizon = 24.0（独立公开协议常数，不改冻结合同文件）
```

- `P2a/P2b` PASS：环境确实携带 `eff`（`coldchain_contract_is_eff=true`）且 horizon=24.0；16 实例 / 51 节点。
- **上一轮结果改标为**：`NON_COLDCHAIN_COLLECTOR_QUALIFICATION / DIAGNOSTIC_ONLY`。
- **仍待你确认**：`default_pilot_contract()+profile` 是否就是本轮**正式问题合同**（它是该数据线的既有约定，
  不是为了好加载而临时替换）；确认前 `IDENTITY_CLOSED` 不写"关闭"。

### ② resume_parity（P1）：**已改为数据比较——并且真的不过**

- `R4` 的"调用成功即通过"已废弃；现在逐字段比较原始 rollout 终局与恢复分支终局，并报告**第一个不同路径**：
  - `P1a` PASS：`served_mask` 全等（两个槽位）。
  - **`P1b` FAIL**：`traces` **不**相等，首个差异路径 `traces[0].serv…`，`original='float64'`（类型差异）。
- 因此 **`RESUME_PARITY = NOT_VERIFIED`**，且**本次不升级**。该差异的性质（表示层 float64/scalar vs array，
  还是真实的行为分歧）**尚未判定**，属下一轮第一优先；不预先解释掉。

### ③ 运行身份不一致（P0）：**确认成立，原因已定位，已修复并隔离**

- 实测：`SUPERSEDED_..._GATES.json` mtime **09:55:31**，`..._stdout.txt`/`..._stderr.txt` mtime **10:02:09**，
  stderr 含 `KeyError: 'restore_returns_dict'` → **JSON 与日志不是同一次执行**，你的判断正确。
- 原因（我的缺陷）：我把赋值键改成 `restore_non_none` 却漏改 R3 断言 → 10:02 那次**崩溃**且未写 JSON，
  磁盘上留下 09:55 那次（当时源码完整跑通 13 PASS/1 FAIL）的旧 JSON；我提交时把两者配成一对。
  **"JSON 已被当前源码+配套日志独立复现"为假**。
- 处置：三个旧产物**逐字节保留**并改名为 `SUPERSEDED_*`，附 `SUPERSEDED_..._RUN_IDENTITY_NOTE.json`
  说明配对无效；新套件的**同一次执行**产出源码 sha + argv + 日志 + JSON：
  `test_sha_in_json == 文件 sha = True`，三者 mtime 相同（10:27:21）。

### ④ R3 修法（P0）：按你的要求，不是"把 dict 改成 tuple"

`P0a` 先验证恢复载荷的**三个对象**及关键字段（实测 `arity=3`，类型 `[list, list, ndarray]`，
快照含 `replanner_state`、`served_mask` 已验证），`P0b` 再做 **capture → restore → 再捕获**
的状态身份校验：同一 clock 重新捕获的 `snapshot_state_hash` 与原始**完全一致**（`0e4becde995b52dd…`，
`first_difference=null`）→ `SNAPSHOT_RECOVERY = PASS`（状态身份，而非调用成功）。

## 2. 四项状态（分列）

| 项 | 状态 |
|---|---|
| `COLLECTOR_CAUSALITY` | **PASS（冷链合同生效下）**：五项 R1 槽位断言成立；单实例滚动级因果（上轮）+ 本轮冷链路径 |
| `SNAPSHOT_RECOVERY` | **PASS**：三对象解码 + capture→restore→re-capture 哈希一致 |
| `RESUME_PARITY` | **NOT_VERIFIED**：served_mask 一致，`traces` 首差异已定位并报告 |
| `IDENTITY_CLOSED` | **PARTIAL**：生效合同哈希已记录；"是否正式问题合同"待你确认 |
| 冷链采集合同身份 | 由 `OPEN` → **已记录（待确认）** |
| 服务器正式资源门 | **NOT_RUN**（本轮计时仅本机单实例样本：rollout 0.317 s、总 1.4 s，均在预登记上限内） |

## 3. 保留的限制与我自己的缺陷

1. `P1b` 未过；`traces` 差异性质未知 → 不得据此认为恢复链路可用。
2. 计时是本机、单实例、K=2 的**资格样本**，不是服务器正式资源门；服务器需独立 run id、源码身份、
   负载/异常/超时记录，且单状态上限须覆盖恢复、校验与 parity 全过程。
3. 旧套件 `test_d3_l1_collector_gates.py` 已被取代，其最后一次运行仍 exit=1（本轮未继续排查），
   其产物已隔离改名并注明配对无效。
4. 我的缺陷（保留原文）：`sha256_file(None)`；rollout 入口名表过窄；赋值键/断言键不一致导致崩溃；
   `rec()` 少参调用导致一次运行在写盘前崩溃（已修）；两次 PowerShell 转义失败（`SyntaxError: unexpected
   character after line continuation`）导致补丁未生效——均未被改写为通过。

## 4. Claim limits

工程资格，不是模型增益。`algorithmic_gain_claim: none`；`states_collected_for_L1 = 0`。
I3 正式负结果、D2 `not_qualified`、D3 G0、D8、E2 与 `stable_online_gain=False` 全部不变。
