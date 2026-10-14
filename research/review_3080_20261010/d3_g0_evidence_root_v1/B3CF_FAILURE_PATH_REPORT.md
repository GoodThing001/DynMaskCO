# D3-G0 Evidence B3-CF — 提交失败路径反例验证报告（READ-ONLY）

- 阶段：`D3_G0_Evidence_B3CF_submit_failure_paths`
- 授权范围（逐字约束）："本批只允许在真实 `StrictOnlineEnv` 的隔离副本中构造与执行反例，记录实际状态前后差异"；"对于 F10，预计当前实现会暴露旁路，这应如实记录为 `COUNTEREXAMPLE_REPRODUCED`，而不是因为测试成功复现缺陷就标记 G-L 通过"；全部预登记反例到达确定结论后停止；"不修改 `_commit`，不接入真实 D3 写回，不放宽 D8，不运行科学效果实验"。
- 执行入口：`C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_b3cf_submit_failures.py`
- 预登记（先于实现冻结）：`B3CF_COUNTEREXAMPLE_SPEC.json`，sha256 `567726481b3b0e47f7dd85a925bd6c85eb20e82ccec17e41d0dda4e96e257749`（本批**未修改**该文件）
- 结果矩阵：`B3CF_FAILURE_PATH_MATRIX.json`，sha256 `2b679bfcfbd6872a8b203e59b790b1ac3108ef731017af5abbdc5b2e8b9c82ae`
- 字节冻结记录：`B3CF_BYTE_FREEZE.json`（本批全部登记字节的 sha256 + 本批新增的容器 pin）；同批还有 `B3CG_SPEC_ADDENDUM.md`（B3-CG 规格附录：A-1 量词限制、A-2 唯一授权写入者、A-3 D0 拆分）
- 原始 stdout：`raw/B3CF_failures_stdout.txt`（sha256 `1e2b61b24cfbc7ace2b2127176c6eb43f009cb08f402bed8582ac9e2ed15c326`）；stderr 为空（0 字节，sha256 `e3b0c442…b7852b855`）
- 结论：**G-L 与 G-C 均 `NOT_ESTABLISHED`**（本批只做反例判定，不做闸门通过判定）

## 0. 本批没有触碰生产实现

| 禁项 | 状态 | 证据 |
|---|---|---|
| 修改 `_commit` / `_plan_and_commit` | 否 | `scripts/simulation/strict_online_env.py` sha256 `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6`（与 B3-CG 阶段一致，未改） |
| 接入真实 D3 写回 | 否 | `real_writeback_enabled=False` |
| 放宽 D8 | 否 | `d8_relaxed=False` |
| 科学效果实验 | 否 | 只跑仿真，不产出效用/收益数值 |

**只读性有两重证据。**

1. 结构性：所有反例只在 `A.clone_state(snapshot)` 产生的脱离副本上操作；`A.prepare_clone` 强制使用**新建**的 replanner（`make_continuation()`），从不驱动 `env.replanner`。
2. **经验性（本批新增）**：`AUDIT` 用**无 hook 的全新 env** 重跑同一实例，把整条生产轨迹（每车服务序列、时刻、picked_order_id、depot 卸货记录、served_mask）规范化后取 sha256，与带探针的那次运行比对。

```
AUDIT  OK  the counterexample probes did not perturb the production trajectory
probed_outcome_sha256 = clean_rerun_outcome_sha256
```

6 个实例（0–5）的 AUDIT 全为 `OK`，见 §4 的稳定性扫描。

## 1. 逐反例判定（实例 0，`--instances 0 --heavy-cap 20`）

| id | 判定 | 一句话 |
|---|---|---|
| CP-F10 | `COUNTEREXAMPLE_REPRODUCED` | 真实提交循环执行了一个含**已服务**客户的后缀，全路径无任何认证 |
| CP-F8 | `COUNTEREXAMPLE_REPRODUCED` | 批准与首次真实写入之间隔了一次不受控的状态变化，写入前无再验证 |
| CP-F9 | `MECHANISM_SATISFIED_ON_REAL_PATH` | 后续真实揭示使旧凭证失效（机制成立，但**无生产调用者**） |
| CP-A-c7 | `COUNTEREXAMPLE_REPRODUCED` | `_validate_state` 对热越界**不抛异常**，只计数 |
| CP-A-c8 | `CONCLUSION_RECORDED` | 门升温地板分析：hard_upper 在容量模型下是否可能被触及 |
| CP-A-c5 | `NOT_REPRODUCED` | 登记的"记录卸货时刻早于物理"**不成立**（见 §3，含正向测量） |
| PREREG | `OK` | 6 个预登记反例全部覆盖，无遗漏 |
| AUDIT | `OK` | 探针未扰动生产轨迹 |

`verdict_counts = {"SETUP_OK":1, "COUNTEREXAMPLE_REPRODUCED":3, "MECHANISM_SATISFIED_ON_REAL_PATH":1, "CONCLUSION_RECORDED":1, "NOT_REPRODUCED":1, "OK":2}`

### CP-F10 — 提交路径旁路（**缺陷复现，如实记录**）

判定规则预先登记为："复现缺陷即记 `COUNTEREXAMPLE_REPRODUCED`，**不得**据此报告 G-L 通过"。矩阵 `critical_rule_honoured` 字段逐字记录该规则。

实测（实例 0）：

```
served_customer_used            = 32        # 已被服务的客户
vehicles_needs_replan_cleared   = true      # 关掉 replanner 触发，排除"其实是 replanner 写的"
before : status=ready      committed_next=None  suffix=[32]
after  : status=committed  committed_next=32    suffix=[]
state_changed                   = true
illegal_head_committed          = true
composed_gate_invoked           = false
certifier_called                = false
```

- 调用的是 env **自己的** `_plan_and_commit`（不是测试里重写的提交逻辑）。
- `needs_replan=False`，所以 replanner 根本不会被调用；这次写入只能来自提交循环。
- `d3_composed_certifier` / `d3_install_gate` 在该路径上零调用。
- before/after 是**实际状态差异**（同一份克隆对象上的字段读出），不是推断。

这独立复现了 B3-CG 静态审计的结论：`_commit`（`strict_online_env.py:260-299`）与 `_plan_and_commit`（`:623-656`）不做任何可行性检查，7 个 `certify_plan` 旁路调用点之一只是表层证据。

### CP-F8 — 批准与首次写入之间的 TOCTOU

```
approved_at_S1                = true
S1_state_hash                 = 0b81882cb625
S2_state_hash                 = 9ff1117d7b5e     # 批准后状态已变
state_changed_after_approval  = true
tail_cleared_on_the_copy      = [38,12,2,5,9,1,14,7,0] -> []
write_after_state_change_without_reverification : status=committed, committed_next=38
head_written                  = 38
```

- 在副本上模拟一次真实揭示（清空可变尾）并使时钟前进，然后**不做任何再验证**就用真实写入原语提交 S1 批准的方案。写入成功。
- 结论措辞（矩阵 `note` 逐字）："re-comparing three hashes does NOT make the commit atomic: nothing binds the re-check to the write, and no other writer is excluded"。这与用户 B3-CG 评审点 (2) 一致：**唯一写入者必须是"由闸门授权的提交事务"**，而不是让闸门自己变成另一个写入者；F8 不能靠再比三次哈希来消除。

### CP-F9 — 揭示使凭证失效（机制成立，但**无生产调用者**）

```
n_reveal_transitions = 10            # 实例 0；6 实例区间 9–13
每个 transition: state_hash_changed = true, plan_digest_changed = true
reverify_allow       = false
refusals             = [plan_digest_anchor_mismatch, plan_hash_legacy_mismatch, state_hash_mismatch]
scope_caveat         = "the mechanism refuses, but NO production path calls it, so this is NOT evidence that continuous safety (G-C) holds"
```

`pre_install_reverify` 在真实揭示前后确实拒绝，但该函数**零非测试调用者**。因此这是"机制存在"而非"持续安全成立"，判定词刻意用 `MECHANISM_SATISFIED_ON_REAL_PATH` 而不是 PASS。

### CP-A-c7 — 热越界不被拒绝

```
excursion_temperature_c = 35.0      # > hard_upper(class 0) = 30.0
validate_state_raised   = null      # 不抛
violations_counted      = 1
```

复现了 B3-CG 静态结论：Ω 是**评价层**集合（`coldchain_evaluator.py:358`：`thermal_count == 0 and thermal_duration <= 1e-9`），而 `_validate_state`（`coldchain_state.py:533-557`）**不校验温度**：越界只被计数、从不被拒绝。

### CP-A-c8 — hard_upper 可达性（结论记录，非通过）

按门升温步长 `door_heat_kwh[z] / thermal_capacity_kwh_per_c[z]` 计算单车在被门升温推到 hard_upper 前可吸收的取货次数：

| zone | target °C | hard_upper °C | 每次开门升温 °C | 到 hard_upper 的最大开门次数 |
|---|---|---|---|---|
| 0 | 18 | 30 | 0.033333 | 360 |
| 1 | 4 | 8 | 0.087500 | 45 |
| 2 | −18 | −12 | 0.120000 | 50 |

`binds_free = false`（三区全部）。即：在本配置的容量/门热模型下，硬上界不是"单次事件即触发"，而是需要几十到几百次开门累积；这**限定了**容量地板能否真正约束（`CONCLUSION_RECORDED`，不构成任何方向的通过判定）。

## 2. 判定词的语义边界（本批新增的诚实约束）

本批在实现过程中修掉了两处**判定词误用**，并把修正写进代码注释，避免以后被当成通过：

1. **`NOT_VERIFIABLE` 不能被写成 `NOT_REPRODUCED`。** 第一版 CP-A-c5 在 `return_finish` 读回 `None` 时 `gap_h=None`，就落到 `NOT_REPRODUCED`——那是**没有测量**却给出了否定判定。现在只有当测量为正（两条物理链被真正区分）才允许 `NOT_REPRODUCED`，否则一律 `NOT_VERIFIABLE`。
2. **夹具选择缺陷（本批自查发现）**：第一版对所有反例都用 `items[0]`（clock 0）。在 clock 0 无已服务客户、无在途货物、无停靠客户点的 ready 车，于是 CP-F10 / CP-A-c7 / CP-A-c5 会因**错误原因**变成 `NOT_VERIFIABLE`。现在每个反例用 `pick(pred)` 选择**真正具备该前置条件**的决策点，且找不到时仍回落 `NOT_VERIFIABLE`（不回落"通过"）。

## 3. CP-A-c5：登记的怀疑被**正向测量否定**（`NOT_REPRODUCED`）

**登记时的怀疑**（见 B3CG 讨论）：`_force_return_all` 经 `_commit(node=0)` 只把冷链推进到 `depart`（`strict_online_env.py:272`），却把 `return_finish` 设为 `depart + travel`（`:273`），于是记录卸货时刻可能"早于物理"，使记录的 `delivered_quality` 比物理乐观，从而削弱任何以交付记录做的交叉校验。

**决定性测量**（纯只读；`transition_segment` 是纯函数，`cc0` 是不可变 frozen dataclass，所以在**同一**调用前状态上独立投影出两条候选物理链）：

- `chain_A` = 只推进到 `depart`（缺陷预测的记录值）
- `chain_B` = 推进到 `arrival`（正确物理预测的记录值）
- 再把**实际记录**的 `delivered_quality` 与两者比较

实例 0 实测：

```
vehicle_id                        = 1
node_before_return                = 32
return_travel_h                   = 0.07676636427640915
coldchain_updated_time_before     = 1.2553534507751465
coldchain_updated_time_after      = 1.3321198150515556
recorded_return_arrival_times     = [1.3321198150515556]
implied_depart_time               = 1.2553534507751465
clock_advanced_to_arrival         = true          # 冷链时钟确实推进到卸货时刻
order 32:
  predicted_quality_if_recorded_at_depart   = 1.0
  predicted_quality_if_recorded_at_arrival  = 0.9998590183371526
  recorded_delivered_quality                = 0.9998590183371526
  matches_depart_chain                      = false
  matches_arrival_chain                     = true
  return_leg_decay_arrival_minus_depart     = 1.4098166284737257e-04
```

**为什么怀疑不成立（代码层面）**：`_advance_fleet` 在写交付记录之前先调用 `_advance_coldchain_to(v, clock)`，其中 `clock == return_finish`（`strict_online_env.py:304`）。回程区间 `[depart, return_finish]` 的时间因此在**零长度**的 `transition_segment`（`:391-402`，`depart_time = arrival_time = clock`）之前就已被推进；`:391` 那一段只负责在正确时刻卸货。`coldchain_state.py:188` 的文档也与此一致："On a depot-return segment, cargo accrues loss until `arrival_time` and is then finalized and unloaded."

**数量级自洽**：class 0 在 18 °C 的 Arrhenius 速率 `k = 0.001·exp(3500·(1/277.15 − 1/291.15)) ≈ 1.836e-3 /h`；回程 0.07677 h ⇒ 预期衰减 ≈ `1.836e-3 × 0.07677 ≈ 1.41e-4`，与实测 `1.4098e-4` 吻合。

**顺带记录一个测量陷阱**：`return_finish` 被 `_advance_fleet` **消费掉**（`:428-429` 置 `None`），所以调用之后读回必然是 `None`（第一版正是踩到这里）。物理卸货时刻应以记录的 `return_arrival_time` 为准。

## 4. 稳定性扫描（6 个实例）

`--instances i --heavy-cap 20`，i = 0…5，每个实例的矩阵另存为 `raw/B3CF_MATRIX_instance{i}.json`，stdout 另存为 `raw/B3CF_failures_instance{i}.txt`：

| inst | F10 已服务客户 / 是否写入 | F8 S1 → S2 / 写入头 | F9 揭示次数 / reverify_allow | A-c7 越界 / 是否抛 | A-c5 回程 h / 记录品质 / 匹配arr / 匹配dep | AUDIT |
|---|---|---|---|---|---|---|
| 0 | 32 / true | 0b81882cb625 → 9ff1117d7b5e / 38 | 10 / false | 35.0 / – | 0.076766 / 0.9998590183371526 / true / false | OK |
| 1 | 50 / true | 3998838e47ca → cceb6cac7e4e / 1 | 9 / false | 35.0 / – | 0.193426 / 0.9996448099034957 / true / false | OK |
| 2 | 31 / true | 6198841a718d → 7f90d5260b16 / 11 | 11 / false | 35.0 / – | 0.294306 / 0.9994596144528394 / true / false | OK |
| 3 | 39 / true | 5c886b4f1319 → 2d5ce0c243d2 / 12 | 11 / false | 35.0 / – | 0.275701 / 0.9998247943867045 / true / false | OK |
| 4 | 22 / true | 551816dfa2e2 → d42fda4228d3 / 16 | 13 / false | 35.0 / – | 0.204580 / 0.9996243317345371 / true / false | OK |
| 5 | 12 / true | 9e6b248a0e44 → 7221be02554e / 12 | 11 / false | 35.0 / – | 0.028893 / 0.9999816084232294 / true / false | OK |

6/6 实例的 `verdict_counts` 完全相同（`3 REPRODUCED + 1 MECHANISM + 1 CONCLUSION_RECORDED + 1 NOT_REPRODUCED + 1 SETUP_OK + 2 OK`）。CP-A-c5 的"记录品质 == arrival 链、≠ depart 链"在 6/6 实例成立，回程时长跨度 0.029–0.294 h，所以这不是单一实例的巧合。

规范矩阵（`B3CF_FAILURE_PATH_MATRIX.json`）保存的是实例 0 的运行；其余实例的完整矩阵保存在 `raw/`。

扫描中修掉一个**真实 bug**（记录在案）：`_outcome_fp` 直接用 `t.dispatch_time` 等字段，在实例 2–5 上抛 `TypeError: Object of type float32 is not JSON serializable`（不同实例的数据类型不同）。修复为显式 `float()`/`int()` 强制转换。这也说明**只在实例 0 上跑会掩盖类型相关的脆弱性**。

## 5. 身份（可复现性）

矩阵 `identities`：

```
data_npz_sha256          c35fa6834b8007922160f7979680c1fb8c6dda4d93309cd5ef509b57d83f20f1
                         (data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz)
profile_hash             df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6
effective_contract_hash  edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833
prereg_spec_sha256       567726481b3b0e47f7dd85a925bd6c85eb20e82ccec17e41d0dda4e96e257749
code:
  scripts/tests/test_d3_b3cf_submit_failures.py   0303cb20b35bda9ff3791e15ddb9df934369a05e93f9c5c52d68f03adcb83a8d
  scripts/simulation/strict_online_env.py         fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6
  scripts/simulation/d3_install_gate.py           adc8df289be1babcf8d6907ff57ab30ab3d66d58fcb47a6c9539bc8591634483
  scripts/simulation/d3_composed_certifier.py     b65ddf6545a84b656cd2e9dd541cea266d2d6bdbc1cd56abd76f7780104aea01
  scripts/coldchain/coldchain_state.py            a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e
```

**字节容器修正（本批附带，v2 追加）**：`d3_g0_evidence_root_v1/` 此前**没有**自己的 `.gitattributes`（同级根目录都有）。加入 `* -text` 后实测：该目录 35 个受控文件中 30 个本来满足 `blob == worktree 原始字节`，5 个不满足（都是 PowerShell 重定向的 `raw/*_stdout.txt`，工作区 CRLF、索引 LF）。一次 `git add --renormalize` 后 35/35 满足。这 5 个文件的 LF/CRLF 两种 sha256 在仓库中**均无任何引用**（`git grep -l -F` 皆 0 命中），所以重规范化不改变任何已登记数值，旧 blob 仍留在历史中。逐文件清单与全部登记字节见 `B3CF_BYTE_FREEZE.json`；该记录**不登记自身哈希**（自引用无意义）。

### 5.1 提交后校验发现的 pin 缺口（v2；**先被发现、后才修好**）

第一次交付（commit `9710081`）只 pin 了 `research/.../d3_g0_evidence_root_v1/`，**漏掉了 `scripts/` 下三个本批登记 hash 的文件**。按约定必须做的提交后校验 `git archive HEAD <path>` → sha256 直接把它抓了出来（24 条登记路径中 3 条不可复现）：

| 路径 | `git archive` 得到 | 登记值 |
|---|---|---|
| `scripts/tests/test_d3_b3cf_submit_failures.py` | `251e07dc…` | `0303cb20…` |
| `scripts/simulation/strict_online_env.py` | `b6db355a…` | `fd17fe77…` |
| `scripts/coldchain/coldchain_state.py` | `d65ebb20…` | `a2ee98dc…` |

后两条是**先于本批存在的缺口**，不是本批造成的：这两个文件的 sha256 同时被更早的记录登记（`02_配送线/文档组件/2026-10-09/IMMUTABLE_BASELINE.json`、`docs/当前规划/决策与审计/a1_audit_source_fingerprints_2026-09-28.json`、`docs/当前规划/文献组件/三线协议审计_2026-10-09/PROTECTED_BEFORE.json`），因此那些记录同样**不能**从 `git archive`/全新 checkout 复现。现在 pin 上它们，那些记录也一并被修好。

**pin 是增量的**：三个文件在 pin 之前实测 `git hash-object --no-filters == index blob`，所以没有重写任何既有 blob、没有任何登记值被改变；pin 集合 18 → 21 个文件。

**故意没做**：`scripts/` 下的 `*.py -text` 一揽子 pin——该目录 438 个受控 `*.py` 中有 **11 个**当前工作区就是 CRLF（如 `tests/test_aggregate_cross_cell.py`、`training/auto_train.py`），一揽子 pin 会重写它们的 blob。窄的、逐文件登记的 pin 集合继续保持（34 个 CRLF `*.sh` 是最初就必须窄的原因）。

修好之后重跑全部 24 条登记路径的提交后校验：**24/24 从 `git archive HEAD` 逐字节复现**（commit `40b3cca`）。缺口、测量与修复证据同时写在 `C-VRP_Cold-chainVehicleRoutingProblem/scripts/.gitattributes` 的注释里和 `B3CF_BYTE_FREEZE.json` v2 的 `pin_sets[1].measured_before_pin` 中。

## 6. 登记项修正（不重写历史）

B3-CG 阶段把 `_force_return_all` 的时间口径列为"**另外登记、不得以零调用次数关闭**"的模拟器语义疑点，措辞为"记录的卸货早于物理 ⇒ 记录品质比物理乐观"。本批 CP-A-c5 给出**决定性的反向测量**：该怀疑在当前代码上不成立（§3）。因此：

- 该疑点状态由"未验证的语义缺陷"改为 **`NOT_REPRODUCED`（带正向测量）**，并附代码路径解释与数量级自洽检查。
- 同时确认它**与 F10 无关**：F10 复现的是"提交路径不做可行性检查"，而回程品质记账是**正确**的。不得用 CP-A-c5 的否定结果去冲淡 F10 的复现。
- 历史登记文本保持不变（append-only）：本报告与 `B3CG_SPEC_ADDENDUM.md` 是修正载体。

## 7. 本批得出与**未**得出的结论

**得出**：
- 提交路径旁路是**真实可执行**的缺陷，不是静态推断（CP-F10，6/6 实例复现）。
- 批准与写入之间的 TOCTOU 窗口**真实存在**（CP-F8，6/6 实例）。
- 凭证失效机制在真实揭示路径上**会拒绝**，但**没有生产调用者**（CP-F9，6/6 实例，`reverify_allow=false`）。
- `_validate_state` 不校验温度，热越界只计数（CP-A-c7）。
- hard_upper 在本配置下需要 45–360 次开门累积才能触及，`binds_free=false`（CP-A-c8）。
- "记录卸货早于物理"**不成立**（CP-A-c5，正向测量）。
- 探针不扰动生产轨迹（AUDIT，6/6 实例 OK）。

**未得出**（不得声称）：
- **G-L 不通过**：不是因为某个测试"成功复现缺陷"就自动标记，而是因为 F10 复现 + F8 复现 + 无唯一授权写入者；本批只提供证据，闸门判定仍为 `NOT_ESTABLISHED`。
- **G-C 不成立**：CP-F9 只证明机制存在，未证明它在任何生产路径上被执行。
- 任何效用/收益数值：本批不产生，`stable_online_gain` 仍为 `False`。
- CP-A-c8 的 `CONCLUSION_RECORDED` 不是"容量地板安全"，也不是"容量地板无关"。

## 8. 复现命令

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem
& D:\AA_Py\A_Anaconda\python.exe -B scripts/tests/test_d3_b3cf_submit_failures.py --instances 0 --heavy-cap 20
# 稳定性扫描
for i in 0 1 2 3 4 5; do python -B scripts/tests/test_d3_b3cf_submit_failures.py --instances $i --heavy-cap 20; done
```
