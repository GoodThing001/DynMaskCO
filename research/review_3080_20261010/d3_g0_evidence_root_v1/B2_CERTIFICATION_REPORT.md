# B2 三重身份与组合认证报告（受限实现 + D1–D8 预注册验收）

**日期**：2026-10-14 · **基准 commit**：`2176af6` · **授权**：受限 B2（用户裁决 2026-10-14）
**性质**：新增 B2 身份/组合认证/安装前复验模块；**不接管真实提交循环**、**不启用写回**、**不声称状态级 rollback**、**不声称跨进程可复核**。
**执行顺序**（按用户要求）：① 静态核查 → ② 预注册 D1–D8 → ③ 实现组合认证。

| 产物 | 作用 |
|---|---|
| `research/…/d3_g0_evidence_root_v1/B2_STATIC_REVIEW.md` | 第一步：静态/经验核查（import 副作用、合同 vs 有效合同、能耗口径、重复计入与未来信息、质量可检性） |
| `research/…/d3_g0_evidence_root_v1/B2_COUNTEREXAMPLE_SPEC.json` | 第二步：**结果产生之前**冻结的 D1–D8 + SCOPE 预注册 |
| `C-VRP_…/scripts/simulation/d3_credential.py` | **B2-1** `CertificationCredentialV1` + `state_identity_information_scope` |
| `C-VRP_…/scripts/simulation/d3_composed_certifier.py` | **B2-2** 组合认证（4 项检查 + 4 项补偿 + 质量能力缺口三态） |
| `C-VRP_…/scripts/simulation/d3_install_gate.py` | **B2-3** 安装前身份复验（`H(S)/H(C)/H(P)` 三重相等） |
| `C-VRP_…/scripts/tests/test_d3_b2_certification.py` | D1–D8 + SCOPE 验收（12/12 PASS，exit 0，stderr 0 字节） |
| `research/…/d3_g0_evidence_root_v1/B2_CERTIFICATION_MATRIX.json` + `raw/` | 机器可读矩阵与原始输出 |

---

## 0. 结论

**D1–D8 + SCOPE 全部 PASS**（`all_pass: true`，`covered_ids` = PRE,D1..D8,SCOPE,SUF,PREREG）。
预注册完整性检查 `PREREG` **通过**：九个预注册 id 全部存在，无一被删。

| 预注册项 | 结果 | 关键实测 |
|---|---|---|
| **D1** 认证后状态漂移 | ✅ 拒绝 | `state_hash` 不同 ⇒ `state_hash_mismatch` |
| **D2** 认证后合同/profile 改变 | ✅ 拒绝 | V2 合同 ⇒ `contract_hash_mismatch` **+** `effective_contract_hash_mismatch`；换 profile ⇒ `profile_hash_mismatch` |
| **D3** pre/post-prepare 混用 | ✅ 显式标注并拒绝 | `state_capture_anchor` 字段在场；两者 `state_hash` 不同 ⇒ `anchor_mismatch` |
| **D4** `anchor_time` 变、旧哈希不变 | ✅ 强身份检出 | `plan_hash_legacy` 相同（16 hex）而 `plan_digest_anchor_v1` 不同 ⇒ `plan_digest_anchor_mismatch` |
| **D5** TW / 容量不满足 | ✅ 组合拒绝 | 分别报 `reason='tw'` / `'capacity'` |
| **D6** C0 能耗超预算 | ✅ 组合拒绝 | B=0 ⇒ 能耗失败（721.93 kWh）**而路线仍然可行** ⇒ 两项独立，非互相遮蔽 |
| **D7** 承诺/所有权/protected 违规 | ✅ 三者各自拒绝 | 重规划 committed ⇒ `committed_in_suffix`；丢单 ⇒ `missing=[7]`；保护车被动 ⇒ `protected_vehicle_changed=[1]` 且 `full_partition_ok=true`（归因干净） |
| **D8** 质量/鱼可售 | ✅ 标记能力缺口**并**拒绝可安装性 | 无在线检查 ⇒ `CAPABILITY_GAP_BLOCKING` + `installable=False`；注入失败检查 ⇒ 拒绝 |
| **SCOPE** 状态身份不得引入未来信息 | ✅ | 53 个快照键分类；`visible_mask` 与 `{c: reveal_time[c] ≤ clock}` 一致；`unrevealed_customers_contributing=[]`；快照**不含**任何未来订单属性数组；翻转未揭示客户的可见位确实改变哈希（可见集被绑定） |
| **SUF** 身份相等**不充分** | ✅ | 两份**完全相同**的凭据，在 `deadline_ok=False` / `certification_complete=False` 下仍被拒绝 |

---

## 1. 静态核查的关键结论（第一步）

1. **import 副作用（实测）**：`run_exp_energy_c0` 可安全 import —— 无实验执行（`main` 在 `:582`、`__main__` 守卫在 `:715`）、**无文件写入**（`scripts/results/data` 的文件数与最新 mtime 前后完全一致）、耗时 0.151 s。
   **但有一个真实全局副作用**：import 会向 `sys.path` **插入 3 条**（`:27/:30/:33`），并传递导入另一个实验入口 `run_exp_reserve`。
   ⇒ 装配层自己建路径，不依赖 import 副作用；该耦合登记为已知风险（R3）。
2. **合同 ≠ 有效合同（实测）**：`raw=32124bfd…` vs `effective=edc1b2ea…`（本项目的 frozen profile 下**确实不同**），而 **env 只携带有效合同**。
   ⇒ 凭据必须**分别**登记两者；原始合同由调用方注入，缺失时记 `raw_contract_reachable=false`，绝不拿有效合同冒充原始合同。
3. **能耗口径完整**：`certify_plan` 对 committed 车把 `committed_next` **恰好前置一次**（`:176-181`），`c0_completion_energy` 含 **WAIT 到 booking_end**（`:139-144`）与**返仓段**（`:145-150`）；与 `build_vehicle_plans`（suffix 不含承诺段、丢弃 closed 车）**互补不冲突**。未发现重复计入。
4. **发现 1 处未来信息读取 + 2 处静默**（组合认证器已逐条补偿）：
   - **F1** `c0_completion_energy:127` `depart = max(t, float(rev[o]))` 读 `reveal_time`：对已揭示订单惰性，但**不会拒绝**未揭示订单 ⇒ 组合认证器**自己**断言"计划内订单在 `clock` 已揭示"。
   - **F2** `certify_plan:173` 把已服务订单**静默剔除** ⇒ 由 `validate_full_partition(served_mask=…)` 显式拒绝。
   - **F3** closed 车的计划条目被静默忽略 ⇒ 断言计划车辆集合 ⊆ 活跃车集合。
5. **质量/可售性在决策点没有可执行检查**：`certify_route` 只查**温区支持**（`:178-183`），而 `projected_energy_kwh / projected_quality_loss / projected_thermal_margin` 全部是 `None`（`:171-173`）；唯一的可售检查是**终端**的 `evaluate_trace_a1:115-117`。
   ⇒ 按你的裁决：**标记为阻断性能力缺口**，不得默认合格。

---

## 2. 三个模块的边界（严格限定）

- **B2-1 凭据**：`state_hash`（+`state_hash_scope='in_process_same_run_only'`、`state_capture_anchor='post_prepare'`）、`contract_hash`（原始，可注入）、`effective_contract_hash`、`profile_hash`、`plan_hash_legacy` **与** `plan_digest_anchor_v1`（含 `anchor_time`/`anchor_load`）、`anchor_fields`、决策坐标、`source_identity`、认证结果、`state_identity_information_scope`。
  **旧 `plan_hash` 保留为历史对照，未被替换或重新定义**；新摘要 schema 名为 `plan_digest_anchor_v1`。
- **B2-2 组合认证**：4 项检查（`validate_repair_plan` / `validate_full_partition` / `certify_route` / `certify_plan`）
  **+** 4 项补偿（`revealed_only` / `no_served_in_plan` / `plan_vehicle_set` / `no_duplicate_order`）
  **+** 质量三态（`CHECKED_SATISFIED` / `CHECKED_FAILED` / `CAPABILITY_GAP_BLOCKING`）。
  返回 `ok`（所有可执行检查通过）与 `installable`（**并且**无阻断性能力缺口）两个**不同**的判定，并明确写下"调用方不得单独把 `ok` 当安装许可"。
- **B2-3 安装前复验**：要求 `H(S)/H(C)/H(P)` **三项全等**，**且** `deadline_ok is not False`、`certification_complete is True`。
  `allow=True` 的含义被限定为"没有任何已知项拒绝安装"，**不是**"环境已安装"，**不是**"状态已提交"。
  模块显式列出**它做不到的三件事**：实际安装结果（B3）、状态级 rollback、持久回执。

---

## 3. 实现过程中发现并修掉的**两个我自己的真缺陷**

1. **组合认证器在畸形计划上崩溃**：D7(c) 注入了一个非法客户 id，`validate_full_partition` 索引 `served_mask` 越界抛 `IndexError`。
   已把每个分量置于 `_guard` 之下：**异常变成拒绝并记录异常类型**，绝不逃逸。理由与 B1 一致——会崩溃的闸门比会拒绝的更糟。
2. **"双真相"导致分量自相矛盾**：`route_feasible` 用并行的 `bad` 标志与 `per_route` 字典两处表示同一判定，实测出现"报告里 25 条路线全可行、分量却 `ok=False`"。
   已改为**从被返回的数据本身推导 `ok`**（`infeasible_routes` 列表），使该类不一致在结构上不可能。修好后 D6 立即干净通过。
3. **`base_plans` 默认取候选本身 ⇒ 对账检查空转**：`compose_certification(plans=X)` 默认 `base_plans=X`，于是 `validate_repair_plan` 变成**自我比较**、恒真。
   这正是 D7(c) 第一版"保护车被动却 `repair_plan_ok=true`"的原因。已改为 **fail-closed**：未提供 incumbent 记为 `base_plans_missing` 并**拒绝**（`repair_plan_unverifiable`），测试也改为显式传入 incumbent。修好后 D7(c) 归因干净（`protected_vehicle_changed=[1]`、`full_partition_ok=true`）。

另修一处测试构造错误：D2 原本用**同一个 env** 建两份凭据，而 `make_credential` 从 env 读有效合同 ⇒ 有效哈希必然相同，测不出"有效合同漂移"。已改为用**第二个 env**（V2 合同 / 换 profile）构造安装期凭据。

---

## 4. 用户风险提醒的逐条落实

| 提醒 | 落实 |
|---|---|
| import 是否有顶层副作用 | 已实测：无实验/无写盘；**但会插入 3 条 `sys.path` 并传递导入 `run_exp_reserve`** —— 装配层自建路径，耦合登记 |
| 合同与有效 profile 是否真正一致 | 已实测**不同**（`32124bfd…` vs `edc1b2ea…`），env 只有有效合同 ⇒ 两者分别登记，原始合同可注入 |
| 路线能耗是否含承诺路段与返仓 | 已读源码确认**含**，且**恰好一次**，不重复计入 |
| 是否重复计入稀缺资源或使用未来信息 | 发现 **F1 未来信息读取**（已由组合认证器补偿）、**F2/F3 静默**（已显式拒绝）；未发现资源重复计入 |
| 是否"四个函数调用了就算冷链硬约束全覆盖" | **明确否定**：D8 把质量/可售记为**阻断性能力缺口**并拒绝可安装性 |

---

## 5. 本轮**没有**做（边界）

未修改 `_commit` / `_plan_and_commit` 的生产提交语义；未启用真实安装写回；**未声称状态级 rollback 已实现**；未把同进程哈希当作跨进程可验证身份（凭据里写明 `in_process_same_run_only`）；未做持久化回执；未启动 E2 / F−D / 历史 32 状态重采集 / G1 / G2 / 最终测试；未训练、未跑效果实验。
所有反例均在**真实 `StrictOnlineEnv` 的独立副本**上运行，`PRE` 检查确认 live 指纹在捕获前后不变。

**成功标准对照**：给定真实状态、合同与候选计划，系统能生成可复核的认证凭据，且**状态/合同/计划任何漂移或硬约束失败都会在写回之前被拒绝** ✅。
**它不证明**真实环境已完成安装 —— 那属于 B3。

---

## 6. 登记身份

| 项 | 值 |
|---|---|
| cal 数据 sha256 | `c35fa6834b8007922160f7979680c1fb8c6dda4d93309cd5ef509b57d83f20f1` |
| `profile_hash` | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| 原始合同 | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0` |
| 有效合同 | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` |
| `d3_credential.py` | `85f32d67ad38f1a88be6aa5a8c4c4af3f48797df6a1ca088d05c672453834c7f` |
| `d3_composed_certifier.py` | `b93c81756adcd15b0cdac4ca98771c5cb6e354d083f2d0f094d344766571a1c9` |
| `d3_install_gate.py` | `adc8df289be1babcf8d6907ff57ab30ab3d66d58fcb47a6c9539bc8591634483` |
| `test_d3_b2_certification.py` | `285bdc928b2f7ee7925b123e2bf61bea18da5acfd89d82f66ffc5fad4ef8f6dd` |

---

## 7. 复现

```powershell
cd D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
& D:\AA_Py\A_Anaconda\python.exe -B scripts\tests\test_d3_b2_certification.py `
    --instances 0 --heavy-cap 6          # exit 0, stderr empty
```

（B2 为纯工程工作，**不需要 GPU**；5xxx 卡相关的训练/效果实验仍未授权。）

---

## 8. 研究位置

G0-Safety 第一批 ✅ → B1 ✅（真实状态反例 + no-op 装配）→ **B2（本报告）** → B3（真实安装、提交一致性、持久回执）→ G0 独立验收 → G1 工程烟测 → G2 科学评价。
**B1/B2/B3 全部完成也不自动意味着 G0 通过**：之后仍需身份冻结、资源门、实际调用链与独立只读验收；历史 32 状态恢复与 E2 训练是另外的资格事项。

当前不变：**D3 G0 未通过**，G1/G2 未启动，**无可归因的 F−D 学习增益**，I3 正式负结果 / 旧 D2 `not_qualified` / `stable_online_gain=False`。
