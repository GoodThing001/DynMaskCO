# B3a：真实提交边界观察器 + 持久化回执设计（只读设计）

**日期**：2026-10-14 · **基准 commit**：`3038a16` · **性质**：只读设计。**不启用写回、不改 `_commit`、不改 `_plan_and_commit`、不启动科学实验。**
**用户要求**：在不改变 `_commit` 历史语义的条件下，分别识别 `callback_invoked → suffix_written → head_committed → verified_consistent`；并设计含四个计划与三重身份的回执，其中 **`P_after_install` 必须来自实际提交后的环境读取**。

---

## 1. 一个必须先讲清的**结构性约束**（决定回执能写到什么程度）

我枚举了 `StrictOnlineEnv` 的**全部**扩展点（`strict_online_env.py`）：

| 扩展点 | 位置 | 调用时机 |
|---|---|---|
| `snapshot_hook` | `:179` / 调用 `:552-554` | **`_plan_and_commit` 之前** |
| `oracle_hook` | `:183` / 调用 `:467`、`:570`、`:600` | **`_plan_and_commit` 之前** |
| `force_suffix` | `:175` / 应用 `:639-643` | **输入覆盖**，不是观察点 |
| `run_resumed` | `:573` | 从快照续跑，不是钩子 |

而真实提交发生在 **`_plan_and_commit` 内部**：`:650-656` 逐个 pop `mutable_suffix[0]` 并调用 `self._commit(...)`（`:260-299`），**晚于两个钩子**。

⇒ **结论（必须如实报告）**：**在当前扩展点下，同一决策点的"提交后状态"无法被观测。**
这不是设计疏忽，而是扩展点的位置决定的。B3a 必须就此做出明确选择，而不是假装能观测。

### 三个可选方案（附代价）

| 方案 | 做法 | 能否拿到同决策点提交后状态 | 是否改生产源码 | 代价 |
|---|---|---|---|---|
| **(a) 下一决策点回读** | 在事件 *k+1* 的 `snapshot_hook` 里读 `v.committed_next` / `v.status` / `v.served_route` / `traces[*].services`，反推事件 *k* 的提交 | ❌ 滞后一步（且区间内 `_advance_fleet` 可能已让 leg 完成） | 否 | 完全只读；需从 `served_route`+`traces` 反推，属**间接但可核**证据 |
| **(b) 测试侧包装 `_plan_and_commit`** | 观察器在**测试/harness 内**用包装函数替换绑定方法：记录前态 → 调原函数 → 记录后态 | ✅ 精确 | **否** | 是对调用路径的**运行期干预**，只能用于"观测方案验证"，**不得**当作"生产路径本来就能观测"的证据 |
| **(c) 新增授权的 post-commit 钩子** | 在 `strict_online_env.py` 的提交循环之后加一个 `post_commit_hook`（`None` 默认可关） | ✅ 精确 | **是**（生产 env 变更） | 最干净，但**需要单独授权**；且属"改 env"，与当前"不改 `_commit`"的边界相邻——**加钩子不改 `_commit` 本体**，但仍需批准 |

**建议**：B3a 现在用 **(a) + (b)** 组合——(a) 作为**长期可用、零侵入**的证据通道，(b) 用于**验证观测方案本身是否完备**；**(c) 留待 B3b 真实写回授权时一并决定**（若批准，则回执的 `head_committed` 可以是精确同决策点证据）。
**不得**用 (b) 的结论去声称"回执已经满足独立离线重验"。

---

## 2. 四态观察器规格

四态必须**分别**判定、分别留证，且每一态都写明**证据来源**（这是本设计的核心）。

| 态 | 语义 | 判据 | 证据来源 |
|---|---|---|---|
| `callback_invoked` | 安装回调被调用 | `InstallLedger.n_installs == 1` 且回执存在 | 账本（**只证明回调执行**） |
| `suffix_written` | 计划已写入 **live** `VehicleState.mutable_suffix` | 逐车 `list(v.mutable_suffix) == 计划 suffix`（**写回后、commit 前**） | `replanner.plan(...)` 返回后、`_plan_and_commit` 的 force/commit 之前 |
| `head_committed` | env 已 pop 并 `_commit` 首个节点 | `v.committed_next == 计划头` 且 `v.status == 'committed'`（头为 0 ⇒ `returning`/`idle`） | **(b) 精确**：包装后立刻读；**(a) 滞后**：下一决策点从 `committed_next`/`served_route`/`traces` 反推 |
| `verified_consistent` | 提交后状态与认证计划一致且合法 | `served_route`/`trace.services` 反映该客户；`validate_full_partition` 通过；**且 `P_after_install` 与 `P_certified` 的差异被逐条解释** | 提交后重建的 `FleetPlan`（见 §3 `P_after_install`） |

**必须显式记录的失败模式**（B1 的 C5 已在副本上证实其机制）：
`callback_invoked=True` 而 `head_committed=False` —— 写进的 `mutable_suffix` 在下一次 reveal 被清空（`:528`），或 `v.status` 已不是 `idle`/`ready` 使提交循环跳过该车（`:651`），**无任何记录**。
⇒ 回执的 `outcome` 必须能表达这一态，而不是把它折叠成"已安装"。

**`outcome` 取值（建议，与 B2-3 的措辞一致）**
`installed_and_committed` / `callback_invoked_but_not_committed` / `suffix_written_but_not_committed` / `refused` / `rolled_back_plan_level_only`

---

## 3. 持久化回执 schema（v1 草案）

```
InstallReceiptV1 = {
  receipt_schema_version: 'install_receipt_v1',
  run_identity: {inst_idx, event_id, clock, reveal_idx, run_id?},

  # ── 四个计划：P_after_install 必须来自提交后的环境读取 ──────────────────────
  P_before:        {plan_hash_legacy, plan_digest_anchor_v1, plans},
  P_proposed:      {plan_hash_legacy, plan_digest_anchor_v1, plans, source},
  P_certified:     {plan_hash_legacy, plan_digest_anchor_v1, plans, credential_hash},
  P_after_install: {plan_hash_legacy, plan_digest_anchor_v1, plans,
                    read_from: 'env_after_commit' | 'env_next_decision_point',
                    read_at: {event_id, clock}},
  #   ⚠️ 禁止用 P_certified 复制生成；必须由 build_vehicle_plans(env, inst, vehicles) 重新读取
  #   ⚠️ read_from 必须如实标注它是同决策点读取还是下一决策点回读

  # ── 四态（§2）────────────────────────────────────────────────────────────
  install_states: {callback_invoked, suffix_written, head_committed, verified_consistent,
                   evidence_source: 'wrapped_plan_and_commit' | 'next_decision_point'},

  # ── 三重身份（B2 凭据）+ 复验结果 ────────────────────────────────────────
  credential: CertificationCredentialV1,
  identity_recheck: {state_hash:{ok,cert,install}, contract_hash:{...},
                     effective_contract_hash:{...}, profile_hash:{...},
                     plan_digest_anchor_v1:{...}, plan_hash_legacy:{...}, anchor_ok, instance_ok},

  # ── 合同来源与变换证明（用户保留 3）──────────────────────────────────────
  contract_provenance: {
    raw_contract_hash, raw_contract_source: 'injected'|'unavailable',
    raw_contract_file_sha256?,           # 若来自文件
    profile_hash,
    transform_check: {applied_equals_effective: bool,
                      recomputed_effective: str, env_effective: str},
    #  证明 apply_objective_profile(C_raw, p) ≡ C_effective，而不是只记三个哈希
  },

  # ── 完整认证结果（B2 组合认证的逐项输出）─────────────────────────────────
  certification: {ok, installable, components:{...}, compensation_results:{...},
                  quality:{state, detail}, blocking_capability_gaps, reasons},

  # ── 提交/失败原因 ────────────────────────────────────────────────────────
  outcome, refusal_or_failure_reason,
  deadline: {wall_seconds, limit_s, deadline_ok},
  exception: {type, message} | None,
}
```

### 3.1 字段级硬性要求（来自用户裁定）

1. **`P_after_install` 不得由 `P_certified` 复制**：必须 `build_vehicle_plans(env, inst_idx, vehicles)` 在提交后重新读取；并在 `read_from` 里**如实标注**是同决策点还是下一决策点。
2. **四态分别留证**，且 `evidence_source` 必须写明；用方案 (b) 时**不得**声称生产路径本身可观测。
3. **`transform_check` 必须实际执行** `apply_objective_profile(C_raw, profile)` 并与 `env.coldchain_contract.contract_hash` 比较（见 §4）。
4. **回执不得声称**：状态级 rollback、跨进程可复核、真实环境已安装（除非 `verified_consistent` 为真且 `read_from` 精确）。

### 3.2 独立重载的定义（与 B1 的 R4 一致，不夸大）

- **可做**：同进程内从回执重算 `state_hash` / `plan_hash_legacy` / `plan_digest_anchor_v1` 并比对；
  `P_after_install` 若为下一决策点回读，则可用 `traces` 交叉核验。
- **不可做（本轮）**：跨进程复算 `state_hash`（JSON 往返不复现，见 B1 R4）。**不得**把同进程哈希当作跨进程身份。

---

## 4. 用户保留 3 的落实：raw/effective 合同**变换证明**

现状（B2）：env 只携带有效合同，原始合同由调用方注入，凭据记三个哈希。
**不足**：三个哈希**不能**证明它们来自**同一次合法变换**——可能记录的是互不相干的三个值。

**B3a 规格**：提交入口必须
1. **提供原始合同来源**：`raw_contract_source ∈ {'injected_object','file','unavailable'}`；若来自文件，同时记 `raw_contract_file_sha256`；
2. **实际执行变换并比对**：
   `apply_objective_profile(C_raw, p).contract_hash == env.coldchain_contract.contract_hash`
   （`apply_objective_profile` 见 `coldchain_contract.py:481-485`；`contract_hash` 覆盖 `asdict` 全字段，故哈希相等即**逐字段相等**）；
3. 若 `raw_contract_source == 'unavailable'` ⇒ **不得**声称变换已证明，回执必须带 `transform_check: NOT_VERIFIED`。
4. 补充（建议，B3b）：原始合同若来自文件，则用 `load_coldchain_contract(path)` 的**重算-比对**（`coldchain_contract.py:341-344`）证明文件自洽。

---

## 5. 观察器的实现边界（本轮不做）

本轮**不实现**观察器与回执（只读设计要求）。实现时：
- 方案 (a)：纯只读，放在 harness 的 `snapshot_hook` 内，与 B1 的 `Grab` 同构；
- 方案 (b)：包装必须**位于测试/harness**，并在回执里把 `evidence_source` 标为 `wrapped_plan_and_commit`；
- 方案 (c)：需**单独授权**；加钩子**不修改 `_commit` 本体**，但属于生产 env 变更。

**停止条件（用户裁定）**：完成真实提交边界设计 + 回执 schema + 质量认证可行性结论即停止。本文件交付前两项。

---

## 6. 本文件没有做的事

未实现观察器/回执；未改 `_commit` / `_plan_and_commit` / env；未启用写回；未声称状态级 rollback；未声称跨进程可复核；未启动科学实验。
