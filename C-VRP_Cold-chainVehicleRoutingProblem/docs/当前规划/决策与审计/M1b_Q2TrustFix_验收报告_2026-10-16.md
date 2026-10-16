# M1-b / Q2-TrustFix 验收报告（2026-10-16）

**基准**：`84e378968de2e63bd2da302d70fbf9c45c3dbfec`（`git fetch` 后本地 `HEAD`、
`origin/main`、`origin/codex/o0cc-cal-phys` 三者相同；工作区仅 `CC_Compare/MAPT_code` 既有脏状态）。

**性质**：依据 **Q2-Fix-T 已冻结的 17 例负控**集中修复信任边界。**未**正式实验、训练、在线 rollout、
生产计划安装、Q3 签署、M2；**未**裁定 `B`；**未**使用 GPU；**未**重跑正式科研实验。

**未修改**：旧报告、旧结果、冻结数据、M1-a、L1 裁决、`MASKCO_code/`
（`git diff` 对 `results/_m1b_q2/negative_controls_q2r.json`、`M1b_Q2FixT_信任边界审计…md`、
`m1a_adapter` 为空）。

---

## 0. 身份（修复前 / 修复后）

| 文件 | 修复前 sha256（截断） | **修复后 sha256** |
|---|---|---|
| `m1b_timing.py` | `ba77a44a…` | `ba77a44a1b1c0ed5abbae3b91e0d39a1b7b055256243489fa92c4553ba4bd83a`（未变） |
| `m1b_quality_certificate.py` | `2aa801d0…` | `2aa801d09c26e6851e75ca4c1672ff9f84f1d59d3e5b95d402fcfc8f96ba4d22`（未变） |
| `m1b_return_feasibility.py` | `44138e0c…` | `45f9b623ee1e5cb24a9eaaee7e2161a017a2648aced8942d201d99d1b09fe2e2` |
| `m1b_thermal_gate.py` | `5ab1841f…` | `a626033990f5c259b33f44788100dc1f6cd54b6624bf80bc251f749596510955` |
| `m1b_q2_certifier.py` | `702bae2a…` | `54aff14d26d1a9a625420ceb7748446cd9bb4175d8bf1882049dffdd0bc6a65b` |
| **`m1b_public_projection.py`（新）** | — | `e4776ae69cd231fe9fbc8d6533a1121dd7d0ec782091c0f98c519c6cdc26f4fe` |
| `test_m1b_q2fixt_trust_boundary.py` | `f9b7…`（HEAD 版） | `503adfd3be5040012734f5b3f2918dec7cc6c36462cba4fe5f806d3f4e80c070` |
| `test_m1b_q2fix_certification.py` | `2f9ca4a6…` | `6d1f953feca917ddd4d1b96046c4a49351eee2859f16ffd1b529d37ccd3c32eb` |
| `M1b_Q2TrustFix_修复规格_2026-10-16.md`（新，**先于实现**） | — | `97962f8e1693beaee6b7d8fb29db2fd8c651f9e0c393eea659f4503ebe8304aa` |
| `results/_m1b_q2fix/trust_boundary_audit.json` | — | `030eef9946fd39e376af4b086e133300ce8c91175d67d76e51aa07b777e6bc37` |
| `results/_m1b_q2fix/regression_under_trustfix.json` | — | `5bf866929b21ea986c85381afe108cfb2d0c451bdac1c620779e859df94234ba` |
| `results/_m1b_q2/negative_controls_q2r_rerun_under_trustfix.json` | — | `361d44ee17615d4b61259f8c1afb584056590536ffc45d6c89de81954b63cd69` |

**新身份**：`M1B_Q2TRUSTFIX`，收据 schema `m1b_q2trustfix_engineering_receipt_v1`，
收据内含 `implementation_sha256` 与 `pre_fix_source_sha256`（修复前哈希的**文字引用**）。

**为什么修复前哈希只记在本报告里、不写进源码**：把 64 位十六进制摘要的字面量表放进源码会触发
仓库的暂存区密钥扫描器（规则 `R2_bare_64_hex_token`）——这是一次**假阳性**（那些是**内容哈希**，
不是凭据）。处理方式是**移除触发源**，而**不是**给扫描器加豁免或使用 `--skip-secret-scan`：
源码内的 `PRE_FIX_SOURCE_SHA256_DOC` 改为一句话引用（指回本报告与修复规格），
本报告仍是这些哈希的**唯一登记副本**。扫描结果：`STAGED_CONTENT_CLEAN`，`n_findings: 0`。

---

## 1. 必须报告的独立结论

| # | 结论 | 判定 |
|---|---|---|
| 1 | `TRUST_FALSE_PASS_RESOLVED` | **成立** |
| 2 | `FLEET_FALSE_REJECT_RESOLVED` | **成立** |
| 3 | `STRICT_ONLINE_BOUNDARY_VERIFIED` | **成立**（限定见 §5） |
| 4 | `THERMAL_CONTINUOUS_NOT_ESTABLISHED` | **成立**（本轮无真实数学证明） |
| 5 | `FULL_INSTALLABILITY_NOT_AVAILABLE` | **成立** |

**注**：第 3 条的"可信"是 **strict-online 软件协议下的可信边界**，
**不**声称 Python 进程对任意恶意代码具有密码学隔离（规格 §5 已写明）。

---

## 2. F-T1：连续证明钩子 fail-closed

| 机制 | 实现 |
|---|---|
| 生产路径拒绝钩子 | `M1bQ2TrustFixCertifier` 记住 `continuous_extremum_proof`，在 `certify()` **首部**直接拒绝（`PRODUCTION_CERTIFIER_REFUSES_PROOF_HOOK`），**不产生**任何证书 |
| 低层门双重保险 | `replay_thermal_envelope` 的连续分支额外要求 `allow_test_only_proof=True`；**默认 False** |
| 隔离测试入口 | `M1bQ2TrustFixTestOnlyCertifier`：收据标 `test_only=True` + `TEST_ONLY_LABEL` + `not_consumable_as_production_certificate=True`，**`composite_pass` 恒 False** |
| 采样/连续独立 | 两个顶层字段继续分开；采样违规仍短路为 `THERMAL_CONTINUOUS_VIOLATED` |
| 不靠 SHA256/字符串/自报 | **没有**任何登记面；唯一通往 `THERMAL_CONTINUOUS_OK` 的是测试专用 opt-in |

**证据**：`trust_boundary_audit.json` 的 **T1-1**（钩子被拒 + 连续级 `NOT_ESTABLISHED` + 复合 False）、
**T1-1b**（低层门忽略注入）、**T1-1c**（测试入口标 TEST_ONLY）、**T1-2**（无身份校验面）、
**T1-3**（采样独立）、**T1-4**（采样违规仍致复合失败）。
Q2-Fix 的 `C1-POS` 同步改为断言"测试入口可达接线但不可认证"。

---

## 3. F-T2：关闭 Ω11 直接调用绕过

| 变更 | 内容 |
|---|---|
| `certify_return_feasibility` | `times is None` ⇒ **立即拒绝** `OMEGA11_TIME_INPUT_UNAVAILABLE`，**删除**自动回退 |
| `register_service_times` | **保留为导入兼容接口**；其表**不进入任何判定**（源码级断言：无 `_SERVICE_TIME_CACHE.get(`） |
| 不自动构造 `tw_start=0` | 已删除 `_LegacyServiceTimes` 的自动咨询路径 |
| 热门的同类回退 | **一并隔离**：`replay_thermal_envelope` 不再接受裸 callable，缺少 `tw_start_of`/`service_time_of` 即拒绝 |
| 真实零服务时长 | **仍接受**（键存在、值为 0）——正控 `POS-1` |

**证据**：Q2-Fix-T **T2-1** 由 `FALSE_PASS` 翻转为 `OK`（拒绝）；
Q2-Fix `A2/A2b` 保持拒绝；`POS-1` 零服务时长正控通过。

---

## 4. F-T3：可信 strict-online 数据投影

新模块 `m1b_public_projection.py`，固定构造边界 `build_projection_from_env`：

| 要求 | 实现 |
|---|---|
| 可见集合来自环境当前事件 | 由 `env.reveal_time ≤ clock + 1e-6` **自行派生** |
| 只有构造层读 reveal 状态 | 投影只交出**已揭示**节点；未揭示节点 getter 抛 `PermissionError` |
| 只复制必要字段 | 白名单 `temp_class/demand/tw_start/tw_end/service_time/initial_quality`，**且只对已揭示节点** |
| 不留全节点真值表 | 内部映射**仅含已揭示节点**（T3-8 强断言 `len(_iq)==1`） |
| 构造后不可变 | `MappingProxyType`；原地写抛 `TypeError` |
| `source_identity` 绑定事实 | **计算**自 `(实例, 事件, clock, 可见集合, contract_hash, snapshot_identity)`，**非自由文本** |
| 自述不取得信任 | 生产认证器**拒绝**任何非 `TrustedPublicProjection` 输入（T3-7） |
| 后续揭示生成新快照 | 每事件独立投影；跨事件查询被拒、身份不同（T3-9） |
| 认证前后身份一致性 | 收据记录 `projection_identity_before/after` 并断言稳定 |

**证据**：T3-6（与环境矛盾 ⇒ 拒绝）、T3-7（自述 provider 被拒）、
T3-8（仅留已揭示 + 不可变）、T3-9（跨事件隔离 + 身份独立）。

**残留（如实保留）**：旧类型 `QC.PublicOrderAttributes` **仍存在**且仍是自述可变的（T3-2/T3-3/T3-4
标为 `NOT_ESTABLISHED_FOR_THE_LEGACY_TYPE_ONLY`）。它**不能**进入认证，但仍是可被误用的弱类型。

---

## 5. F-T4：完整车队状态与 `NOT_APPLICABLE`

新增 `classify_thermal_applicability`：

| 情形 | 判定 |
|---|---|
| 从未 dispatch + `idle` + 在 depot + 无货 + 无承诺 + 无候选后缀 + 无待维持热状态 | **`THERMAL_NOT_APPLICABLE`** |
| 未派车但**候选要出车** | **拒绝** |
| **有货但无冷链状态** | **拒绝**（原始负控保持，T4-4） |
| 有未完成承诺但缺状态 | **拒绝** |
| `returning` / `committed` | 遵守真实在途及后缀约束（`committed` 从 `committed_finish` 续放后缀） |
| `closed` 无货无待执行 | **`NOT_APPLICABLE`**（历史返仓由 Ω11 的 closed-history 门审计，不重复热重放） |

**复合规则**：仅**适用**车辆须通过热门；N/A 车辆**不计为通过、也不拖累**。
证据：T4-2（三车混合：v1 = `NOT_APPLICABLE`，采样级 `THERMAL_SAMPLED_OK`）、
T4-3（单车 idle 不再拖垮采样级；复合 False 的**唯一**原因是设计上的连续级 fail-closed）、
T4-4（有状态缺失仍拒绝）。

---

## 6. 回归结果（机械核验）

| 测试 | 结果 |
|---|---|
| **Q2-Fix-T 信任边界审计** | **23 checks：0 FALSE_PASS、0 FALSE_REJECT、0 NOT_ESTABLISHED**（3 项为"旧类型限定"标签） |
| **Q2-Fix 21 项** | **`Q2FIX_ALL_PASS`：22 checks、0 不通过、FALSE_REJECT = 0** |
| **原 Q2-R 16 例（未改测试文件）** | **0 FALSE_PASS**、0 FALSE_REJECT；A1/A2b/A4b = `OK`，B1 不再复现为假通过 |
| **M1-a S-1…S-17** | `M1A_SYNTHETIC_ENGINEERING_ACCEPTED`，`ENGINE_UNMODIFIED` |
| **M1-a 独立验收** | `VERIFIED`，9 checks / 0 disagreement |
| **Q2-Fix 独立验收** | `VERIFIED`，13 checks / 0 disagreement |
| **连续温度等级** | **`THERMAL_CONTINUOUS_NOT_ESTABLISHED`**（唯一允许的当前状态） |

### 旧判据不可继承事项（**不得**读作原检查仍通过）

1. 原 Q2-Fix-T 的 **T1-1** 判据（"注入 `(True,{})` ⇒ `CONTINUOUS_OK`"）**不可继承**——
   该行为已按 F-T1 移除；
2. 原 **T2-1** 判据（"Ω11 直接调用静默乐观"）**不可继承**——现拒绝；
3. 原 **T4-2/T4-3** 判据（"未派车车辆 ⇒ `STATE_MISSING` ⇒ 车队假拒绝"）**不可继承**——现 N/A；
4. 原 **T3-2/T3-6**（"生产路径接受自述 provider"）**不可继承**——现拒绝；
5. Q2-Fix 的 **`C1-POS`** 判据由"注入即可通过"改为"仅测试入口可达且不可认证"；
6. **任何**期望"复合在连续级未建立时通过"的旧判据**不可继承**；
7. 任何依赖收据 **存储** `THERMAL_HARD_OK` 键的旧判据**不可继承**（该键仍不存储）。

**旧测试文件保留原样**；旧判据的**原输入与原预期**仍在其中，其分歧由 `LEGACY-1/LEGACY-2` 记录。

### 兼容层披露（本轮新增）

| 符号 | 用途 | 是否影响认证 |
|---|---|---|
| `M1bQ2Certifier` / `M1bQ2FixCertifier` 别名 | 历史文件可**导入** | 指向**生产类**，不接受钩子 |
| `_Receipt['THERMAL_HARD_OK']` | 历史文件可**读** | 只读返回**采样级**；键**不存储** |
| `check_omega10_initialization` | 旧 per-lot 语义 | **不在**认证路径；带 `LEGACY_SHIM_WARNING` |
| `register_service_times` / `_SERVICE_TIME_CACHE` | 历史文件可**导入** | 表**不被任何判定读取** |
| `_legacy_attrs` / `_legacy_projection` | 旧 callable 参数翻译 | **翻译后仍受覆盖门约束**；快照身份为 `None` 时不做绑定校验（已在收据标注） |

---

## 7. 未解决 / 仍须单列

| 项 | 状态 |
|---|---|
| **连续时间温度证明** | **未建立**（`THERMAL_CONTINUOUS_NOT_ESTABLISHED`）；复合因此 **fail-closed by design** |
| **G5 注册表信任** | 未解决（无审核注册表） |
| **G6 收据持久化** | 未解决（收据仍为内存/文件 JSON，无写入失败处理） |
| **C2 生产安装身份** | 未解决（M1-b 收据未与 `run_repair_transaction` 共享状态身份） |
| **C3 M1-a/M1-b/C0 联合认证** | 未解决（M1-b 复合不含 M1-a 四项与真实 `certify_plan`） |
| **真实 C0 预算 `B`** | `AWAITING_USER_BUDGET_RULING`（合成 `B=12345.678` 仅验接口与硬预算拒绝语义） |
| **L2 七要素** | 未签署 |
| **旧类型 `PublicOrderAttributes`** | 保留为弱类型；认证拒绝之，但仍是可被误用的接口 |

---

## 8. 明确不主张

**未**实现完整生产安装；**未**裁定 `B`；**未**签 L2 七要素；**未**运行 M2。
**未**写入 `FULL_INSTALLABILITY_ESTABLISHED` / `L2_PASS` / `A3_PASS` / `stable_online_gain=True`。
**未**用合成钩子通过替代真实连续温度证明。

**停止，等待用户审查。**
