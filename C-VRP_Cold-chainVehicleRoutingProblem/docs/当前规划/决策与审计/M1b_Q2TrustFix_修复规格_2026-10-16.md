# M1-b / Q2-TrustFix 修复规格（实现前冻结）（2026-10-16）

**基准**：`84e378968de2e63bd2da302d70fbf9c45c3dbfec`（`git fetch` 后本地 `HEAD`、
`origin/main`、`origin/codex/o0cc-cal-phys` 三者相同；工作区仅 `CC_Compare/MAPT_code` 既有脏状态）。

**性质**：依据 **M1-b/Q2-Fix-T 已冻结的 17 例负控**集中修复已证实的信任边界缺陷。
**不是**新科学杠杆；**未**重新开放 L1/L2 正式实验次数；**未**裁定预算 `B`；**未**生产接线。
本文件**先于任何源码修改**写入。

**禁止更改**：旧报告、旧结果、冻结数据、M1-a、L1 裁决、`MASKCO_code/`。

---

## 0. 修复前身份（逐文件 SHA256，留档）

| 文件 | 修复前 sha256 |
|---|---|
| `scripts/m1b_adapter/m1b_contract_identity.py` | `0f19fcfb1d80ceb1dc5c64962c85a948137befbdc21a1d41e8535aca738d25a4` |
| `scripts/m1b_adapter/m1b_immutable_snapshot.py` | `38acab48996211191c60ca88edc9190c327815381fe2b287ec2d7f8b06432c21` |
| `scripts/m1b_adapter/m1b_return_feasibility.py` | `44138e0c719e638fa216ddd7ed0c8f6d20f4791bf412dcac2db3bc6601b43a9f` |
| `scripts/m1b_adapter/m1b_quality_certificate.py` | `2aa801d09c26e6851e75ca4c1672ff9f84f1d59d3e5b95d402fcfc8f96ba4d22` |
| `scripts/m1b_adapter/m1b_thermal_gate.py` | `5ab1841f41aa8d905c91477e5acb29d891b0c27db93585c3672e252c0b45ba3b` |
| `scripts/m1b_adapter/m1b_q2_certifier.py` | `702bae2a6a0c5f9e964caa6b941be92a841e0f811547d789fcdc1b84fb8fb2a7` |
| `scripts/m1b_adapter/m1b_timing.py` | `ba77a44a1b1c0ed5abbae3b91e0d39a1b7b055256243489fa92c4553ba4bd83a` |
| `scripts/m1b_adapter/m1b_quality_certificate_v1_q2r.py` | `c9cae4db381a1842e321c58e4f549edb6aacd084993a9399ee8b221761523159` |

**本轮新身份**：实现版本号 **`M1B_Q2TRUSTFIX`**，收据 schema
`m1b_q2trustfix_engineering_receipt_v1`。收据须携带 `implementation_version`、
`implementation_sha256` 与 `pre_fix_source_sha256`。

---

## 1. 修复假设（可证伪）

> **H-TRUSTFIX**：Q2-Fix-T 记录的 2 处假通过（T1-1、T2-1）与 2 处假拒绝（T4-2、T4-3）
> 及 2 处来源身份缺口（T3-2、T3-6）均可由**可定位的实现缺陷**修复；
> 修复后 T1-1/T2-1/T4-2/T4-3 **按新规格翻转**、T3-2/T3-6 **不能再由自述绕过**，
> 且**不引入新的假通过**、**不破坏**既有必要不变量（含"有货无状态必须拒绝"）。

**证伪条件**：任一反例未翻转，或出现新假通过，或"有货无状态必须拒绝"被破坏，
或 M1-a / 原 Q2-R 16 例出现退化。

---

## 2. 修复后预期（事前固定）

| 例 | 修复前 | **修复后预期** | 判定 |
|---|---|---|---|
| **T1-1** | `FALSE_PASS`（任意 callable ⇒ `CONTINUOUS_OK` + `composite_pass=True`） | **连续级一律 `NOT_ESTABLISHED`**；不得由任意 callable 得到 `THERMAL_CONTINUOUS_OK` | 必须翻转 |
| **T2-1** | `FALSE_PASS`（Ω11 直接调用静默乐观） | **`OMEGA11_TIME_INPUT_UNAVAILABLE`**（缺少逐实例 `times` 即拒绝） | 必须翻转 |
| **T4-2** | `FALSE_REJECT` | 未派车车辆判 **`THERMAL_NOT_APPLICABLE`**，不拖累车队 | 必须翻转 |
| **T4-3** | `FALSE_REJECT` | 同上；**单独一辆从未派车空计划车不再阻断复合** | 必须翻转 |
| **T3-2** | `NOT_ESTABLISHED`（自述取得信任资格） | **外部自述的 `served_nodes`/`revealed_nodes`/来源字符串不得自行取得信任资格** ⇒ 拒绝 | 必须解除 |
| **T3-6** | `NOT_ESTABLISHED`（与 env 矛盾的声明仍可读） | **由固定构造边界从环境当前事件派生可见集合** ⇒ 矛盾声明被拒 | 必须解除 |
| T1-3 / T1-4 | `OK` | 保持：采样级独立，采样违规时复合必拒 | 保持 |
| T2-2 / T2-3 / T2-4 | `OK` | 保持 | 保持 |
| T3-1 / T3-3 / T3-4 / T3-5 | `OK`/`NOT_ESTABLISHED` | T3-3/T3-4 转为**不可变且不含全节点真值** | 必须解除 |
| T4-1 | `OK` | 保持 | 保持 |

---

## 3. F-T1：连续证明钩子必须 fail-closed

1. **正式认证路径不得**再通过任意 `continuous_extremum_proof` callable 得到
   `THERMAL_CONTINUOUS_OK`。
2. 未有**真实解析证明 + 独立审核**前，**连续等级一律 `THERMAL_CONTINUOUS_NOT_ESTABLISHED`**。
3. 直接注入 `(True, {})` 必须**拒绝或保持连续级未建立**。
4. **测试替身只允许在隔离测试入口使用**，且该入口产出的收据必须标
   `TEST_ONLY_NOT_A_PRODUCTION_CERTIFICATE`，**不得**产生可供生产消费的可信证书。
5. **不得**仅凭 SHA256、字符串登记、用户自报来源或注册表命中就宣布数学证明成立。
6. `THERMAL_SAMPLED_OK` 与连续级**继续独立**。

**实现约束**：正式 `M1bQ2TrustFixCertifier` **不接受** `continuous_extremum_proof` 参数；
隔离测试入口为**独立类**，其收据带 `test_only=True` 且 `composite_pass` **不得**被读作生产证书。

---

## 4. F-T2：关闭 Ω11 直接调用绕过

1. `certify_return_feasibility` 中**缺少逐实例 `times` 必须直接拒绝**。
2. `register_service_times` **可保留导入兼容**，但其结果**不得进入任何正式认证判定**。
3. **不允许**自动构造 `tw_start = 0`。
4. 缺少服务时间、时间窗或实例身份时**拒绝**。
5. **真实服务时间为 0 的输入必须允许**（键存在、值为 0）。
6. 核对 `m1b_thermal_gate` 的独立 legacy callable 回退：**同类假通过一并隔离**。
7. **不得**因调用者绕过顶层 orchestrator 而恢复乐观时间估计。

---

## 5. F-T3：可信 strict-online 数据投影

**固定构造边界**（唯一可用内部 reveal 状态之处）：

1. 从**运行环境当前事件的已揭示订单集合**构造输入；
2. 只有该构造层可使用内部 reveal 状态确定可见集合，**不得**把未揭示订单的 `reveal_time`
   或真实属性转交算法；
3. 对**当前已揭示**订单复制属性，**仅包含实际需要的字段**；
4. **不得**把全节点真实数据表保留在 provider 内部；
5. 投影**一经构造不可变**；
6. `source_identity` 绑定**实例、事件、可见集合、数据合同及快照身份**；
7. **外部自述**的 `served_nodes`/`revealed_nodes`/来源字符串**不得自行取得信任资格**；
8. 后续揭示**必须生成新快照**，不得修改旧快照；
9. 认证前后进行**身份一致性检查**。

**必须提供**：① 真实已揭示节点允许访问；② 未揭示节点 getter 不可访问；
③ 未揭示节点真实值**不在内部数据中**；④ 伪造来源或可见集合**被拒绝**；
⑤ 构造后修改输入**失败**；⑥ 两次事件的可见集合/身份/属性**不能混用**。

**边界说明**：此处"可信"指 **strict-online 软件协议下的可信边界**，
**不**声称 Python 进程对任意恶意代码具有密码学隔离。

---

## 6. F-T4：完整车队状态与 `NOT_APPLICABLE`

**逐车适用性判定**——下列条件**全部**满足 ⇒ **`THERMAL_NOT_APPLICABLE`**
（而**不是** `THERMAL_REPLAY_STATE_MISSING`）：

- 从未 dispatch；· `idle`；· 当前在 depot；· **没有货物**；· **没有未完成承诺**；
- **没有候选执行后缀**；· **没有必须继续满足的热状态**。

**反之必须拒绝或按真实约束处理**：

| 情形 | 处置 |
|---|---|
| 未派车但**候选要出车** | **不得** N/A |
| **有货但无冷链状态** | **必须拒绝**（原始负控保持） |
| 有未完成承诺但缺状态 | **必须拒绝** |
| `returning` / `committed` | 遵守**真实在途及后缀约束** |
| `closed` | 用**可信历史审计**；既无货也无待执行任务时**不得**重复做不适用的热重放 |

**复合规则**：适用车辆必须通过全部必要热门；**N/A 车辆不得自动当作温度证明成功，
也不得无故拖累整个车队**。

---

## 7. 回归要求（机械核验）

1. **原 Q2-R 16 例**：**无已知假通过**；
2. **Q2-Fix 21 项**：逐项披露**语义兼容**情况；
3. **Q2-Fix-T 17 例**：T1-1 / T2-1 / T4-2 / T4-3 **按新规格翻转**；T3-2 / T3-6 不可由自述绕过；
4. **新增**多车 `idle`/`ready`/`committed`/`returning`/`closed` 组合正负控；
5. **新增**未揭示真值不可读取、不可变、跨事件身份负控；
6. **M1-a S-1…S-17 无退化**；
7. **独立只读验收**与**逐文件哈希**。

**旧判据不可继承事项**须单列；**不得**声称原检查全部通过；**不得**把旧测试返回码非零
读作原检查仍通过或读作修复破坏旧判据。

**验收次数**：每测试各执行一次；失败保留不删；**不**追加运行以改变结论。

---

## 8. 必须报告的独立结论

- `TRUST_FALSE_PASS_RESOLVED` **或失败**；
- `FLEET_FALSE_REJECT_RESOLVED` **或失败**；
- `STRICT_ONLINE_BOUNDARY_VERIFIED` **或未建立**；
- **`THERMAL_CONTINUOUS_NOT_ESTABLISHED`**（除非确有新的、独立审核的数学证明）；
- **`FULL_INSTALLABILITY_NOT_AVAILABLE`**。

**尤其不得用合成钩子通过替代真实连续温度证明。**

---

## 9. 明确不主张

**不**实现完整生产安装；**不**裁定 `B`；**不**签 L2 七要素；**不**运行 M2；
**不**写入 `FULL_INSTALLABILITY_ESTABLISHED` / `L2_PASS` / `A3_PASS` / `stable_online_gain=True`。
G5（注册表信任）、G6（收据持久化）、C2（生产安装身份）、C3（M1-a/M1-b/C0 联合认证）仍单列。
