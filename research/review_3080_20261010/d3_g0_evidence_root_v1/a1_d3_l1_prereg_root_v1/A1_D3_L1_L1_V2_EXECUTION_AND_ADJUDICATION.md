# A1-D3-L1 正式机制比较 · L1-v2 执行记录与独立裁决

- 执行轮次：**唯一一次**正式 F−D-Geo 机制比较（用户明确授权，见下）
- `run_id`：`d3l1-l1-two-arm-20261016-l1v2`
- 授权绑定 HEAD：`558fb5c68b4759dbbf9819f6802707d1d5e2bc8b`
- 执行预登记：`A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md`
  预登记 SHA-256：`e02b98587682a42f9142104e6798b6e8ef4db4135e9ee0dc0eaf2aa749e449ba`
- 签署包：`L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V3.json`
- 凭据：`l1_gates/L1_V2_EXECUTION_AUTHORIZATION.json`
  凭据摘要：`293b33ab6e684ac498c1bd3992d1cf7dcf3cefc571825d8833cc2088c051eb39`
- 运行目录：`l1_v2_execution_run/`；消费回执：`l1_gates/L1_V2_EXECUTION_consumption/`（目录**之外**，恰一条）
- **状态：执行完毕；本预登记的机会已消费；轮次判定 `BLOCKED`**（原因见 §4）
- 独立只读裁决：`l1_gates/L1_V2_EXECUTED_ROUND_ADJUDICATION.json` = **14/14**
  `L1_V2_EXECUTED_ROUND_INDEPENDENTLY_ADJUDICATED`

---

## 1. 合法入口与一次性消费

执行只经由正式入口，无旁路：

```
python scripts/simulation/d3_l1_l1_formal_run.py --supervise \
  --out-dir l1_v2_execution_run \
  --authorization l1_gates/L1_V2_EXECUTION_AUTHORIZATION.json \
  --preregistration A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md \
  --consumption-root l1_gates/L1_V2_EXECUTION_consumption
```

- 凭据在执行前经**逐字段**比对：21 条绑定全部一致，0 不匹配，实时漂移 0。
- 消费回执**恰一条**，位于运行目录**之外**；运行目录内 0 条（D11）。
- 之后**未重跑、未补跑、未筛选状态、未放宽预算、未拼接历史记录**；相位日志显示 64 次 `slot`、
  64 次 `slot_finished`、1 次 `completed`、序号连续，**无第二遍痕迹**（D14）。

## 2. 身份与见证（起止两端一致）

| 项 | 值 |
|---|---|
| 冻结 watch 路径 | **42**（38 源码 + 4 非源码资产） |
| 起始身份 | `ok=true`，42/42 |
| 监督器尾部身份 | `ok=true`，42/42，漂移 0 |
| worker 相位日志 `identity_end` | 42 条 |
| `source_identity_sha256` | `6139249946497ac1dde4848cda7531df85446667075fb50c012607c275d05d0c` |
| `state_binding_sha256` | `f5eedd1027342496939e7257e64f2d25dd6f7cd8787e731b6aa9dde213eefee1` |
| F ckpt / backbone | `f5acf83f…a7f5` / `03615b08…38db` |
| data npz / profile | `66573ddc…9c89` / `63e56212…57e8` |

## 3. 实际结果

### 3.1 分母（完整 64 状态，原发布顺序）

| 结果 | 数量 |
|---|---|
| 完成比较（两臂均走完掩码） | **63** |
| 异常（状态 (0,3)，首个评分调用抛错） | **1** |
| 超时 | **0** |
| 空池 | **0** |
| 未比较（NOT_COMPARABLE） | **0** |
| **分母合计** | **64** |

### 3.2 V1 首步分歧（仅对**真正可比较**的 63 个状态）

| V1 判定 | 数量 |
|---|---|
| **DIVERGENT** | **48** |
| IDENTICAL | 15 |
| NOT_COMPARABLE | 0 |
| NOT_APPLICABLE_EMPTY_MASK | 0 |
| NOT_ATTEMPTED | 0（全 64 口径下为 1，即该异常状态） |

**可比较状态中的分歧率 = 48/63 = 76.2%。**

分歧字段分布（48 个 DIVERGENT 状态中）：`successor` 48、`position` 44、`predecessor` 44、
`customer` 25、`vehicle_id` 22、`slot_anchor` 22。

两步掩码走完后**最终计划不同**的状态：**57/63**（6 个状态的 `plan_hash` 仍相同）。

### 3.3 V2 完整修复（未认证）

| 臂 | 走完整掩码且无失败原因 |
|---|---|
| F | **63/63** |
| D-Geo | **63/63** |

失败词表命中：**无**（无 `timeout` / `rollback` / `no_legal` / `nan` / `score_len` / `score_shape`）。

### 3.4 V3 可认证计划

**全部 64 状态均为 `NOT_RUN`**：未提供认证器、未做 C0 正式认证、**未安装任何计划**。
本轮的"可执行/可认证"措辞**不成立**。

### 3.5 顺序与共享预算归因

| 归因 | 数量 |
|---|---|
| NOT_ORDER_ATTRIBUTABLE | **64** |
| ORDER_ATTRIBUTABLE | **0** |
| ORDER_INDETERMINATE | **0** |

执行顺序 `F → D-Geo`，两臂共享每状态 10 s 单一原点
（`per_state_both_arms_share_one_10s_allowance`）。**本次无任何状态因顺序而超时**，
故不存在被顺序混杂的 V2 差异。

## 4. 异常分母与根因（如实记录）

| 项 | 内容 |
|---|---|
| 状态 | `day_id=0, position=3`，`state_hash=72b9d8e3a4103fa5382961756386c20707507ac061164283916cd1ff08379af3` |
| 阶段 | `state_comparison`，**F 臂第一次评分调用** |
| 异常 | `ValueError: index can't contain negative values`（`jax.lax.pad`） |
| 抛出处 | `mtrained_replanner.score_fn` 第 106 行 `cust = jnp.pad(cust, ((0,0),(0,pm)))` |
| 根因 | `d3_l1_real_arm.frozen_jit_shapes()` 把 `M_max` 取为 **`n_legal_expected` 的最大值 = 26**，而该张量实际按 **掩码池大小** 补齐：本状态掩码池 = **27** ⇒ `pm = 26 − 27 = −1` |
| 实测 | 64 状态掩码池 `M`：min 6，**max 27**；超过 26 的状态**恰 1 个**（即 (0,3)）；`Nv` max 30 = `Nv_max`，无越界 |
| 定位 | 这是**冻结 F 路径的代码缺陷**（`M_max` 取值来源错误），**不是该状态的属性**；64 状态里只有它踩到 27 |
| 处置 | 按预登记与授权**终止并上报**，**未修补、未重跑**。修补会改动冻结 watch 路径 ⇒ 需要**新身份 + 新授权**，超出本次批准 |

`ledger_ok=false` 对该状态成立且仅因"状态在账本可交叉核验之前就抛错"；其余 63 状态账本**全部通过**。

## 5. 资源统计

| 量 | 实测 | 冻结上限 | 余量 |
|---|---|---|---|
| 初始化 + 编译 | 7.063 s | 60 s | 8.5× |
| 每状态墙钟（两臂共享） | min 0.250 / median 0.313 / max **0.625** s | 10 s | 16.0× |
| 整批（worker 受控墙钟） | 38.718 s | 180 s | 4.65× |
| F 评分调用合计 | **126** | 128 | 2 |
| D-Geo 评分调用合计 | **126** | 128 | 2 |
| 每臂每状态调用 | **恰 2**（63 状态全部） | 2 | 0 |
| 账本问题 | **0** | 0 | — |

顺序执行与共享预算造成的额外时间开销：**0 次超时**，故本次无此类开销。

## 6. 可形成与不可形成的结论

**可以形成**
1. 在完全相同的初始有序 legal 上，F 与 D-Geo 在 **48/63（76.2%）** 可比较状态上**首步动作分歧**；
   分歧恒含 `successor`，多数伴随 `position`/`predecessor`，约半数含 `customer`/`vehicle_id`。
2. 两步掩码后两臂**最终计划不同**的状态为 **57/63**（未认证）。
3. 两臂在 63 个完成状态上**均走完整掩码且无失败原因**（V2，未认证）。
4. 失败/超时分布：**超时 0、空池 0、未比较 0、异常 1**；全部状态保留在 64 分母中。

**不可以形成**
1. 任何修复是**可认证**的（V3 = `NOT_RUN`）；
2. 任何**决策质量**、**在线效用**、**显著性**或**学习增益**主张；
3. 任何关于**历史训练期 backbone** 的陈述（`HISTORICAL_BACKBONE_UNVERIFIED` 不变）；
4. 把 63/64 的可回答分母当作"完整 64 状态结论"——该分母是否可接受**属签署人判断**，
   本记录与审计工具**均不代作此判断**；
5. 反向亦不成立：**不得**据此宣称"F 与 D-Geo 等价"，因为 48/63 明确分歧。

## 7. 下一步机制判断

1. **机制问题已获得实质答案**：在冻结的 64 状态上，两评分器**确实产生可归因的首步动作分歧**
   （48/63），且分歧**未被顺序/共享预算混杂**（ORDER_ATTRIBUTABLE = 0）。
   这是 L1 的回答，**不是**增益、质量或效用结论。
2. **本次仍有 1/64 状态未被回答**，原因是**冻结 F 路径的 `M_max` 取值缺陷**（应为 27，取成 26）。
   该缺陷可由一行改动修复，但会改动冻结 watch 路径 ⇒ **必须新身份 + 新授权**；
   **本次机会已消费，不得重跑**。
3. 因此存在两个**互斥**的后续选项，**均需用户明确决定**：
   - **(A) 接受 63/64 作为 L1 的最终回答**（在报告中明确标注 (0,3) 未被回答及其原因），
     不再消耗任何执行机会；
   - **(B) 授权一次新的、明确的新身份执行**（修复 `M_max`、重建身份与预登记、重新签署，
     并说明这是一次**新机会**而非本次重跑）。
4. **不建议**在未获 (A) 或 (B) 决定前推进 L2/L3、E2 或任何安装路径。

## 8. Claim limits

1. 本记录描述**一次**工程机制诊断的执行；`stable_online_gain = False` 不变。
2. 64 状态**曾用于开发**，L1 结论**不是**独立科学验证；L2 须另用新鲜开发日。
3. 离线机制诊断**不满足** L3 在线发布门（8+2 s）。
4. C0 正式预算仍为 **`NON_BLOCKING_UNRESOLVED`**（V3 = `NOT_RUN`，无需裁定）。
5. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、
   D3 G0 未通过、D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果 **全部不变**。
6. **未运行**：E2、L2、L3、40 日批次、最终测试、任何计划安装或其他正式实验。
