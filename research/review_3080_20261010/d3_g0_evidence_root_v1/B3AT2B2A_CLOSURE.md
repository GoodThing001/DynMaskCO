# B3-AT2B-2a — 收束裁决记录（独立只读复核，有限验收）

- 收束对象：`B3-AT2B-2a`（对象级写入权限封闭，隔离实现）
- 最终提交：`fde946af2d388e83794b5508b0910a9f3ea39a60`（`main` 与 `codex/o0cc-cal-phys` 同指向）
- 裁决建议：**CLOSED_AS_ENGINEERING_EVIDENCE**（工程证据项就此收束，**不再扩展新功能**）
- 复核方式：**独立进程 + 只读脚本**（重新执行 harness、重新做 `git archive` 字节复验、重算身份）
- **复核性质限制（必须与结论同时引用）**：这**不是**外部第三方评审。本会话内没有真正的独立评审者；下面所有"复核"都指"独立进程、只读、可重跑"，其可信度上限止于此。若你需要外部评审，本项应保持 `ACTIVE`。

## 1. 复核依据（本会话内实际产生并已提交的证据）

| 证据 | 位置 | 结果 |
|---|---|---|
| 预登记（迟到） | `B3AT2B2A_COUNTEREXAMPLE_SPEC.json` | `LATE_PREREGISTRATION_BEFORE_FIRST_EXECUTION`，含草稿冻结 SHA 与资格限制声明 |
| 八项原始判据 + 3 项迟到补充 | `B3AT2B2A_EXCLUSION_MATRIX.json` | 第十次执行全绿（14 行） |
| 十次运行原件 | `raw/B3AT2B2A_run1..run10_{stdout,stderr}.txt` | 前八次失败**逐字保留**，未因全绿删除 |
| AT2B-1 回归 | `raw/B3AT2B1_regression_after_2a_*.txt` | exit 0，13 行判定不变 |
| 字节冻结 | `B3AT2B2A_BYTE_FREEZE.json` | 30 条登记路径，提交后 `git archive HEAD` **0 失败**（路径键经两次修正，哈希未变） |
| 历史文件身份 | `strict_online_env.py` | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6`（**未修改**） |
| 既有批次回归 | 七个冻结记录 | B3AT2B1 15 / B3AT2H 8 / B3AT2G 27 / B3AT2A 29 / B3AT1 26 / B3AC 30 / B3CF 24，**全部 0 失败** |

## 2. 收束时确认的判定（均已到达预登记判词）

原始八项：`B2A-DEFAULTOFF=IDENTICAL_HISTORY_BEHAVIOUR`、`B2A-ADVANCE=LEGITIMATE_ADVANCE_NOT_BLOCKED`、
`B2A-REVEAL=REVEAL_ALLOWED_WITH_COMMITMENT_FROZEN`、`B2A-ATTR`/`B2A-NESTED=REFUSED_AND_STATE_UNCHANGED`、
`B2A-STALE=STALE_REFERENCE_INEFFECTIVE`、`B2A-COMMIT=REFUSED`、`B2A-POS=COMMIT_ALLOWED_INSIDE_SCOPE`。

迟到补充三项（**不得**追认成原始预登记项）：

| 行 | 判定 | 作用 |
|---|---|---|
| `B2A-CAND` | `REFUSED_SUFFIX_GROWTH_REQUIRES_DECISION_SCOPE` | 候选生成 ≠ 权威状态被修改（`actual_authoritative_write=false`） |
| `B2A-COMPATPOLICY` | `REFUSED_AND_STATE_UNCHANGED` | 修掉"规则存在但从未生效"的缺陷：policy 随线程局部 scope 传递，兼容安装在 `certified` 策略下**打开 scope 时即事前拒绝** |
| `B2A-DELETE` | `REFUSED_FOR_UNCOMMITTED_DELETION_AND_ALLOWED_FOR_THE_COMMITTED_HEAD` | execution scope 只允许消费等于 `committed_next` 的头部；中间删除与无承诺裸删被拒 |

## 3. 收束时必须继续保留的限制（不得随时间淡化）

1. **迟到预登记资格限制**：实现草稿先于 spec 存在，本批资格**不等同于**严格"先预声明后实现"的批次。
2. **`served_mask` 原地 numpy 写入未被拦截**（仅由 reveal 验证器 V4 事后覆盖）。
3. **正式 `certified` 安装能力未证明**：`SCOPE_DECISION_COMMIT` 的"带完整认证凭据写入"在本批**从未被行使**（D8 阻断），因此"安装资格"未建立。
4. **`B2A-DELETE` 的绑定残余**：绑定到车辆承诺身份，**未**绑定到服务完成事件对象。
5. **威胁模型边界**：不覆盖任意反射（`object.__setattr__`）、ctypes、恶意内存修改。
6. **兼容安装模式**是**标注过的非认证**机制；本批任何一次使用都**不计作** certified 安装。
7. **P2 未建立**：本批证明的是"隔离执行实例内的对象级排他"，**不是**"生产路径所有真实提交都经过门"。

## 4. 关于"独立复核"的诚实说明

复核脚本与 harness 在本会话内由同一实现者编写，因此：

- **重跑一致**能证明结果可复现、字节可复验、判定词未被事后修改（预登记在首运行前冻结，十次运行原件保留）；
- 但它**不能**发现"实现者共同的系统性误解"这一类缺陷。若你的结案标准要求这一点，`B3-AT2B-2a` 应保持 `ACTIVE / NOT_VERIFIED`，并交由外部评审。

## 5. 对研究主线的影响（按 2026-10-14 重定向裁定）

- 本项收束后，**D3 工程门不再优先占用主线资源**；`D3 G0` 仍未通过，`D8` 仍 `CAPABILITY_GAP_BLOCKING`，这两点与主模型归因实验的**资格矩阵**（P1）分开处理。
- **不启动 AT2B-2b**；不再新增安全功能。
- 下一批转向：**D3 同掩码 F vs D-Geo 归因实验的执行资格审查（P1）** → 唯一主模型科学杠杆的预声明（P2）。
- 任何后续工程任务都必须先回答："它在解除哪一个 **MaskCO 模型增益**的必要阻断项？"

## 6. 未变结论

P2 / G-L / G-C `NOT_ESTABLISHED`；D8 `CAPABILITY_GAP_BLOCKING`；D3 G0 未通过；G1/G2 未启动；
MaskCO 可归因在线增益未建立；`stable_online_gain=False`；I3 正式负结果（I3 对 `cond_hist`
配对差 −144.680，95% CI [−191.109, −100.125]）与旧 D2 `not_qualified` 保留不改写；
`algorithmic_gain_claim: none`（本批是提交安全工程，不构成算法创新）。
