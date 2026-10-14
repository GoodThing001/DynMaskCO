# 第 2 项：I3 采集资格文件溯源（只读）

**日期**：2026-10-14 · **基准**：`f10cb4772` · **性质**：只读溯源；不训练、不采集、不改任何冻结树或历史裁决
**唯一问题**（用户裁定）：**能否在其他可信历史归档中找回与 I3 原登记身份一致的资格文件？**
找到就记录原件 SHA 与来源；找不到就保留"当前不可就地复验"；**不得补造**。

**追溯对象**：`results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json`
（I3 正式裁决文档"证据入口"第 83 行引用）

---

## 0. 结论（一句话）

**该资格记录只在本地工作树存在单副本**（未跟踪、被 `.gitignore` 忽略、且不在证据包中），
**服务器与各离线归档中均无副本**；
但它的两个内嵌身份哈希（`audit_sha256`、`merged_teacher_sha256`）**被服务器侧 6 个逐字节一致的工件交叉携带**。
⇒ 判定 **`not_recoverable_from_second_archive_but_identity_corroborated`**：
**身份可交叉佐证，记录本身不可找回**；**交叉佐证不得替代原件，也不得补造**。

## 1. 找到的副本（本地唯一）

```
路径   C-VRP_Cold-chainVehicleRoutingProblem/results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json
大小   1338 B
mtime  2026-10-09 00:44:21
SHA256 10e60da10a80d74ae59cc10333f012f1f09e9541497db2d69c84f48ed776564a
```

| 检查 | 结果 |
|---|---|
| git 跟踪 | **未跟踪** |
| 被忽略 | **是** —— `.gitignore:28  /C-VRP_Cold-chainVehicleRoutingProblem/results/*` |
| 是否在 Release 证据包 `DynMaskCO_raw_evidence.tar.gz` | **否**（474 个成员中 `STAGE_ACCEPTANCE` 命中 **0**） |
| 版本分布 | `v1`–`v4` **均无**该文件；仅 `v5` 有 |
| **单副本风险** | **高**：唯一副本是未跟踪、被忽略的工作树文件 |

## 2. 服务器侧（有界只读清查）

| 检查 | 结果 |
|---|---|
| `results/i3_registered_1008_v5/` 目录 | **存在**（41 个顶层文件） |
| 该资格文件 | **缺失** |
| 全盘按文件名搜索（`-maxdepth 9`） | **0 命中** |
| 全盘 `*STAGE_ACCEPTANCE*` | **0 命中** |
| `B_REPRO_CLEAN/results`、`B_REPRO_CLEAN/repro_work` | 无 `*128*ACCEPT*` 匹配 |
| `correction_validation_1009_v2|v3/recovered` | **不存在** |

⇒ 服务器上**不是路径写错**，而是该文件**确实没有**。

**未覆盖范围（不得外推为"全服务器不存在"）**：登录用户无读权限的路径、主机上未挂载的归档、外部磁带/备份。

## 3. 记录内容与 I3 文档逐项一致

| 记录字段 | 值 | 与 I3 文档 |
|---|---|---|
| `states` / `unique_states` / `days` | 128 / 128 / 32 | 一致（第 11 行"32日128唯一状态"） |
| `action_feasible_branches` / `infeasible_action_branches` / `hard_failed_action_feasible_branches` | 230 / 26 / 0 | 一致（"230可执行分支hard"、"26动作不可行"） |
| `all_label_counts` | valid 102、negative 55、positive 47、invalid 26、small_or_tie 16 | 一致（"102有效标签（47正55负）…16有效小差值"） |
| `training_states` / `fixed_last_192_steps` | 96 / true | 一致（第 12 行"同96状态两遍"、"各192唯一连续步"） |
| **`formal_training_pair_independently_accepted`** | **false** | **正是 I3 文档第 15 行所指的那个字段** |
| `online_gain_established` | false | 与既有结论一致 |
| `scope` | 声明"linked to separately executed server terminal-trace audit; does not claim a new local raw-archive replay" | 与文档第 15 行口径一致 |

⇒ 可以确认：**这份本地文件就是 I3 文档所引用的那一份**（内容逐项吻合），
不是同名异内容的其他文件。

## 4. 内嵌身份的两个哈希及其佐证

`audit_sha256 = ac747189c1a2f02e5268ee934c8ad0cecd1780a3c76ec9a6a51dac80a3c22a72`
`merged_teacher_sha256 = 9028f75770d2ac8d7b115101cf58af297839e8ef1f6cdf48d8c717b8e5081495`

**携带这些身份、且在本地与服务器上逐字节一致的工件（6 个）**：

| 工件 | 大小 | SHA（前 16） | 携带 | 本地 vs 服务器 |
|---|---|---|---|---|
| `v5/TEACHER_AUDIT_ACCEPTED.json` | 305 | `5db9c4c84bd70fcf` | `audit_sha256` | **MATCH** |
| `v5/SOURCE_MANIFEST.sha256` | 82,634 | `b75bb9c7abab38b0` | `audit_sha256` | **MATCH** |
| `v5/teacher_merged/report.json` | 17,823 | `bd89557140ffaa03` | `merged_teacher_sha256` | **MATCH** |
| `v5/maskco/train_report.json` | 238,999 | `c096cc2fd8c74882` | `merged_teacher_sha256` | **MATCH** |
| `v5/reconstruction_control/train_report.json` | 215,682 | `075942cfe9ca45e4` | `merged_teacher_sha256` | **MATCH** |
| `i3_online_recovery_1009_v1/SOURCE_MANIFEST.sha256` | 86,414 | `79d8de47af327ae4` | `audit_sha256` | **MATCH** |

**仅本地存在（服务器缺失）的佐证件 2 个**：
`v5/INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json`（24,764 B，`0825ad07afbf2392`）、
`i3_online_recovery_1009_v1/INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json`。

⚠️ 需要说清的分寸：这两个"独立"身份检查件**都只在本地**；
真正**服务器侧也存在**的审计身份载体是 `TEACHER_AUDIT_ACCEPTED.json` 与两份 `SOURCE_MANIFEST.sha256`。
所以"审计身份可由服务器侧佐证"成立，但"独立检查件本身已被服务器保存"**不成立**。

## 5. 遵守的禁止项

* **未**重建、推断或补写任何历史字段；
* **未**用佐证工件替代缺失的记录；
* **未**修改任何冻结树 / gate / checkpoint / 封存结果；
* **未**编辑 I3 文档的证据入口链接（保留原文，实际状态记在本文件里）。

## 6. 建议（需你决定，本轮未执行）

1. **可耐久保存**：该记录只剩单副本，建议把这份 **1338 B** 原件（或其 SHA）**显式保存进受跟踪的证据位置**
   —— 这是一次**有意的保存动作**，**不是重建**；但会改变"该文件只存在于被忽略目录"的现状，故请你确认后我再做。
2. **反向风险**：`INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json` 与
   `INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json` 同样**只在本地**、且同样被忽略
   ⇒ 属于同一类单副本风险，建议一并纳入第 1 条的保存决定。
3. **不要在 I3 文档里改链接**：链接对本地树是有效的；把"服务器不可就地复验"记在本报告即可。

## 7. 不主张

不主张 I3 的采集资格成立或失效 —— 本项只回答**文件溯源**；
记录内容与文档一致且身份被服务器侧工件交叉携带，但这**不等于**该资格已在服务器上就地复验。
**稳定核心在线增益仍为 `False`。**
