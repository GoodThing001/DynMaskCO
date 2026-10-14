# P0-I3 历史资格原件保全记录

**日期**：2026-10-14 · **性质**：仅证据保全，**不进行任何科学实验**
**授权**：用户 2026-10-14「P0-I3 历史资格原件保全任务」（8 步 + 停止条件 + 结论约束）
**基线**：`daaa4980b8a3d71d49b2f35776d3be75101c987c`（执行时 `origin/main` = 本地 HEAD = 该提交；
分支 `codex/o0cc-cal-phys`；无 `index.lock`；无其他 git 进程；工作树除既有无 `.gitmodules` 映射的
幽灵子模块 `CC_Compare/MAPT_code` 外**干净** ⇒ **无工作树冲突**）

---

## 1. 步骤 1–2：基线、工作树与三份原件（只读核验）

| # | 原件路径 | 字节 | mtime | SHA-256 | 服务器副本 |
|---|---|---|---|---|---|
| 1 | `results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json` | 1338 | 2026-10-09 00:44:21 | `10e60da10a80d74ae59cc10333f012f1f09e9541497db2d69c84f48ed776564a` | **缺失**（全盘按名搜索 depth≤9，0 命中） |
| 2 | `results/i3_registered_1008_v5/INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json` | 24764 | 2026-10-09 00:43:33 | `0825ad07afbf23926bad3b4a14b50f2a4930d988092afbbe86b8c9cbb3bfd7c6` | **缺失** |
| 3 | `results/i3_online_recovery_1009_v1/INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json` | 153550 | 2026-10-09 15:33:02 | `1f860e7594948844305160e2120fd04d627bcc00416974cc34ab56dea494f1ef` | **缺失** |

三份**全部存在**，字节长度与 SHA 与既有溯源登记一致 ⇒ **不触发停止条件**。

## 2. 步骤 6：隐私与敏感信息检查（**复制与推送之前**执行）

扫描器 `scan_privacy.py`（自哈希记入清单），结论 **`clear_to_preserve`**、**0 项阻断**：

| 检查项 | 结果 |
|---|---|
| 私钥块 / ssh 公钥 / password 字段 / secret·token·api_key 字段 / authorization·bearer / 连接串 / 令牌样式 | **全部 0 命中** |
| **配置的 SFTP 密码是否出现在文件中** | **否**（仅输出布尔值，不回显凭据） |
| **配置的 SFTP 主机是否出现在文件中** | **否** |
| 配置的登录用户名 | 出现在 2 份文件中（report-only） |

**唯一的 report-only 发现**是 `/home/hzeng/...` 绝对路径中的登录用户名（1 + 348 处）。
判定为**不阻断**的理由：本**公开**仓库已在 `CLAUDE.md` 与大量登记工件中发布同一路径与用户名，
本次保全**未新增任何暴露面**；且不含任何凭据材料。**未发现敏感信息 ⇒ 允许推送原件。**

⚠️ 扫描器首轮**拒绝**了本次保全 —— 原因是我自己的路径解析 bug（`parents[4]` 应为 `parents[3]`），
它**没有静默扫空就放行**。这正好验证了 fail-closed 设计有效；修正后复跑通过。

## 3. 步骤 3–4：复制与独立清单

* **方法**：`shutil.copyfile` **二进制复制**；原件**未移动、未编辑、未删除**（验证见 §4 的 `orig` 列）。
* **归档目录**：`.../p0_i3_preservation_root_v1/archived_originals/`
* **清单**：`ARCHIVE_MANIFEST.json`，逐项登记
  原始路径 / 归档路径 / `sha256_before_preservation` / `sha256_after_copy` / 字节 / mtime /
  服务器副本状态 / 来源角色，另含保全时间（UTC）、基线提交、环境、方法与隐私扫描结果及脚本 SHA。
* **避免自我认证**：清单**不**记录"承载它自己的那个提交"（那会构成循环）。
  清单内 `archive_commit_identity` 字段明确指向**事后**写出的 `ARCHIVE_COMMIT_IDENTITY.json`；
  而该身份文件自身的提交可由 `git log -- <path>` 解析，**不声称**它自证。
* **行尾冻结**：本根与 `archived_originals/` 各放 `* -text`，使 `blob == 工作树字节`，
  否则 `core.autocrlf=true` 会把 LF 变 CRLF，令登记的 SHA 在 checkout 后失效。

## 4. 步骤 5：独立验证（`verify_archive.py`，与保全脚本相互独立）

| 归档文件 | sha256[前16] | 等于保全前 | 等于清单 | 原件未变 | **blob==工作树** | 判定 |
|---|---|---|---|---|---|---|
| `LOCAL_128_STAGE_ACCEPTANCE_20261009.json` | `10e60da10a80d74a` | ✔ | ✔ | ✔ | ✔ | **OK** |
| `INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json` | `0825ad07afbf2392` | ✔ | ✔ | ✔ | ✔ | **OK** |
| `INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json` | `1f860e7594948844` | ✔ | ✔ | ✔ | ✔ | **OK** |

* **与既有溯源登记逐项比较**：`P0_I3_QUALIFICATION_PROVENANCE.json` 登记的受理件 SHA
  `10e60da1…` 与归档件重算值 **一致**。
* `problems: []`，`VERDICT: verified`（原件见 `ARCHIVE_VERIFICATION.json`）。
* **`blob==工作树`** 这一列是关键：它证明 git 实际存储的字节与工作树逐字节相同 ⇒ 登记身份可跨 checkout 复现。

## 5. 步骤 7–8：提交范围与 diff

仅提交**新增保全文件**与必要的保全说明：

```
 .gitattributes                                       |   3 +
 ARCHIVE_MANIFEST.json                                |  73 +
 P0_I3_PRIVACY_SCAN.json                              |  76 +
 archived_originals/.gitattributes                    |   1 +
 archived_originals/INDEPENDENT_128_...json           | 813 +
 archived_originals/INDEPENDENT_ACTUAL_...json        | 2349 +
 archived_originals/LOCAL_128_STAGE_ACCEPTANCE...json |  33 +
 preserve_originals.py                                | 183 +
 scan_privacy.py                                      | 127 +
 verify_archive.py                                    | 128 +
```

**未修改**任何冻结树、历史 gate、checkpoint、原始结果或既有裁决；**未**编辑 I3 文档的证据入口链接。

## 6. 停止条件执行情况

| 条件 | 是否触发 |
|---|---|
| 任一文件不存在 | 否（三份全在） |
| SHA 不一致 | 否（保全前/保全后/清单/溯源登记四向一致） |
| 原件内容发生变化 | 否（§4 `orig` 列全 True） |
| 发现敏感信息 | 否（0 阻断项） |
| 工作树冲突 | 否（仅幽灵子模块） |

**均未触发 ⇒ 归档与推送继续。**

## 7. 结论约束（原样遵守）

**保全成功仅表示新增了一份经过哈希核验的历史原件副本**，
**不表示 I3 采集资格重新成立**，**不改变任何科学裁决**。
`stable_online_gain` 仍为 **`False`**。

## 8. 附带发现（**未纳入本次授权范围**，仅供你决定）

同目录/同族还有**同类单副本风险**文件，本次**未**归档（授权仅三份）：

| 文件 | 字节 | SHA（前 16） |
|---|---|---|
| `i3_online_recovery_1009_v1/INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json` | 153285 | `9a951ce1c96e2de5` |
| `i3_online_recovery_1009_v1/INDEPENDENT_FIXED_MODEL_ACCEPTANCE.json` | 1357 | `99ab6725b80d3b5f` |

若要一并保全，请再给一次授权（同一流程：只读核验 → 隐私扫描 → 复制 → 独立验证）。
