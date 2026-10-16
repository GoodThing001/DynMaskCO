# 公开仓库密钥暴露处置报告 · L1-v2 归档

- 处置日期：2026-10-16
- 触发：用户要求检查公开 Git 仓库中被提交的 guard/执行密钥、执行授权文件与执行上下文是否含不应公开的凭据
- 仓库：`https://github.com/GoodThing001/DynMaskCO`（**公开**）
- 结论：**发现 2 个已公开的 256-bit 执行密钥**（其中 1 个为本轮引入，1 个为**先前已存在**）；
  **未发现**其他凭据（`sftp.json` **未被跟踪**，见 §5）
- 处置：**按暴露处理、永不复用、私有留档原件、公开副本就地脱敏**，并给出历史清理 runbook（§4）
- **本报告不输出任何密钥明文**；只记录**单向摘要**

---

## 1. 暴露清单（仅单向摘要）

| # | 仓库内路径 | 归属 run_id | 字节 | 文件 sha256 | 密钥 sha256 | 首次公开于 |
|---|---|---|---|---|---|---|
| 1 | `l1_v2_execution_run/guard/execution_secret_12adf7457768bfb5.txt` | `d3l1-l1-two-arm-20261016-l1v2` | 64 | `f786b3c202870670226f892f5ace31aa4ff04c52087e47b3be29cc8dc443a389` | 同左 | commit `bdbc18ad7`（本轮 L1-v2 执行提交） |
| 2 | `v122_evidence/run_v122/guard/execution_secret_b3e36be45ceb6bb4.txt` | `d3l1-real-f-forward-20261015-v122` | 64 | `b9600f71696e70d27e75e68778cb65e21e000346f9813e1ad246076182215972` | 同左 | **早于本轮**（V1.22 轮次即已公开，本轮复查时发现） |

两者均为 **64 位小写十六进制 = 256 bit**，由 `d3_l1_v120_execution_context.publish_execution_context`
的 `secrets.token_hex(32)` 每次运行**新生成**，写入 `guard/execution_secret_<8字节随机>.txt`。

**扫描范围与结果**：对**全部 38,072 个受跟踪文件**逐个做字节级包含检查，命中**仅上述两个文件自身**。
即密钥值**未被**复制到任何日志、报告、JSON 记录或代码中。

## 2. 影响评估：为什么这**没有**残余授权能力

| 事实 | 依据 |
|---|---|
| 每个 run **独立生成**密钥，路径带随机后缀 | `token_hex(32)` + `token_hex(8)`，运行期临时生成 |
| 密钥只用于签名**本 run** 的执行上下文 | `compute_hmac(secret, canonical_payload(context))` |
| 守卫同时要求**运行目录绑定**与**消费回执存在** | `REFUSAL_REASONS`：`run_id_mismatch` / `run_dir_mismatch` / `consumption_receipt_missing` / `hmac_mismatch` |
| 两个 run 的凭据均**已一次性消费** | V1.22 与 L1-v2 各有恰一条回执，位于运行目录之外 |
| 消费回执**无法由被闸门保护的进程伪造** | 回执由监督器 `authorize()` 以 `O_EXCL` 写出 |

因此暴露的密钥**不能**授权任何新工作。其唯一实际降低的性质是：**该次已完成的 run** 的执行上下文
**可被伪造**（例如有人重放该 run 的 `--worker` 以生成伪证据）。由于该 run 及其裁决已冻结归档、
且此后不会再有复用同一密钥的合法执行，该风险为**取证性质**而非授权性质。

**处置口径：按暴露处理，永不复用。** 不需要轮换既有生产密钥（无此类密钥）；后续任何 run **本来就会**
自行生成新密钥，本处置只是确保**不再公开**。

## 3. 已执行的处置

1. **私有留档原件**（字节级不变，位于仓库**之外**）：
   `~/dynmaskco_private/exposed_execution_secrets/2026-10-16/`
   - `d3l1-l1-two-arm-20261016-l1v2__execution_secret_12adf7457768bfb5.txt`（sha256 `f786b3c2…`）
   - `d3l1-real-f-forward-20261015-v122__execution_secret_b3e36be45ceb6bb4.txt`（sha256 `b9600f71…`）
   - 同目录 `README.txt`：说明"已暴露、禁止发布、原件留档、引用的凭据均已一次性消费"
   **核对**：留档文件字节数与摘要与 §1 完全一致。
2. **公开副本就地脱敏**：两个文件改为 `REDACTED-EXPOSED-EXECUTION-SECRET` 存根，写明
   run_id、文件 sha256、**密钥 sha256（单向）**、字节数、首次公开 commit、消费时间与私有留档路径。
   存根不含密钥值。
3. **`EXECUTION_CONTEXT_*.json` 一律保持原样**：HMAC **不可逆推密钥**，且密钥既已公开，
   删除 HMAC 只会削弱"守卫确实绑定了 run / 目录 / 已消费凭据 / nonce"这一**承重证据**。
   `secret_path` 字段同样保留（它指向的路径仍然存在，只是内容被脱敏）。
4. **脱敏账本**（受跟踪、可审计）：`l1_gates/L1_EXPOSED_SECRET_REDACTION.json`。
5. **工具**：`scripts/tests/d3_l1_exposed_secret_redaction.py`（默认 `--dry-run` 语义清晰，
   幂等：已脱敏文件不会被二次处理；**从不打印密钥值**）。

## 4. 历史清理 runbook（**未执行**，需你明确决定）

现状：脱敏只作用于**当前树**。`bdbc18a`（及更早含 V1.22 密钥的提交）中的**历史 blob 仍含明文**，
任何人 `git log -p` 或按历史 SHA 取 blob 仍可读到。要真正从历史移除，必须**重写历史**。

### 4.1 代价（必须先接受）

- 重写会**改变所有下游提交 SHA** ⇒ 用户授权中钉住的 **`git HEAD 558fb5c…`** 与整条
  **提交号固定的审计链**将失效；签署凭据、裁决记录中引用的 commit 号需要一份**映射表**保留可追溯性。
- 已推送的历史被重写后，**其他人的克隆**不受影响（旧对象仍在他们本地），需强制更新；
  若仓库已被 fork/缓存（GitHub 可能保留一段时间），**无法保证**全网清除。

### 4.2 推荐路径（保留科研可追溯关系）

**步骤 0（已完成）**：原件私有留档 + 单向摘要入公开树 ⇒ 即使历史被清除，**暴露事实与身份**仍可核验。

**步骤 1**：在公开树保留**考古记录**（本报告 + 脱敏存根 + 脱敏账本），说明
"某提交曾含某 sha256 的密钥，于 2026-10-16 脱敏，原件私有留档于何处"。

**步骤 2**：执行历史清除（**建议先做完整镜像备份**，并在**私有**镜像上演练一次）：

```bash
# 备份（含全部 refs 与 reflog），确认可完整恢复后再继续
git clone --mirror https://github.com/GoodThing001/DynMaskCO.git DynMaskCO-backup.git

# 用 git-filter-repo 替换两个文件的历史内容为脱敏存根（--replace-text 的规则文件须私有、勿提交）
cat > /private/purge-rules.txt <<'EOF'
f786b3c202870670226f892f5ace31aa4ff04c52087e47b3be29cc8dc443a389==>REDACTED-EXPOSED-EXECUTION-SECRET
b9600f71696e70d27e75e68778cb65e21e000346f9813e1ad246076182215972==>REDACTED-EXPOSED-EXECUTION-SECRET
EOF
git filter-repo --replace-text /private/purge-rules.txt --force

# 或按路径整文件删除（代价：该次 run 的 guard 目录不再有占位）
# git filter-repo --path-glob '*/guard/execution_secret_*.txt' --invert-paths --force
```

**步骤 3**：生成**旧→新 commit 映射**并作为**新证据**提交，使科研追溯链在重写后仍可对照：

```bash
git log --format='%H %s' > /private/new-history.txt       # 与备份里的旧列表比对，产出映射表
```

**步骤 4**：强推两个分支并通知协作者重新克隆：

```bash
git push --force-with-lease origin main
git push --force-with-lease origin codex/o0cc-cal-phys
```

**步骤 5**：向 GitHub 申请清理不可达对象 / 联系支持（若仓库曾 fork，历史可能仍存在于 fork 中）。

### 4.3 备选路径（**不改历史**，如果追溯关系优先）

- 保持历史原样，接受"历史中仍有已暴露且**已失效**的密钥"；
- 依据：两个密钥**均已一次性消费、无残余授权**，且**每个 run 本来就独立生成新密钥**；
- 配合本报告的考古记录，暴露事实已充分披露。
- 若选此路径，建议**联系 GitHub Support** 评估是否可对特定历史内容做服务端处理。

**我的建议**：若仓库确实公开且被外部索引，选 §4.2（重写历史）**并**保留 §4.2 步骤 3 的映射表；
若仓库实际访问受限或你更重视签署链完整性，选 §4.3。**两条路径我都不擅自执行。**

## 5. 未发现的项（排除清单）

| 检查项 | 结果 |
|---|---|
| `.vscode/sftp.json`（服务器凭据） | **未被 Git 跟踪**；`.gitignore:25` 忽略 `.vscode/`。此前一次检索报"tracked: True"是**我的匹配缺陷**（把 `tools/_sftp_*.py` 误当 sftp 凭据文件），已更正 |
| `.pem` / `.key` / `.p12` / `.pfx` | 0 个受跟踪 |
| 其他 64-hex 裸密钥形态文件 | 0 个（脱敏后复查） |
| 密钥值出现于日志 / JSON / 代码 | 0 处（38,072 个受跟踪文件全量字节扫描） |
| 授权文件 `L1_V2_EXECUTION_AUTHORIZATION.json` | **无密钥**；含 run_id / SHA 绑定 / 人类批准文本，均属**应公开**的可审计内容，**保持原样** |
| 执行上下文 `EXECUTION_CONTEXT_*.json` | 含 `hmac_sha256` 与 `secret_path`，**无密钥值**；按 §3.3 保持原样 |

## 6. 教训（过程缺陷，如实记录）

1. **raw run 目录整体提交**把运行期临时密钥一并带入公开历史。防护：把 `guard/` 与
   任何 `secrets.token_hex` 产物加入 `.gitattributes`/`.gitignore` 白名单机制，
   **在 `git add -A` 之前**按"运行期临时材料"过滤。
2. 本轮的交付脚本用 `git add -A`，**没有任何密钥形态检查**。建议在 `tools/git_deliver.py`
   加一道**预提交扫描**：拒绝任何"64-hex 单行 txt"或匹配 `secret|passwd|token|BEGIN .*PRIVATE KEY`
   的新增/修改文件，除非显式豁免。
3. 该缺陷**由用户复查发现**，不是自检发现；说明交付链缺少密钥扫描这一环。

## 7. 边界

- 本报告**不输出**任何密钥明文；所有引用均为 sha256。
- 本报告**不授权**任何历史重写；§4 是 runbook，执行与否由你决定。
- **未**进行新的真实评分、训练、在线 rollout 或计划安装。
- `stable_online_gain = False` 及全部既有正式裁决不变。
