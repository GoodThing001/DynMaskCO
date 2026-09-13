# O0-CC 源码字节基线与行尾处置

> 状态：**SEALED — 字节封口提交 D 与服务器离机备份均已验证**
>
> 核查日期：2026-09-13
>
> 适用范围：O0-CC 门控源码、正式/诊断冻结包及 CAL-PHYS 起算基线
> 本文只记录审计事实和退出条件；不授权修改 `MASKCO_code/`、重写既有提交或改写冻结证据的科学内容。

## 1. 已确认的提交链

```text
6af56fb  CC_Compare 父提交
└─ 145f3b4  目录迁移
   └─ 93f0e54  O0-CC 门控框架与 Task 4.3 代码
      └─ c5ae9ef  正式及诊断冻结证据
         └─ 3f2b7e3108a3d5c6486ba3b63bbbd0e197d167bc  字节封口提交 D
```

- 当前分支：`codex/o0cc-cal-phys`；
- `MASKCO_code/` 在上述提交链中 0 改动；
- `CC_Compare/` 第三方仓库未进入提交 D；
- 提交中无 `.pyc`、`__pycache__` 或运行缓存；
- 2026-09-13 独立重跑 8 套门控测试，共 **94/94 PASS，exit 0**。

`93f0e54` 表示门控代码落地提交，`c5ae9ef` 表示包含冻结证据的仓库快照，`3f2b7e3…` 表示可跨 checkout 重建的字节封口。后续 manifest 不得用单个含义不明的 `git_commit` 同时代替这些身份。

## 2. 已关闭的身份歧义

正式身份函数对文件执行原始字节 SHA-256。提交 D 前的 Windows 工作树曾得到：

```text
compute_sha256  b0b562f2eeffa8ee37eb75f20c3e3fa6c2cf9f98bbcb754be9cafb9cc3a1b7ad
control_sha256  5332b67e0dc57ccea5d47022f76d2d8ed4f4d7e03170d733ff8bba537d9d5a28
analysis_sha256 c1e78f364f4fb823ebf459a4f1fc63a814dbb7a8c6f2d57c718d63960ecec7fe
```

其中 `run_action_oracle.py` 和 `sequential_oracle.py` 的工作树是 CRLF，而当时 `HEAD` Git blob 是 LF。提交 D 将活跃身份文件固定为 LF；当前工作树、`HEAD` blob 与 fresh worktree 的权威身份一致为：

```text
compute_sha256  9185ad4b66b1e0bd0e6039f3f07fa8066d945a1a37975ef7db42163f7e494300
control_sha256  5332b67e0dc57ccea5d47022f76d2d8ed4f4d7e03170d733ff8bba537d9d5a28
analysis_sha256 c1e78f364f4fb823ebf459a4f1fc63a814dbb7a8c6f2d57c718d63960ecec7fe
```

因此：

- `b0b562f2…` 只能标记为 `windows_worktree_pre_lf_normalization`；
- `9185ad4b…` 是提交 D 后的权威可复现 compute 身份；
- 22/22 个正式身份文件已满足 `raw worktree bytes == HEAD blob bytes`；
- 提交 B message 中的 `f83ad3c4/eb9a7da7/a56f4e55` 是 Task 4.2 旧记录，不代表 Task 4.3 内容；不建议为此 amend B，因为会重写 B 和后继 C。

## 3. 冻结包字节保存风险及关闭证据

提交 C 的 147 个文件中曾有 39 个文件的工作树原始字节与 Git blob 不同，原因是 Git 文本行尾过滤。提交 D 已将两个冻结包根设为 `-text`，并按原封存字节重新登记这 39 个文件。正式 DEV-GATE v4 在当前目录与 fresh worktree 中均验证通过：29 个声明文件，`bundle_sha256=1bbcc978b0f4a60f82389629dc709cbc47373be38bba89d28395b21f9bc5e466`。

冻结包是字节容器，不是可自由规范化的源码目录。正式及诊断冻结包下的所有内容继续使用 `-text` 保存，避免 Git 改写行尾。提交 D 未重新生成旧结果，也未更新旧 verdict。

封口前核查还发现归档目录中存在 19 个未跟踪 `.pyc`，分布于 5 个 `__pycache__` 目录。提交 D 前已在下列两个精确归档根内清理；封口后复查计数均为 0：

```text
C-VRP_Cold-chainVehicleRoutingProblem/archive/正式冻结包/
C-VRP_Cold-chainVehicleRoutingProblem/archive/诊断与作废冻结包/
```

以后复验仍不得使用指向仓库根、用户目录或未解析变量的递归删除。

## 4. 字节封口工序与提交 D 的实际内容

提交 D 共 41 个文件：`.gitattributes`、`.gitignore` 与 39 个按原字节保存的冻结文件。完整封口工序仅包含下列治理性动作：

1. 根 `.gitattributes`：活跃源码统一 LF；冻结包目录使用 `-text`；二进制格式显式 `-text`；
2. 提交前将两个活跃 CRLF 文件的工作树原始字节规范为与既有 Git blob 一致的 LF；因为规范后与 `HEAD` 已一致，这两个文件不构成提交 D 的新增 diff；
3. 将受过滤影响的 39 个冻结包文件按原封存字节重新加入 Git；
4. 精确的本地忽略项，例如 `/.claude/settings.local.json` 与 `/docs/superpowers/`；不得笼统忽略 `/docs/` 或整个 `CC_Compare/`。

提交 D 未在自身内容中写入自己的 commit SHA；Git commit identity 是内容的一部分，这避免了不可解的自引用。`O0CC_SOURCE_BASELINE.json` 已登记 D 的完整 SHA，并在服务器离机备份通过后转为 `sealed`，待后续文档提交 E 保存。

实际提交 SHA：

```text
3f2b7e3108a3d5c6486ba3b63bbbd0e197d167bc
```

## 5. 权威机读基线

权威记录为同目录的 `O0CC_SOURCE_BASELINE.json`。该记录已填入提交 D、三类代码身份、94/94 测试证据、archive 双环境验证、隔离性检查与服务器目的端备份证据，状态为 `sealed`。

离机副本位于 `o0cc-backup-server-01:/home/hzeng/backups/MASKCO-Main/o0cc/codex-o0cc-cal-phys-D-3f2b7e3.bundle`，大小 191134245 bytes，SHA-256 为 `0187fd200b2111dfeddadd9c4e07dc05cb072c22cb7ab2d58198dad577a33c0c`。服务器依次通过仓库上下文中的 `git bundle verify`、`bundle list-heads` 和独立 `clone --no-checkout` 恢复验证；观察到的 ref 为 `refs/heads/codex/o0cc-cal-phys`，tip 与恢复后 HEAD 均为提交 D。bundle 及旁路 `.sha256` 均为 `0444` 只读。

版本化记录使用稳定别名 `o0cc-backup-server-01`，不保存私有 Tailscale 地址；精确网络 locator 保留在执行人的私有传输日志中，其规范字符串 SHA-256 为 `98f57068bd53dd2640aca60d85f8ca9b1d12ecad3fd8dc3e43764fd9d79f2b03`。这样既能由私有原始记录复核，也不会把私有网络拓扑永久写入可能公开的 Git 历史。

服务器 Git 2.51.0 下，第一次在非 Git 仓库上下文直接运行 `git bundle verify` 返回 rc=1，信息为“需要一个仓库来校验一个归档”。该尝试没有执行归档内容验证，分类为前置条件不足，而非 archive 验证失败。按预定 runbook 先执行 `git init` 后复验通过，因此最终 `git_bundle_verify_pass=true` 有效。提交 E 后必须另建带 E SHA 的 bundle 或推送受控远端，不得覆盖本 D bundle。

## 6. 字节基线退出门控

必须全部满足：

- [x] 22 个 identity 文件的工作树原始字节与 `HEAD` blob 完全一致；
- [x] 工作树重算身份等于机读基线；
- [x] 临时 fresh worktree 重算身份等于同一基线；
- [x] 正式 DEV-GATE v4 在当前目录和 fresh worktree 中均 verify PASS；
- [x] 两个归档根中无 `.pyc`、`__pycache__`；
- [x] 8 套门控测试重新 94/94 PASS；
- [x] `MASKCO_code/` 仍为 0 改动；
- [x] `CC_Compare/` 未意外进入提交 D；
- [x] 已生成完整分支 bundle，并在服务器目的端完成 SHA、bundle、ref 和恢复 checkout 三重验证；
- [x] 提交 D 的完整 SHA 已回填进当前机读基线草案；
- [x] 离机证据已回填，机读基线已转为 `sealed`；
- [ ] 由后续文档提交 E 保存本记录；之后的 CAL-PHYS freeze seal 也绑定提交 E 中的本记录字节。

源码字节基线的技术阻断已经清零；提交 E 是把本记录纳入版本历史的最后治理动作。CAL-PHYS 协议自身仍为 `draft_blocked`，还必须关闭现实对象、仪器、设计、样本量与分析阈值阻断后，才可进入相应注册状态。源码基线 sealed 不授权执行 P1，更不授权 confirmatory P2–P6。
