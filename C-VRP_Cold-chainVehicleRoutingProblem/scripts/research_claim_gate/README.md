# `scripts/research_claim_gate/` — 工具说明与行为变更登记

**状态**：本目录是负结果线的四位一体证据审计工具集。`program_support_audit.py` 在 2026-10-10 被改造为 **v2**；按用户裁定**保留 v2**，并以本文件登记其行为变更（不改回、不双轨）。

## 文件

| 文件 | 作用 |
|---|---|
| `program_support_audit.py` | **v2 支持审计器**：把一个"主张是否有支持"拆成四个独立维度 |
| `verify_negative_archive.py` | 183 文件封存包与 C1 原判据的只读校验（**强制输出到被审项目之外**） |
| `evidence_gate.py` | 共用的 fail-closed 断言、有限性检查、JSON/哈希读取 |
| `NEGATIVE_ARCHIVE_CURRENT_INVENTORY.json` | 183 文件 pin 清单 |
| `PROTOCOL_DRAFT.json` | 草案 |
| `test_program_support_audit.py` / `test_evidence_gate.py` | 24 项 / 33 项测试 |
| `reports/support_audit_2026-10-10.json` | 真实运行产物 |

## 四维分离（v2 的核心）

| 维度 | 问题 | 通过的判定 |
|---|---|---|
| `numeric_consistency` | 报告的数字能否从所引归档工件复现 | `reproduced_from_cited_artifacts` / `no_cited_numeric_claim` |
| `eligibility` | 源文件**与原始逐实例轨迹**是否仍在盘、可重放 | `eligible_raw_trajectories_and_sources_present` |
| `statistical_support` | 是否有在注册多重性/alpha 纪律下排除零的逐实例配对区间 | `established_under_registered_discipline` |
| `generalization_scope` | 是否限于实测配置、且措辞未越界 | `scoped_to_measured_configurations` |

**缺输入（`input_missing`）在任何维度都是失败**，永不通过；CI 跨零永远报「未建立」，永不报「等价」。

## 行为变更登记（v1 → v2）

1. **硬错误由"未捕获异常"改为"结构化报告"**：出错时写入 `audit_error` 并以 **2** 退出，而不是抛异常落到退出码 1。**通过/失败契约本身不变**——仍只有 **0（干净）** 与 **2（存在超额/不支持/缺输入/头条不一致/阻塞检查）**，源码 `exit_code_for()` 只返回 0 或 2（已独立复核）。
2. **新增可选参数 `--allow-tool-output`**：默认**不**向被审树写入任何内容；传入后**仅**允许写入本工具自己的 `reports/`（已独立复核 `main()` 的路径校验）。
3. **`claim_excesses` 内容、顺序、措辞逐字未改**（仍是 v1 的 5 项）；所有新发现一律进 `additional_findings`，因此下游不可能把"新增检查"误读成"新增超额"。
4. **自检夹具**：`--selftest` 三个夹具必须分别为 `clean→0`、`unsupported_claim→2`、`missing_input→2`。

> 变更来源：由子代理实施、上一轮会话独立复核；本次续办**再次独立复核**了 `exit_code_for()` 与 `--allow-tool-output` 的两处实现。

## 用法

```powershell
python -B scripts\research_claim_gate\program_support_audit.py --selftest
python -B scripts\research_claim_gate\program_support_audit.py --project . --output <项目外的路径>
python -B scripts\research_claim_gate\verify_negative_archive.py --project . --output <项目外的路径>
```

## 纪律

- 审计器**只读**：历史证据与脚本按数据读取，**从不执行**其代码。
- 输出**必须**写到被审项目之外（`verify_negative_archive.py` 会强制拒绝项目内路径），否则"记录检查"这一动作本身会改动被审对象。
- 非零退出是**正确行为**，不是缺陷；真实运行退出 **2** 是设计结果。
- 本工具**不**授予任何科学结论：退出 0 只表示"在所审文档集内、按四维定义未检出超额"。
