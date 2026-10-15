# 独立审查意见：F backbone 训练期身份（**撤回**原报告的建议标签）

- 类型：**新增的独立审查意见**（不改写、不覆盖 `A1_D3_L1_BACKBONE_PROVENANCE.md` / `.json` 原文）
- 触发：用户复核指出原报告 §2.1 的"脚本 mtime 早于训练运行"**与事实矛盾**
- 结论：原报告给出的 `TRAINING_BACKBONE_VERIFIED_BY_COMMAND_RECORD` **撤回**；按用户裁定改标
- 本意见仅为溯源结论，不含任何 F−D-Geo 比较、不含任何增益主张

## 1. 我原来的错误

`A1_D3_L1_BACKBONE_PROVENANCE.md` §2.1 称 `_run_mpre_seed.sh`（mtime **2026-09-23 08:41**）"早于训练运行"。
事实相反：该 checkpoint 于 **2026-09-22 11:11** 生成，脚本晚于它约 **21 小时**。
因此**不能**用现有脚本来证明"训练该 checkpoint 时使用过这条命令"。用户判断正确。

## 2. 修正后的时间线（服务器只读证据，全部为运行自身留下的记录）

| 时间（+0800） | 证据 | 含义 |
|---|---|---|
| 2026-09-20 20:12:16 | `MASKCO_code/ckpts/cvrp100.ckpt` mtime | backbone 文件在该时刻已存在且此后未变（**这一条不因脚本矛盾而失效**） |
| 2026-09-22 **09:47:23** | `_mpre_clean.log` 内 XLA 时间戳 | **s42-clean 运行开始**（该日志即 s42-clean 的运行日志） |
| 2026-09-22 09:47:09 | `mpre_reinforce_s42_clean/state_manifest.json` mtime | 状态清单落盘（256 点 / 64 实例） |
| 2026-09-22 11:11:34 | `model.ckpt` + `summary.json` + `_mpre_clean.log` 结束 | 运行完成 1000 步 |
| 2026-09-22 11:15:53 | `training_identity.json` | 身份文件写入 |
| **2026-09-23 08:41:42** | `_run_mpre_seed.sh` mtime | **脚本当前内容晚于 s42 运行** |
| 2026-09-23 11:27:31 / 11:28:25 | `mpre_reinforce_s43_clean/`、`s44_clean/` | 后续两个种子的运行 |

运行日志的关键内容（`_mpre_clean.log`）：

```
collected 256 decision points across 64 instances; s_TRAIN=0.1334
Nv_max=29 M_max=29
... step 1000/1000 loss=-0.0357 reward_mean=-0.061
done: 1000 updates, final_loss=-0.0357, s_TRAIN=0.1334
saved: results/m0_scale/mpre_reinforce_s42_clean/model.ckpt + summary.json
```

要点：该日志**确认了 s42-clean 的运行时间、实例数、状态数、s_TRAIN 与输出目录**，
但 `grep -l 'cvrp\|train_mpre_reinforce\|--data' _mpre*.log` **无任何命中** —
**python stdout 不记录 argv，故日志无法绑定 backbone**。

## 3. 时间矛盾的解释（有证据，非推定）

1. 脚本的日志重定向是 `> results/m0_scale/_mpre_s${SEED}_clean.log`；服务器上**恰好**存在
   `_mpre_s43_clean.log`（09-23 11:27）与 `_mpre_s44_clean.log`（09-23 11:28）——与脚本命名完全一致；
2. 而 s42-clean 的日志名为 **`_mpre_clean.log`**（不含 seed），**不符合**脚本的重定向模板；
3. 脚本 mtime（09-23 08:41）**早于** s43/s44 运行（09-23 11:27/11:28），**晚于** s42 运行（09-22 11:11）。

⇒ 现有脚本最可能是**为 s43/s44 而写**（按 seed 参数化，连带改变日志名）。
s42 当时使用的命令形式未被保留；脚本的 `--out "…_s${SEED}_clean"` 模板与 s42 目录名一致，
但那只说明**配方一致**，不构成**执行绑定**。
另：`~/.bash_history`（1,962 行）中 `grep 'mpre_reinforce'` **无命中**；
`.bash_history` mtime 为 2026-10-04，且存在旋转临时文件，故历史命令可能未被保留。

## 4. 修正后的标签（采纳用户裁定）

| 标签 | 状态 |
|---|---|
| `CURRENT_BACKBONE_CANDIDATE` | **true**（`MASKCO_code/ckpts/cvrp100.ckpt`，sha256 `03615b08…c6d38db`，305,061,150 B） |
| `BACKBONE_PATH_DOCUMENTED_IN_SCRIPT` | **true**（脚本明确写 `--cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt`；该机器仅一份 `cvrp*.ckpt`） |
| `TRAINING_EXECUTION_BINDING_UNVERIFIED` | **true**（脚本晚于 s42 运行；日志与 history 均无法绑定） |
| `HISTORICAL_BACKBONE_SHA_UNVERIFIED` | **true**（没有任何写入训练运行记录的 backbone SHA） |
| ~~`TRAINING_BACKBONE_VERIFIED_BY_COMMAND_RECORD`~~ | **撤回（用户不批准）** |
| `TRAINING_BACKBONE_VERIFIED` | **false** |

仍然成立（不因撤回而失效）：backbone 文件自 **2026-09-20** 起未变（服务器 mtime，早于 s42 运行）；
本地与服务器 `cvrp100.ckpt` 及 `training_identity.json`（`e8ede390…`）**逐字节相同**；
服务器 `model.ckpt` sha 等于登记值 `f5acf83f…`。

## 5. 剩余缺口与下一步（P1 未完成项）

1. 训练前备份/上传清单中是否存在早于 2026-09-22 且绑定该路径的 SHA（未查完）；
2. 旋转的历史 shell 记录或 tmux 抓取中是否存在 s42 当时的直接命令行（未查完）；
3. 结论：**在找到"绑定 s42 运行"的独立记录之前，`HISTORICAL_BACKBONE_SHA_UNVERIFIED` 保持不变**。

**建议（采纳用户方案）**：不因该缺口让项目无限等待——可**单独**批准
`F_RECONSTRUCTED_WITH_CURRENT_BACKBONE` 的**加载与评分接口工程验证**（不涉及 F−D-Geo 比较），
结果必须同时标注 `HISTORICAL_BACKBONE_UNVERIFIED`，**禁止**直接用作正式学习贡献证明；
是否建立一份身份完整的新 F，须另立预声明决定，**不得**用重训权重悄悄替换历史主模型。

## 6. Claim limits

1. 本意见仅修正身份证据强度，不评价 F 的有效性，也不含 F−D-Geo 结果。
2. 未执行训练、未构造模型、未前向；服务器侧全部为只读查询，未写入任何文件。
3. 时间线证据来自运行自身留下的文件与日志时间戳；未对 mtime 做篡改检测（不做此主张）。
