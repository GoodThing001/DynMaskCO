# A1-D3-L1 任务 B：F backbone 训练期身份溯源（只读审计）

- 轮次：`A1-D3-L1-MECHANISM-PREREG` / 任务 B
- 用户裁决②：**要求先补训练期溯源证据，不接受直接认定同一文件**；允许审计训练日志与历史文件身份
- 本次**未**进行重新训练，**未**运行 F−D-Geo 比较；全部为只读查询
- 机器可读证据：`A1_D3_L1_BACKBONE_PROVENANCE.json`

## 1. 结论（分级标签）

| 标签 | 状态 |
|---|---|
| `CURRENT_BACKBONE_CANDIDATE` | **成立**：当前文件 `MASKCO_code/ckpts/cvrp100.ckpt`，sha256 `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`，305,061,150 B |
| `TRAINING_BACKBONE_PATH_BOUND` | **新成立**：训练运行的命令记录显式指定 `--cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt`，且该机器上**只有这一个** `cvrp*.ckpt` |
| `TRAINING_BACKBONE_SHA_RECORDED_IN_RUN` | **不成立**：没有任何**写入训练运行记录**的 backbone SHA（`training_identity.json` / `summary.json` 均无该字段） |
| `TRAINING_BACKBONE_VERIFIED` | **暂不登记**，理由见 §3（残差假设） |

**建议标签**：`TRAINING_BACKBONE_VERIFIED_BY_COMMAND_RECORD`（推荐，需用户批准）；
若用户不接受，则保持 `HISTORICAL_BACKBONE_UNVERIFIED`。
在获批准之前，任何使用该文件重建的模型必须标注
`F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`。

## 2. 证据表

### 2.1 训练运行的命令记录（新增、最关键）

服务器 `/home/hzeng/project/MASKCO-Main/_run_mpre_seed.sh`（mtime **Sep 23 08:41**，早于训练）：

```bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/training/train_mpre_reinforce.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --num-steps 1000 --batch-states 8 --K 4 --seed "$SEED" \
    --out "results/m0_scale/mpre_reinforce_s${SEED}_clean" \
    > "results/m0_scale/_mpre_s${SEED}_clean.log" 2>&1
```

该记录同时绑定了：训练脚本、数据文件、**backbone 路径**、超参（1000/8/4，与 checkpoint 元数据一致）、
输出目录名（`mpre_reinforce_s${SEED}_clean`，与持有的 checkpoint 目录一致）与 GPU 环境变量。

### 2.2 服务器文件事实（只读）

| 项 | 值 |
|---|---|
| `MASKCO_code/ckpts/cvrp100.ckpt` sha256 | `03615b08…c6d38db`（与本地一致） |
| 该文件 size / mtime | 305,061,150 B / **2026-09-20 20:12:16 +0800**（早于训练运行） |
| 该机器上 `cvrp*.ckpt` 副本数 | **1**（`find /home/hzeng -maxdepth 7 -name 'cvrp*.ckpt'`） |
| `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt` sha256 | `f5acf83f…903a7f5`（= 本地登记值） |
| 该目录服务器 mtime | `model.ckpt` 2026-09-22 11:11:34；`training_identity.json` 11:15:53 |
| 训练日志 | `results/m0_scale/_mpre_s42_clean.log` 存在（该次运行的 stdout/stderr 捕获） |

> 注：本地该目录的 mtime 是 **2026-09-24**（同步所致），因此**本地 mtime 不能用作训练时间证据**；
> 服务器 mtime（2026-09-22）才是运行时间证据。

### 2.3 独立登记过该 SHA 的历史记录（多处、相互独立）

| 记录 | 内容 |
|---|---|
| `IMMUTABLE_BASELINE.json:403` / `PROTECTED_BEFORE.json:403` | `"../MASKCO_code\ckpts\cvrp100.ckpt": "03615b08…"` |
| `SHA256SUMS.txt:1` | `03615b08…  MASKCO_code/ckpts/cvrp100.ckpt` |
| `protocol.json` | `cvrp_sha256: 03615b08…`（多处） |
| `equipment_matrix`/`evidence_matrix.json:988`、`candidate_manifest.json:1082` | `"sha256": "03615b08…"` |
| `论文封存与收口检查_2026-10-09.json:1045-1046` | `expected_sha256 == actual_sha256 == 03615b08…` |
| `I3固定教师决策训练_执行预声明_2026-10-08.md:157` | "CVRP 权重 SHA=`03615b08…`" |
| `I3_固定训练验收与在线入口恢复_2026-10-09.md:33` | "原CVRP：`03615b08…`" |
| `方向A_补跑进度与I3实现准备_2026-10-08.md:273` | `03615b08…` 通过 |
| `配送_多步教师完整模型前后向与独立复算验收_2026-10-11.md:11` | "原CVRP checkpoint SHA256=`03615b08…`；原 234 叶 76,259,840 参数" |
| `_run_eval_all.sh:13`、`_run_step5.sh:13` | `CVRP="../MASKCO_code/ckpts/cvrp100.ckpt"`（同线其它运行共用同一路径） |

## 3. 残差假设（为什么仍不直接登记 `TRAINING_BACKBONE_VERIFIED`）

1. **没有 SHA 被写入训练运行自身的记录**：`training_identity.json` 只记 `src_sha256 / data_sha256 /
   model_ckpt_sha256`，`summary.json` 只有超参与 loss 历史；`--cvrp-ckpt` 的值只存在于**外壳脚本**中。
2. **命令记录 → 文件字节**这一跳依赖：该路径上的文件自运行以来未被替换。证据是服务器 mtime
   （2026-09-20，早于运行）与"全机只有一个 `cvrp*.ckpt`"，但 mtime 本身不等于字节不变性证明。
3. 上述 §2.3 的 SHA 登记**多在 2026-10-08 之后**（训练之后），它们证明的是"该文件在 10-08 时等于该 SHA"，
   而不是"训练时加载的字节"。

因此本文件**不推定通过**：`TRAINING_BACKBONE_VERIFIED` 需用户批准；
推荐按 §1 的 `VERIFIED_BY_COMMAND_RECORD` 处理并明确残差假设。

## 4. 尚未执行、可选的低成本补证（如需进一步收敛）

1. 读取服务器 `results/m0_scale/_mpre_s42_clean.log`（运行日志）确认无其它 backbone 路径引用；
2. 检索服务器 shell 历史（`.bash_history` 存在，160,003 B，mtime 2026-10-04）中是否存在
   直接调用 `train_mpre_reinforce.py` 且带 `--cvrp-ckpt` 的命令行；
3. 检索是否有更早（≤2026-09-22）的、把 SHA 与该路径绑定的记录（例如备份清单、上传清单）；
4. 若均无，则按 §5 处理。

（以上均需另行授权；本轮未执行。）

## 5. 若历史身份最终不可追溯

- 允许使用当前文件做一次**工程级**模型恢复与评分接口验证，标签必须为
  `F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED`；
- 该评分器**不得**被无条件称为"历史训练 F 的精确复现"，**不得**用于正式学习贡献归因；
- 是否建立一份身份完整的新 F，必须**另立预声明**决定；**不得**用重新训练的权重悄悄替换历史主模型。

## 6. Claim limits

1. 本文档只回答"backbone 身份证据到什么程度"，不回答 F 是否有效、是否优于 D-Geo。
2. 未执行任何训练、未构造模型、未前向；不含 F−D-Geo 比较。
3. 服务器只读查询由项目既有 `tools/_srv.py` 完成；未在服务器写入任何文件。
4. I3 正式负结果、D2 `not_qualified`、D3 G0、D8、E2 与 `stable_online_gain=False` 均不变。
