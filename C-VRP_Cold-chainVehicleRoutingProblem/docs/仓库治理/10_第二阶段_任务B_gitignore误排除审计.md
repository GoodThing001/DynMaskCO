# 10 · P0 第二阶段 · 任务 B：`.gitignore` 误排除审计

> 只读审计。**本轮未修改任何 `.gitignore`，未 `git add` 任何被忽略文件，未执行会扩大暴露面的规则变更。**
> 全部为提议；执行需你批准（见 §6）。

## 1. 规则现状（逐行实测）

| 文件 | 行 | 规则 | 作用域 | 评价 |
|---|---:|---|---|---|
| `C-VRP_Cold-chainVehicleRoutingProblem/.gitignore` | 2 | `ckpts/` | **未锚定**（任意深度） | 当前无害（无嵌套 `ckpts/`） |
| 同上 | 3 | `data/` | **未锚定** | **本任务的问题所在** |
| 同上 | 4 | `!scripts/data/` | 含中间分隔符 ⇒ **锚定到该 .gitignore 所在目录** | 只救回 `<ext>/scripts/data/`，**救不到嵌套副本**（§3 已实测） |
| 同上 | 5 | `logs/` | **未锚定** | 当前无害（无嵌套 `logs/`） |
| 根 `.gitignore` | 2 | `logs/` | 未锚定（仓库任意深度） | **必须保持未锚定**：它故意覆盖 `MASKCO_code/logs/` 与各 vendored 仓库的 `logs/` |
| 根 `.gitignore` | 3 | `ckpts/` | 未锚定 | **必须保持未锚定**：`MASKCO_code/ckpts/cvrp100.ckpt`（305 MB）靠它排除。**若改成 `/ckpts/`，该 305 MB 权重会变为未跟踪可见，一次 `git add -A` 就能提交，直接触发 GitHub 100 MB 上限** |

**结论 1**：根 `.gitignore` 的 `logs/`、`ckpts/` **不能动**（改了会暴露 305 MB 权重）。
**结论 2**：真正需要处理的只有扩展 `.gitignore` 第 3 行的 `data/`。

## 2. 未锚定 `data/` 的实际匹配面（精确计数）

在扩展树内、排除 `results/` 与 `__pycache__`、逐个用 `git check-ignore -v` 归因：

| 被 `data/`（扩展第 3 行）匹配的路径组 | 文件数 | 是否本意 |
|---|---:|---|
| `C-VRP_…/data/**`（冻结输入：npz + 3 个 `DEV_MANIFEST.json` + `M0_TEACHER_MANIFEST.json` + `DATA_MANIFEST.sha256` + 4 个 md） | **123** | ✅ 本意（2.17 GB，必须忽略） |
| `C-VRP_…/03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/scripts/data/*.py` | **13** | ⚠️ **误伤**（但见 §3：内容没丢） |
| `C-VRP_…/negative_result_paper/data/*.npz` | **3** | ⚠️ **误伤**（唯一内容） |
| `C-VRP_…/archive/历史工程资产/代码快照/data/*.py` | **2** | ⚠️ **误伤**（唯一内容） |
| **合计** | **141** | |

后缀构成：`117 .npz + 15 .py + 4 .json + 4 .md + 1 .sha256`。

同法核对 `logs/`（扩展第 5 行）：匹配 **107** 个文件，**全部**在 `C-VRP_…/logs/` 下，**没有任何嵌套命中**
⇒ **把 `logs/` 锚定成 `/logs/` 是零影响的 no-op**。`ckpts/` 同理（扩展内无嵌套 `ckpts/`）。
`MASKCO_code/{ckpts,logs}` 由**根**规则排除，属本意。

## 3. 逐个判定:这 13 个"负结果复现 Python 文件"到底是不是必需复现源码

**判据**：与 `C-VRP_…/scripts/data/<同名>.py`（**已跟踪**）逐字节比对。

| 文件 | 与 live 版 | live 是否跟踪 |
|---|---|:--:|
| `CVRPTWDataloader.py` / `ColdChainDataloader.py` / `coldchain_teacher_dataset.py` / `coldchain_visible_features.py` / `generate_coldchain_data.py` / `generate_cvrptw_data.py` / `generate_dev_split.py` / `generate_devcal.py` / `generate_m0_teacher_split.py` / `generate_o0cc_split.py` / `generate_self_training_labels.py` / `regenerate_baseline.py` / `repair_state.py`（共 13） | **13/13 逐字节相同** | 全部 ✅ |

**结论（重要，且更正第一阶段的风险描述）**：

> 这 13 个文件**不是"丢失的复现源码"**——它们的**内容 100% 已在 git 里**（在 `scripts/data/` 下，同名、同字节）。
> 被忽略掉的只有**"它们曾以恢复快照的目录布局存在"这一事实**，不是源码本身。
> 因此这条误排除的**信息损失为零**，风险等级应**下调为低**（第一阶段把它读成"不在 git 里"是按路径说的，
> 按内容说并不成立；此处更正）。

而**真正只有盘的**是另外两组（见 §4）。

## 4. 真正只有盘的历史源码 / 数据（唯一内容）

| 文件 | 字节 | SHA256 | 全仓是否有同内容 | 判断 |
|---|---:|---|---|---|
| `archive/历史工程资产/代码快照/data/add_zone_density.py` | 2,775 | `c00af528e2ca8c57a54ee35aefe4f55148176e2b1400651c9f44302973d0f3fb` | **无** | **必需**：历史工程资产，唯一副本 |
| `archive/历史工程资产/代码快照/data/lkh3_solve.py` | 8,158 | `37eca057d6df2198873e6873581849750108124a8b61f22af43965e10ddae1cd` | **无** | **必需**：LKH3 求解器封装，唯一副本 |
| `negative_result_paper/data/edod_0.0.npz` | 211,754 | `ea5c194d1cdfa4e74adf61f670cb25c70fc05a1e7fe55389cbb2e39f6da176da` | **无** | **必需**：负结果论文"揭示度轴"实验的冻结输入（`results/07_揭示度轴/` 三个 EDoD 实验的**数据来源**） |
| `negative_result_paper/data/edod_0.2.npz` | 212,282 | `23dc4fe0a092c9c71db49de94b96ea0c9a8ba1342046bd06f70b1ca487b319f6` | **无** | 同上 |
| `negative_result_paper/data/edod_0.8.npz` | 213,158 | `9a619829a7a6554dee0d8f33ee750c21bb6a3319e9d9fb7b8a1c862359682271` | **无** | 同上 |

**这三份 npz 尤其关键**：`negative_result_paper/README.md` §3 的复现命令只声明了 `data/m0_scale/*.npz`，
**从未声明自己文件夹里的 `data/edod_*.npz`**，而这正是 `07_揭示度轴` 那组结果的输入。
换句话说：**论文的"揭示度轴"证据，其输入数据既不在 README 里，也不在 git 里，只在这台机器上。**
（这一点同时是 `negative_result_paper` 的文档缺陷，见 §7。）

## 5. 顺带发现：所有**数据身份清单**本身都不在 git

| 清单 | 作用 | 跟踪 |
|---|---|:--:|
| `data/baseline/DATA_MANIFEST.sha256` | 30 个基线 npz 的 sha256 登记表 | ❌ |
| `data/baseline/50_node/{dev_cal,dev_gate,dev_proto}/DEV_MANIFEST.json` | 三个角色切分清单（`dev_cal` 内容哈希 `18266cdc…` 被写进 `objective_profile.json`） | ❌ |
| `data/m0_scale/M0_TEACHER_MANIFEST.json` | 教师数据集清单 | ❌ |
| `data/README.md` + 3 份数据规范/现状 md | 数据目录说明 | ❌ |

**后果**：`data/` 里 2.17 GB 数据被忽略是**正确的**，但**它们的哈希登记表同样被忽略**，
所以**干净 clone 连"数据是否被替换过"都无法判定**。
这与 `results/o0cc/scale_v2/objective_profile.json` 里冻结的 `dataset_sha256` 形成**单向依赖**：
profile 已入库（有 9 条 dataset_sha256），而数据与数据清单都没入库。
→ 建议把 **`DATA_MANIFEST.sha256` 与 3 个 `DEV_MANIFEST.json` 单独入库**（合计 < 100 KB，纯文本/JSON）。
这是本任务里**性价比最高的一个动作**：不动 `.gitignore`，用 `git add -f` 精确加 4 个文件即可，风险极低。

## 6. 最小化修订方案（三选一，**本轮均未执行**）

### R1 —— 只锚定，不加例外（diff 最小）

```diff
- data/
+ /data/
```
影响：`data/` 的 123 个文件仍被忽略；**18 个文件变为未跟踪可见** =
13 个重复的 repro `.py` + 2 个 `archive/历史工程资产/代码快照/data/*.py` + 3 个 `negative_result_paper/data/*.npz`。
副作用：13 个重复文件会造成 `git status` 噪声，且若误用 `git add -A` 会把它们重复入库（约 +110 KB 纯冗余）。

### R2 —— 锚定 + 只回放唯一内容（**推荐**）

```diff
- data/
+ /data/
+ # 恢复快照里的 scripts/data/ 与 scripts/data/ 逐字节相同（13/13），不再重复入库
+ 03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/scripts/data/
```
影响：**恰好 5 个唯一文件**变为可见（2 个 archive `.py` + 3 个 npz，合计约 436 KB），零冗余。
`logs/`、`ckpts/` 不动（已证明是 no-op，改它们只增加 diff 风险）。

### R3 —— 零 diff（保守）

不动 `.gitignore`；改为把 5 个唯一文件的 SHA256 抄进索引（**已在 `11` 完成**），
并把 13 个重复文件的"路径 → 已跟踪等价路径"映射写进文档，使恢复快照的布局**可重建**。
优点：零风险、零暴露；缺点：5 个唯一文件仍只在磁盘上。

### 我的建议

**R3 立即生效（本轮已完成，零改动）** + **R2 作为下一个审批批次** + **§5 的 4 个清单文件用 `git add -f` 精确入库**
（三者可分开批准，互不依赖）。

## 7. 变更影响清单（若批准 R2）

| 类别 | 具体 | 数量 | 是否需额外动作 |
|---|---|---:|---|
| 变为未跟踪可见 | `archive/历史工程资产/代码快照/data/{add_zone_density,lkh3_solve}.py` | 2 | 建议随批次 `git add` |
| 变为未跟踪可见 | `negative_result_paper/data/edod_{0.0,0.2,0.8}.npz` | 3（~640 KB） | ⚠️ **需先确认**：`negative_result_paper/` 是**冻结归档**，其 `scripts/research_claim_gate/verify_negative_archive.py` 会**遍历该目录每个文件求哈希**；新增"被跟踪"状态本身不改变工作树，但会让该归档首次出现二进制入库 —— 需你确认是否接受 |
| 保持忽略（不变） | `C-VRP_…/data/**` | 123 | 无 |
| 保持忽略（不变） | `C-VRP_…/logs/**` | 107 | 无 |
| 保持忽略（不变） | `MASKCO_code/{ckpts,logs}` | — | 无（**根规则不动**） |
| 重新忽略 | 13 个重复 repro `.py` | 13 | 无（内容已在 `scripts/data/`） |
| 仍不可复现 | `data/` 的 2.17 GB 与 4 个身份清单 | — | 见 §5 建议 |
| 对代码/实验的影响 | **零** | — | 只影响 git 跟踪状态，不改任何字节、不改任何判据 |

**风险点（必须写清）**：
1. 若同时锚定**根** `.gitignore` 的 `ckpts/`，会暴露 `MASKCO_code/ckpts/cvrp100.ckpt`（305 MB）⇒ **禁止**。
2. 改 `.gitignore` 后 `git status` 条目会从 2 条涨到十几条；**交付必须继续用显式 `--paths`**，
   绝不能用无 `--paths` 的 `git add -A`（`git_deliver.py` 的默认分支就是 `add -A` + 排除项，**必须避免**）。
3. `negative_result_paper/` 的任何增删都会改变 `verify_negative_archive.py` 的逐文件哈希基线
   ⇒ 该归档的入库动作应**单独一批**，且入库后不得再动其内容。

## 8. 本轮未做（明确边界）

- 未修改任何 `.gitignore`。
- 未 `git add -f` 任何被忽略文件。
- 未移动/删除 `data/`、`logs/`、`ckpts/`、`archive/`、`negative_result_paper/data/` 下任何文件。
- 未改变任何实验输入的字节。
