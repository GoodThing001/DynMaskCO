# REPOSITORY_MAP：DynMaskCO 仓库地图

本文件是**已发布仓库**的入口地图，说明每个目录是什么、当前结果在哪里、哪些内容有意留在本地。
项目内部细节以 [工作区导航](工作区导航.md)、[项目当前状态](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md)
和 [GITHUB_SYNC.md](GITHUB_SYNC.md) 为准；本文件只回答"在 GitHub 上该读哪个文件"。

更新日期：2026-11-21（与《MASKCO_完整模型与项目手册_20261121》同版）。

## 1. 当前结论在哪里读

| 想知道 | 读这个已发布文件 |
|---|---|
| 完整模型、机制、数据、结果与消融（最新手册） | [research/…/model_documentation_20261121/output/MASKCO_完整模型与项目手册_20261121.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/output/MASKCO_完整模型与项目手册_20261121.md)（同目录另有 `.pdf`、`.docx`） |
| 手册的项目导航与阅读顺序 | [project_navigation.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/build/project_navigation.md) |
| 结果的逐项解释与统计口径 | [results_interpretation.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/build/results_interpretation.md) |
| 数据、方法、指标的对照表 | [data_methods_metrics.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/build/data_methods_metrics.md) |
| FLEET_PRICE 扩展（不替代主结果） | [fleet_price_explanation.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/build/fleet_price_explanation.md) |
| 20 页汇报逐页讲稿 | [MASKCO_20页PPT逐页汇报稿_20261121.md](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/output/MASKCO_20页PPT逐页汇报稿_20261121.md) |
| 统一 16 法比较的端点、配对表与分组报告 | [unified_external_20261120](research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120)（`**/RESULTS.md`、`**/endpoints.csv`、`**/paired*.csv`） |
| 锁版 ORDER_SPLIT 的实现与 Split 求解器 | [maskco_perishable_external_20261119/reserved_validation](research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation)（`solver/*.py`、`solver/split_order.cpp`、`locked_method.json`） |
| 锁版配置与权重身份 | `reserved_validation/solver/locked_method.json`、`reserved_validation/locked_method.json` |

当前口径（手册第 1.4 节与第 626 行合计表）：锁版 **ORDER_SPLIT** 在 **62** 个不同百客户案例上，
与 **12** 个外部方法加 **3** 项内部参照按同一协议比较；三个完整批次共 **2,976** 个终点，
**81,251** 条路线通过独立复算；主比较使用单物理核、共同可行初解、模型就绪后 60 秒预算。

**已知呈现限制**：手册与汇报稿的 Markdown 里，图片引用写的是本机绝对路径
（`D:/PyCharm_/MASKCO-Main/research/...`），因此在 GitHub 页面上不会内联渲染。
图片本体已随手册发布：手工章节用图在
[model_documentation_20261120/build/figures](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261120/build/figures)，
21 日新增消融图在
[model_documentation_20261121/figures](research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121/figures)；
需要带图阅读时请下载 `output/*.pdf`（已发布）。

## 2. 顶层目录

| 路径 | 内容 | GitHub |
|---|---|---|
| [C-VRP_Cold-chainVehicleRoutingProblem](C-VRP_Cold-chainVehicleRoutingProblem/README.md) | 当前主项目：脚本、配置、文档、数据、结果 | 发布 |
| [CC_Compare](CC_Compare/README.md) | 10 个对比方法的数据/模型/脚本/配置/文档 | 发布（作者源码原字节只读） |
| [MASKCO_code](MASKCO_code) | 原 MASKCO 作者代码，冻结只读 | 发布，不改字节 |
| [research](research) | 研究执行树（手册、报告、逐单元输出、归档） | **只发布白名单文件**，其余为本地 4.7 GB |
| [results](results) | 根级汇总结果 | 本地保留 |
| [tools](tools) | 仓库交付工具（本文件与 `git_deliver.py`） | 发布 |
| [tmp](tmp) | 临时目录 | 本地，忽略 |
| `AGENTS.md` / `CLAUDE.md` / [工作区导航.md](工作区导航.md) / [GITHUB_SYNC.md](GITHUB_SYNC.md) | 项目规则、导航与同步边界 | 发布 |

## 3. C-VRP 主项目结构

| 路径 | 用途 |
|---|---|
| [scripts/static_profit/run.py](C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py) | **当前净效益入口**（`publication_model_v1.json` + 严格评价器） |
| [scripts/static_delivery/run_rrp.py](C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py) | 历史 RRP 入口（v3/33 托盘口径，不与当前口径混用） |
| `scripts/tests/` | 验收测试 |
| `configs/static_delivery/` | 预注册配置与执行登记（含 SHA 文件） |
| `docs/static_delivery_research_20261021/` | 研究协议、执行记录、评估方案 |
| `data/` | 冻结数据与来源清单 |
| `archive/` | 恢复包、历史快照、服务器同步归档 |
| `results/` | 冻结结果、验收记录、检查点目录 |
| `comparisons/` | 对比方法的委托与核验入口 |
| [项目当前状态.md](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md) | 当前状态与最近阶段结论（按时间倒序追加） |
| [00_项目导航.md](C-VRP_Cold-chainVehicleRoutingProblem/00_项目导航.md) | 项目内导航 |

## 4. 本地专用与发布边界

`research/` 在 `.gitignore` 中被整体忽略（`/research/`），因为其中约 4.7 GB 是逐单元输出、
渲染图、归档 ZIP 与权重，不适合放进 Git。**手册、报告和实现不在被忽略之列**：它们由
[tools/git_deliver.py](tools/git_deliver.py) 按显式白名单强制加入索引，所以能在 GitHub 上看到，
同时又不会把原始输出或凭据带进仓库。

已知的本地专用内容与不发布理由，见 [GITHUB_SYNC.md](GITHUB_SYNC.md) 的表格（服务器配置与凭据、
检查点权重、论文全文、作者代码运行环境、归档 ZIP、临时目录）。作者源码的许可不因本次同步而扩展。

## 5. 同步已发布内容

```powershell
python -B -X utf8 tools\git_deliver.py              # 只打印清单：分组、文件数、体积
python -B -X utf8 tools\git_deliver.py --check      # 清单文件是否都已被 Git 跟踪
python -B -X utf8 tools\git_deliver.py --apply      # 显式 git add -f（不提交、不推送）
```

工具自带守卫：默认 dry run；`EXCLUDE` 永远优先于 `INCLUDE`（逐单元输出、渲染图、日志、
归档、权重、字节码无法被误加）；单文件超过 8 MB、总量超过 60 MB、或路径命中凭据特征时
直接失败而不是静默跳过。提交与推送仍需人工显式执行，符合 AGENTS.md 第 8/9 条。

## 6. 推荐阅读顺序

1. 本文件 → [项目当前状态](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md)（了解最近结论）
2. 《MASKCO_完整模型与项目手册_20261121》：章节 1 概览 → 模型 → 算法 → 统一实验 → 消融 → FLEET_PRICE
3. [unified_external_20261120](research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120) 的 `RESULTS.md` 与 `endpoints.csv`（核对数字）
4. `reserved_validation/solver/` 的代码（核对实现）
