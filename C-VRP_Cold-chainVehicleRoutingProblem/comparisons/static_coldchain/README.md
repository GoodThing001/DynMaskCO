# 当前静态冷链物流对比方法源码与适配

文档版本：[全项目文档索引](../../docs/文档索引与版本状态.md)。

2026-10-20。用户要求为当前静态任务新建独立源码目录。本目录与旧CC_Compare分开；旧动态适配器、旧成绩和资格不直接继承。研究目标为最大单次配送净效益，但定价/车辆合同尚未闭合，**当前已运行的只有经典路由基线**。

- `upstream/`：7个作者/官方仓库的8份固定commit快照，保留LICENSE、源码和上游附带文件；未执行其训练/生成脚本，也未解包附带的模型/数据压缩文件。快照里的包文件包括原作者__init__.py，原样保留；本地新增适配器不建包__init__.py。
- `archives/`：原下载ZIP，本地保留。
- `SOURCE_PINS.json`：固定commit、URL、ZIP SHA、逐成员核验摘要；各仓库`SNAPSHOT_PROVENANCE.json`保留完整逐文件SHA。
- `adapters/`：本项目独立适配器与检查；不修改upstream。
- `.runtime/packages/`：pip --target隔离的运行库；版本见`RUNTIME_VERSIONS.json`。一次venv ensurepip失败的空运行目录保留，实际未使用。
- [方法适用性](方法适用性.md)、[机器分类](method_eligibility.json)：评估原17项及新增ALNS/HGS/CaR。

本轮已保存：ALNS、OR-Tools、PyVRP、HGS-CVRP、RouteFinder、DeepACO、CaR。多数为MIT，OR-Tools为Apache2.0，以各源LICENSE为准。未把所有既有大型权重再次复制进新目录。数据和论文许可与代码许可分别核定。

本次实际求解使用官方OR-Tools9.14.6206、PyVRP0.14.0 wheel，不声称已经编译本地C++源码。其源码快照固定版本另有SHA。新目录原文件不得混入旧动态补丁。

PyVRP实际运行匹配源码为`upstream/PyVRP-0.14.0/`，官方v0.14.0 commit`5d9776a954b810bdb3fe71d47b1e7de7cffe90d2`。早期按既有清单下载的`upstream/PyVRP/`项目版本为1.0.0a0，单独保留其身份，不冒称为实际执行的0.14.0源码。

从扩展根运行：

```powershell
python -B -X utf8 comparisons/static_coldchain/adapters/solomon_preliminary.py --out results/static_delivery/solomon_new_run --seconds 10
python -B -X utf8 comparisons/static_coldchain/adapters/quality_evidence_demo.py --out results/static_delivery/quality_new_run.json
python -B -X utf8 comparisons/static_coldchain/adapters/test_evidence_and_solomon.py
```

输出路径须为新路径。Solomon结果是先少车后短距离的100客户经典验证，独立双精度评价；不计算商品利润。品质程序只计算文献参数的变化量；`require_profit_inputs`在缺少价格等证据时拒绝利润计算。两者未接入既有run_rrp命令，不改变原RRP目标。

净效益适配顺序：ALNS+共同独立评价器，小例数学规划，再接MaskCO/RouteFinder/DeepACO；CaR作为复杂约束候选。相同静态数据、硬约束、目标与计算预算是资格前提。原距离模型仅事后重算净效益时，应标成迁移控制，不称同目标方法。

完整研究和[导师初步结果](../../docs/static_delivery_research_20261020/initial_results.md)另见[证据包](../../docs/static_delivery_research_20261020/README.md)。研究任务本身未提交/推送；用户随后授权统一发布，范围见[GitHub同步说明](../../../GITHUB_SYNC.md)。没有运行新神经训练或改动旧任务。
