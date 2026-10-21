# AGENTS.md

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。当前使用文档与历史快照的范围以索引为准。

## 当前任务与权威入口

用户于2026-10-21确认科研冻结并进入MASKCO算法改进：静态冷链、多车、所有车辆总净效益最大化，其余为硬约束；并授权RTX5090服务器实验。当前入口为`C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md`、`docs/static_delivery_research_20261021/算法阶段_v0.md`和`scripts/static_profit/run.py`。72项模型/数据/分析文件按FREEZE_LOCK封存，算法仅train/validation，达到预先分析计划门槛并另行锁定方法与检查点后才开正式测试。旧单车/多车RRP与动态取货均保留原身份，不覆盖当前目标。

## 工程规则

1. `MASKCO_code/`与`CC_Compare/models/`中的第三方原代码保持只读。用户于2026-10-21授权CC_Compare按data/models/scripts/configs/docs/results/checkpoints重构，仅保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR；新增算法仍放扩展目录，CC_Compare/scripts仅作委托与核验入口。原comparisons/static_coldchain来源快照及冻结路径保持不变。
2. 当前最大净效益入口为`scripts/static_profit/run.py`，固定`publication_model_v1.json`及严格评价器；利润单位U、原100客户、20吨、K≤100。旧RRP入口`scripts/static_delivery/run_rrp.py`保留原v3/33托盘口径与显式K，不混用。使用`python -B`，不得新增扩展`__init__.py`，用`scripts/project_paths.py`解析根目录。
3. 固定作者commit、原JSON/TXT和SHA。不得生成客户实例、伪造坐标或用无文献pilot填物理缺项。保留目标/profile/单位身份。
4. 当前业务目标为品质折算交付价值减距离与制冷费用的总净效益，容量/时窗/品质/回仓/覆盖为硬约束；学习交叉熵不替代业务目标。旧RRP年化牵引、制冷、工资欧元仅为历史基准身份，不能当当前目标。
5. 数据按几何分组防D1/D2泄漏；正式训练显式split。连续等待和有向路线均由独立评价器重算，不复制参考解时刻作为评价。
6. 全参考审计不表示所有成本已一致，不冒称LAHC/MIP/CP复现，不用功能smoke推断泛化或正收益。
7. 后端/GPU选择先于JAX导入。用户已授权SFTP服务器运行；长任务用独立tmux、明确资源和输出目录，不碰其他运行任务、旧最终种子或正式历史实验。
8. 取货/负结果关键材料在`archive/transition_20261020/`。任何清理先核验归档及当前SHA，拒绝穿透Windows reparse/junction；用户原private、原未跟踪4项、`.git`和凭据留原地。禁止git clean/reset、整仓git add、自动提交或推送。
   本轮目录整理例外：用户已确认其余对比方法归档后移出活动树，旧CC_Compare整体保存在`C-VRP_Cold-chainVehicleRoutingProblem/archive/cc_compare_reorganization_20261021/legacy/CC_Compare/`，包括MAPT内嵌Git历史与未跟踪文件；原字节核验，根`.git`及其他保护范围不移动。
9. 用户随后明确授权推送到GoodThing001/DynMaskCO，并同步版本化目录结构；本轮可显式提交并正常快进推送main/当前分支，不得force push或改写历史。凭据、private及大型本地归档/模型排除范围见GITHUB_SYNC.md。恢复包仍为历史证据，原始失败/NO-GO和已知缺失不改写。

## graphify

仅用户键入`/graphify`时使用`~/.Codex/skills/graphify/SKILL.md`；先读取该技能再按其执行。

旧完整说明原字节已保存在`archive/transition_20261020/pickup_negative_key_evidence.zip`的`AGENTS.md`成员，逐件SHA见MANIFEST.json。
