# AGENTS.md

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。当前使用文档与历史快照的范围以索引为准。

## 当前任务与权威入口

用户已批准2026-10-20静态冷链配送迁移，选择先复现公开单车RRP，再扩展多车。当前入口为`C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md`和`docs/static_delivery/README.md`。旧动态取货规则只适用于历史证据，不能覆盖此用户明确新指令。

## 工程规则

1. `MASKCO_code/`和`CC_Compare/`第三方原代码保持只读；新代码只在扩展目录。
2. 当前入口`scripts/static_delivery/run_rrp.py`有audit/train/smoke/solve/fleet-smoke/fleet-solve。默认协议v3为`rrp_mip_dimensional_v1/source_day_domain`，9h总tour为另一个显式运行域；多车K必填，原33托盘容量不改。使用`python -B`，不得新增扩展`__init__.py`，用`scripts/project_paths.py`解析根目录。
3. 固定作者commit、原JSON/TXT和SHA。不得生成客户实例、伪造坐标或用无文献pilot填物理缺项。保留目标/profile/单位身份。
4. RRP主目标是年化牵引、制冷、工资欧元；距离另外报告。原动态“cost=纯距离”不适用于新RRP总成本。
5. 数据按几何分组防D1/D2泄漏；正式训练显式split。连续等待和有向路线均由独立评价器重算，不复制参考解时刻作为评价。
6. 全参考审计不表示所有成本已一致，不冒称LAHC/MIP/CP复现，不用功能smoke推断泛化或正收益。
7. 后端/GPU选择先于JAX导入。用户已授权SFTP服务器运行；长任务用独立tmux、明确资源和输出目录，不碰其他运行任务、旧最终种子或正式历史实验。
8. 取货/负结果关键材料在`archive/transition_20261020/`。任何清理先核验归档及当前SHA，拒绝穿透Windows reparse/junction；用户原private、原未跟踪4项、`.git`和凭据留原地。禁止git clean/reset、整仓git add、自动提交或推送。
9. 用户随后明确授权推送到GoodThing001/DynMaskCO，并同步版本化目录结构；本轮可显式提交并正常快进推送main/当前分支，不得force push或改写历史。凭据、private及大型本地归档/模型排除范围见GITHUB_SYNC.md。恢复包仍为历史证据，原始失败/NO-GO和已知缺失不改写。

## graphify

仅用户键入`/graphify`时使用`~/.Codex/skills/graphify/SKILL.md`；先读取该技能再按其执行。

旧完整说明原字节已保存在`archive/transition_20261020/pickup_negative_key_evidence.zip`的`AGENTS.md`成员，逐件SHA见MANIFEST.json。
