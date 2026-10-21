# GitHub同步与本地文件范围

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。

目标仓库：[GoodThing001/DynMaskCO](https://github.com/GoodThing001/DynMaskCO)。用户已明确授权本轮提交和推送；沿用当前分支，向main及当前工作分支正常快进同步，不重写历史。

版本化结构包含当前代码、文档、配置、原RRP公开来源及61件清单、当前验收结果，以及本地清理产生的旧文件删除。原版MASKCO_code保持不变。此前CC原树与子模块保留规则对应当时发布；2026-10-21用户授权保留7项并按职责分类，旧CC完整树已本地归档，作者源码原字节不改。用户随后明确要求同步3080、5090并推送GitHub，本轮按逐文件清单显式暂存、提交和正常快进推送。

以下是有意保留的本地差异，不属于漏传：

| 本地文件 | GitHub中的处理 |
|---|---|
| .vscode服务器配置、private、研究执行密钥和用户原根results | 不上传；既有敏感研究残留从当前索引移除但原件留本地，不读取或打印其内容 |
| 3.782GB旧工作区恢复ZIP、298.8MB关键证据ZIP | 保留本地；仓库保存原SHA、逐路径清单和恢复说明，原Git提交历史也保留 |
| 检查点weights.npz、optimizer.npz、运行payload.zip | 保留本地；发布检查点配置/SHA、训练报告、运行日志、路线和独立验收记录 |
| 下载的论文全文与源全文TXT | 保留本地；文献引用、DOI、来源清单和自写核验报告发布，不能把数据许可空缺改写成授权 |
| 对照方法的.runtime安装环境、下载ZIP和文献获取缓存 | 留在本地；comparisons/static_coldchain中的合法第三方源码、LICENSE和适配脚本随目录同步，尚不属于68项RRP验收 |
| 临时测试目录、旧运行副本、未启用候选二进制 | 保留本地或既有恢复包，仓库不复制可重建大件 |

Git不保存空目录，因此“结构一致”指版本化工程目录和文件增删一致，不包含上述本地专属文件逐字节镜像。归档ZIP和模型文件没有伪装成可从GitHub直接下载；还原方法和SHA见[历史保全说明](C-VRP_Cold-chainVehicleRoutingProblem/archive/transition_20261020/README.md)。

原RRP文件按固定作者commit保留原字节和归属；作者仓库未单独声明数据再分发许可，不能由本次同步推出其他使用权。`.gitattributes`对当前来源/验收文件关闭换行转换，确保克隆后仍能核验其原SHA。

历史RRP代码与结果为协议v3的功能验收：68项测试、run06完整骨干3步、run07原n34实际两车三臂；全部原参考成本和作者算法尚未一致复现，未建立方法收益或泛化。当前研究已转入静态多车总净效益目标，权威入口为[项目当前状态](C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md)和[算法阶段](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_research_20261021/算法阶段_v0.md)，历史RRP身份不覆盖当前目标。

后续研究交付同时发布：20条文献证据、原Solomon和作者VRPTWQ各56份数据、7个官方仓库的8份源码快照及适配器、12条经典路由可行结果和独立9项检查。早期PyVRP接口失败记录另行保留。完整下载快照中的预编译二进制、权重和文献全文仍按上表排除，逐成员来源清单保留其原身份。番茄和鱼的30条品质计算与路由结果分开报告；价格等证据尚缺，未计算单次配送净利润，未改变RRP年化成本基准。入口见[本轮证据包](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_research_20261020/README.md)。


## 2026-10-21对比目录重构

当前[CC_Compare](CC_Compare/README.md)按data/models/scripts/configs/docs/results/checkpoints分类，保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。

旧约7.75GiB完整树、MAPT内嵌.git及未跟踪文件、逐文件恢复清单与导航原件备份仅本地保存。checkpoints二进制也不发布。根.git未移动；旧MAPT工作树路径移出活动目录，本轮显式同步审阅旧路径删除与新路径新增，不把恢复树重新加入。

[整理与恢复说明](C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)。历史结果、失败和原manifest不改写。

## 3080、5090同步范围

两台服务器使用原项目路径，按SHA清单部署代码、数据、配置、报告以及本地选定的9份候选检查点。服务器旧CC及零散启动脚本在各自的`C-VRP_Cold-chainVehicleRoutingProblem/archive/server_sync_20261021/`恢复目录保留；既有研究运行树、私有材料、Git与凭据不纳入覆盖清单。3080旧RRNCO任务经用户明确授权停止，5090其他项目GPU任务不受影响。

科研FREEZE_LOCK的72件在本地与服务器核对。GitHub有意不包含其中两篇下载论文全文，完整72件核验需要本地已获取的论文；论文来源与SHA仍保留，不把GitHub克隆误称为已包含全部原件。候选检查点只到服务器，GitHub保留来源/SHA说明。

本轮操作与核验见[同步报告](C-VRP_Cold-chainVehicleRoutingProblem/results/workspace_sync_20261021/README.md)。


本轮采用不可变内容快照完成测试、服务器部署和Git暂存。同期独立dev11诊断继续运行，其实时日志和未封存输出留原位；只发布固定启动/资源/源码身份。快照之后的并行开发修改保留在工作区，不能通过覆盖或重置消除。
