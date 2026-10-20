# GitHub同步与本地文件范围

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。

目标仓库：[GoodThing001/DynMaskCO](https://github.com/GoodThing001/DynMaskCO)。用户已明确授权本轮提交和推送；沿用当前分支，向main及当前工作分支正常快进同步，不重写历史。

版本化结构包含当前代码、文档、配置、原RRP公开来源及61件清单、当前验收结果，以及本地清理产生的旧文件删除。原版MASKCO_code和CC_Compare原代码保持不变，第三方子模块指针也不因本次同步改变。

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

代码与结果仍为协议v3的功能验收：68项测试、run06完整骨干3步、run07原n34实际两车三臂；全部原参考成本和作者算法尚未一致复现，未建立方法收益或泛化。

后续研究交付同时发布：20条文献证据、原Solomon和作者VRPTWQ各56份数据、7个官方仓库的8份源码快照及适配器、12条经典路由可行结果和独立9项检查。早期PyVRP接口失败记录另行保留。完整下载快照中的预编译二进制、权重和文献全文仍按上表排除，逐成员来源清单保留其原身份。番茄和鱼的30条品质计算与路由结果分开报告；价格等证据尚缺，未计算单次配送净利润，未改变RRP年化成本基准。入口见[本轮证据包](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_research_20261020/README.md)。
