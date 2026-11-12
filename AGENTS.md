# AGENTS.md

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。当前使用文档与历史快照的范围以索引为准。

## 当前任务与权威入口

用户于2026-10-21确认科研冻结并进入MASKCO算法改进：静态冷链、多车、所有车辆总净效益最大化，其余为硬约束；并授权RTX5090服务器实验。当前入口为`C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md`、扩展内`docs/static_delivery_research_20261021/算法实现交付与后续评估方案_v1.md`和`scripts/static_profit/README.md`。2026-10-22算法开发阶段收口，交付身份为MASKCO静态冷链任务适配，尚无已成立的独立改进方法；`算法阶段_v0.md`仅为历史设计。用户随后明确批准固定冷链MASKCO条件性重复评估v1，现行执行入口为扩展内`docs/static_delivery_research_20261021/固定冷链MASKCO条件性重复评估_v1.md`：仅12次训练与705次求解；基础fold0原12+一次性例外3累计预留15，不退款或重置；其他额度不变。新根种子2000+100*fold+10*repeat、shuffle根+1，搜索3000/3001/3002；用户随后明确批准239件实验包上传到原5090项目repeat01目录并允许共享GPU；资源仅要求显存/磁盘及单CPU准入，保留其他任务并记录共享负载；原独占/整卡空闲要求由追加资源修订单替代。远端预检通过后单次启动，主失败零自动重试。既有STOP、正式101—110/outer-test、提交和推送继续关闭。用户随后明确启用本聊天repeat01每15分钟监控：只检查既有主作业，结束后下载SHA核验及分析总汇报，报告一次后暂停；不重启或追加实验，主失败仍零自动重试。该次repeat01已结束：首训练1800秒保护触发，子退出124/主退出1，完成模型0/12、求解0/705，主要指标NOT_COMPLETE；结果与原日志已下载核验并汇总于扩展内`results/static_delivery/fixed_adapt_repeat_20261022/RESULTS.md`。已总汇报，监控暂停，无自动重试；后续11模型及705求解未启动，预算原登记保留。72项模型/数据/分析文件按FREEZE_LOCK封存，算法仅train/validation，达到预先分析计划门槛并另行锁定方法与检查点后才开正式测试。旧单车/多车RRP与动态取货均保留原身份，不覆盖当前目标。

用户随后明确批准repeat02显式恢复，现行执行登记为扩展内`docs/static_delivery_research_20261021/固定适配重复评估_repeat02_执行记录.md`及`configs/static_delivery/fixed_adapt_repeat02_execution_v1.json`。repeat01失败及预算原件不变；f0_r0重试追加1项例外，基础累计16/5/4/4，其余11模型沿用原预留。仅12目标模型与原705求解；每模型编译准备最多4CPU/1800秒、单CPU加载300秒、从0开始1440更新1800秒、保存导出300秒，主保护120000秒。复用诊断缓存并逐模型记录实际命中；所有模型完成锁定后才求解，任一阶段失败停止后续启动，零自动重试。共享GPU0、训练和求解CPU9，编译资源另报。正式测试、周期监控、提交和推送仍关闭。

用户于2026-10-23明确启用本聊天repeat02每15分钟监控（原automation更新，不新建重复任务），只检查既有作业；完成或失败后下载SHA核验、分析并总汇报，报告一次后暂停。正常无变化保持安静，主失败仍零自动重试，不启动新实验。该授权覆盖此前repeat02“周期监控关闭”的状态，正式测试、提交和推送仍关闭。

repeat02已于2026-10-23结束并完成回收汇总：fold0/1六模型各1440更新完成，fold2/3六模型及全部705求解未启动；f2_r0前显存准入5031<5120 MiB，主退出1，无自动重试。最终报告为扩展内`results/static_delivery/fixed_adapt_repeat02_20261022/RESULTS.md`，主要指标NOT_COMPLETE，356件输出/6根文件SHA通过、23源码身份一致；12个权重/优化器文件留远端。预算16/5/4/4不退款不重置，monitor automation已PAUSED；不得自动恢复或追加，正式测试、提交和推送关闭。

## 工程规则

1. `MASKCO_code/`与`CC_Compare/models/`中的第三方原代码保持只读。用户于2026-10-21授权CC_Compare按data/models/scripts/configs/docs/results/checkpoints重构，仅保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR；新增算法仍放扩展目录，CC_Compare/scripts仅作委托与核验入口。原comparisons/static_coldchain来源快照及冻结路径保持不变。
   用户随后于同日要求扩充至约10项并下载作者开源代码，现行范围增加AttentionModel、POMO、CaDA，共10项；原始MASKCO基础适配另作参照，不计入10项，也不把本项目改进版计作对比方法。当前清单以CC_Compare/configs/methods.json及SOURCE_PINS.json为准。新下载作者代码放models且原字节只读，适配器仍放扩展目录；旧七项迁移报告保留历史身份，不重写。
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


用户于2026-10-23明确批准repeat03显式续接：只读继承repeat02六份1440步检查点，仅训练fold2/3六模型，12份权重锁定后执行原705单元；预算16/5/4/4不变，新增配置0。现行登记为扩展内`docs/static_delivery_research_20261021/固定适配重复评估_repeat03_执行记录.md`和`configs/static_delivery/fixed_adapt_repeat03_execution_v1.json`。共享5090，显存准入5120 MiB及其他门槛不变；新模型/每折求解启动前资源不足按60秒排队，累计最多43200秒，超限WAITING_RESOURCE_LIMIT；排队单列，外层保护163200秒。阶段上限及原训练/求解/审计不变，运行失败零自动重试。已单次启动tmux `maskco_fixed_adapt_repeat03`，GPU0、单CPU18；只观察该作业，不重启repeat01/02，不补跑。周期监控继续PAUSED，正式测试、提交和推送关闭。

用户于2026-10-23明确开启repeat03周期监控：原heartbeat `automation`更新为ACTIVE，每15分钟只检查已启动repeat03，结束后下载SHA核验、分析并总汇报一次，写reported_once后暂停。不启动/重启/补跑，不降低资源门槛；排队状态正常等待。此授权覆盖此前repeat03周期监控PAUSED，预算16/5/4/4及正式测试、提交推送关闭不变。

用户于2026-10-23调整repeat03监控：不再每15或30分钟检查/汇报，原automation下一次定于北京时间2026-10-24 01:00；若仍正常运行，静默改为02:00再查，此后按预计完成时刻安排、间隔至少1小时。仅异常与最终交付通知，完成后暂停原automation。预算、资源准入、实验与零自动重试边界不变。该安排覆盖此前每15分钟周期监控。

用户最新要求：repeat03原automation改为每6小时检查一次，保持ACTIVE；覆盖此前每15分钟及预计完成时刻01:00/02:00安排。不额外缩短轮询间隔，正常推进和排队不通知，仅异常或最终总汇报通知，汇报后暂停。实验、预算、资源准入和零自动重试边界不变。

repeat03于北京时间2026-10-24 00:40:08正常结束，退出0；继承6模型、新训练6模型、共12个1440步检查点锁与705单元全部完成。独立审计PASS_FULL_705，重算60,128个有效incumbent；72冻结首尾一致。主要几何宏平均MASKCO−ALNS为−119.0165 U，基础259363.9246 U、ALNS259482.9411 U、原VNS259243.4269 U；47例均值对ALNS 4胜43负，对原VNS 39胜8负。完整报告为扩展内`results/static_delivery/fixed_adapt_repeat03_20261022/RESULTS.md`；2221件输出与6根原件SHA通过、21源码/计划/账本身份一致，自有工作器已退出。24件权重/优化器大文件（两批各12）留原远端路径，目录与SHA见checkpoint_catalog.json。本轮结束不追加实验，预算16/5/4/4不变；正式测试、提交和推送关闭。

repeat03总汇报已交付，reported_once标记已保存，原automation已PAUSED；不重复通知，不启动后续实验。

用户于2026-10-24仅授权的经济边际价值短名单阶段A已完成并交付，输出位于`research/paper_preparation_20261023/economic_shortlist_diagnostic_20261024/`。固定WM墙钟trained/best_only的16个10/50秒训练上下文；H遗憾2.009759 U，D 14.829065 U，E 0.786334 U；E−H遗憾下降1.223425 U、60.8742%，4/4例改善，数值门槛通过，仅值得提出阶段B。16次前向、20640位置、160 singleton、26799总路线请求，4740条路线级复算通过，最大绝对差4.093e-12；主进程24.164秒、shell27秒。主进程单CPU0/GPU1，外部nvidia-smi遥测未锁CPU亲和性，严格整作业树单核耗时不作主张。阶段A结束，不自动启动48单元、训练或增量实验；预算16/5/4/4、既有STOP、正式测试关闭、监控PAUSED和不提交推送保持。

用户于2026-10-24明确批准的一次经济短名单阶段B已完成并停止：专属b01，48/48、退出0，CPU3/共享GPU1、整自有作业树单CPU。fixed32 E−H +31.7086U；wall60 E−H −28.6301U且仅1/4实例改善；E−D +313.0635U。按原门槛STOP_PHASE_B_RULE，不自动补种子、改评分或追加性能实验。8610完整状态及361560次独立路线复算通过，最大绝对差5.457e-11；104输出SHA、10源码及7根文件身份一致，自有工作器退出。报告位于扩展内results/static_delivery/economic_shortlist_phaseB_20261024/b01/RESULTS.md。总主墙钟1701秒，登记至主结束1706.844秒，未新增训练或validation配置，基础16/5/4/4不变；锁定主方法及12终点不替换，108项、正式测试、周期监控、提交和推送保持关闭。阶段A动作收益保留原身份，阶段B的同时间负结果保留。
