# 全项目整理与双服务器同步（2026-10-21）

当前CC_Compare按data、models、scripts、configs、docs、results、checkpoints划分，models内保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。旧CC完整树及导航原件位于本地archive/cc_compare_reorganization_20261021，不重新发布。

本轮使用逐文件SHA清单同步3080与5090的原项目路径。旧服务器CC及零散脚本先核对SHA，再整体改名归档到服务器archive/server_sync_20261021；被覆盖的旧文件另存overwritten。每次移动的原路径、归档路径与逐文件SHA保存在服务器moves.json和inventories。历史research、results运行树、ckpts、logs及历史论文材料、private、凭据和Git保留。3080旧RRNCO三进程由用户授权停止；5090其他项目GPU任务保留。

本地当前净效益20项测试、历史RRP62项测试通过。全项目导航、Python语法、冻结72件、原始数据及旧恢复包另有既定核验。测试只检查接口与语义，不形成方法泛化或正收益结论，不开启正式外层测试。

GitHub发布代码、文档、配置、数据来源与结果文本。论文全文、9份候选检查点、运行权重/payload、7.75GiB本地恢复树和私有材料不发布。服务器额外获得候选检查点及冻结需要的两篇论文原件；从GitHub单独克隆不能自动完成需要全文原件的72项核验。

- [本轮核验汇总](verification.json)
- [当前净效益测试](static_profit_tests.log)
- [历史RRP测试](static_rrp_tests.log)
- [对比目录整理](../cc_compare_organization_20261021/README.md)
- [GitHub范围](../../../GITHUB_SYNC.md)

release_manifest.json、原始服务器审计/移动回执、传输ZIP、Git暂存与提交回执保留本地，不重复进入仓库。使用维护脚本workspace_release_20261021.py生成清单、扫描、部署与校验；workspace_remote_apply_20261021.py只接受项目内显式路径并拒绝穿透链接。


整理期间检测到未冻结算法文件profit_neural.py、profit_search.py及测试在整理期间有后续开发修改。因此早前对比目录整理工具的“19个算法文件完全未变”快照检查当前不再适用；原成功回执和原清单保留，不改成新的SHA。本轮使用最新代码重新测试和同步，以当前发布清单为准，FREEZE_LOCK的72件不变。


同期dev11诊断仍在独立research运行树执行；该树原位保护，实时run.log和未封存输出不加入本次静态发布，只发布启动/资源/源码身份等固定元数据。后续结果须独立核验后再发布，不能把当前进行中状态写成已完成实验。


代码仍有并行研究开发，因此本轮发布先按内容SHA建立不可变快照，测试、服务器部署及Git索引使用同一快照；冻结之后出现的工作区修改仍留原位，不能用当前磁盘的后续编辑覆盖已经审计的发布内容。capture时间和逐文件SHA保存在本地release_manifest.json，最终提交与服务器版本标记互相对应。
