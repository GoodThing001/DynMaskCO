# 论文口径与多车扩展资料

文档版本：[全项目文档索引](../文档索引与版本状态.md)。本页为当前v3说明或已完成计划，历史研究身份另行标注。

用户授权按补查原件→论文优先统一时钟/公式/成本→多车扩展执行。

- [执行记录与当前验收/阻塞](执行记录.md)
- [论文逐式审计与未闭合项](paper_standard_audit.md)，[BibTeX](paper_standard_audit.bib)
- [两台服务器补查](server_recovery.md)
- [公开多车数据候选、单位和取舍](multi_public_data.md)
- [设计与实施计划](实施设计与计划.md)
- [当前使用说明](../static_delivery/README.md)

本地最终：68测试通过；实际三步小模型训练与重载、单/多车三臂求解通过；30参考主域均可行但成本仍有残差；30实例×两域60次Split检查可行。多车采用原RRP订单和同型33托盘车，不生成新实例。

服务器v3完整骨干run06及n34实际2车三臂run07均已通过，退出0，10份输出SHA已核验；旧审批阻塞已解决。[run06验收](../../results/static_delivery/followup_20261020/server_run06_verification.json)和[run07验收](../../results/static_delivery/followup_20261020/server_run07_verification.json)保存独立重算、实际调用和资源身份。历史run05仅属v2。

原论文式(20)与数值示例、MIP/CP传热及σ量纲冲突已显式裁定；未获取前作全文和初始渗透积分完整生成说明，继续标注缺口，不以新扩展抹去。
