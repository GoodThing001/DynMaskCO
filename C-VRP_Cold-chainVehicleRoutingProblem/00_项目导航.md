# 项目当前状态：静态冷链配送与多车扩展

文档版本：[全项目文档索引](docs/文档索引与版本状态.md)，区分当前v3与历史研究记录。

2026-10-20，按用户批准的公开单车RRP→多车路线实施。当前为订单预先已知、−20°C配送；每车只装分配货物，逐客卸货，空载回仓。原动态取货和负结果为历史证据。

[当前说明与命令](docs/static_delivery/README.md) · [参数公式](docs/static_delivery/参数与公式.md) · [本轮执行记录](docs/static_delivery_followup_20261020/执行记录.md) · [论文审计](docs/static_delivery_followup_20261020/paper_standard_audit.md) · [历史归档](archive/transition_20261020/README.md)

当前协议v3，默认`rrp_mip_dimensional_v1/source_day_domain`：论文式(20)系数2、MIP等待传热、表1的σ乘法。9h总tour为另一个显式运行域；论文内部差异与仍缺前作全文/积分表头都已标注，不能称全文所有公式完全一致。

本地68/68测试通过；实际小模型3步训练每步37组更新，权重/优化器重载预测完全一致；单车和多车三臂均可行。30参考在主一天域均完整可行，9h域28份；全部参考成本仍未一致。全部30原实例×两种时域的60次近邻路序Split检查可行，每域22例用1车、8例用2车。功能结果没有建立正收益或泛化。

多车独立时钟、装载和托盘位置，全局恰好一次，成本逐车求和。原订单和33托盘容量不变；K为显式实验上限，Split仅优化给定排列/等待下连续分割。不是作者原生多车基准复现，也没有学习分车头。

服务器run06已完成当前v3完整16+6层256维骨干的3步训练，每步237组参数更新，重载预测完全一致，单/多车小例三臂和全量扫描通过。run07只读加载该检查点，对原n34实例的三个方法均实际使用2车；逐车载量/全局覆盖/空载返仓与本地独立成本重算通过。两个run退出0，10份输出SHA核验通过；当前检查点为results/static_delivery/remote/run06/outputs/followup/train/checkpoint。旧run05仅属历史v2。

当前活动区为`scripts/static_delivery/`、`configs/static_delivery/`、`data/public_sources_20261020/`和`results/static_delivery/`。原61文件与两归档包SHA复核一致，19保护记录、用户原4项Git状态保持，上游只读。番茄/鱼品质和多温舱pilot未启用；旧取货/NO-GO及证据缺口不改写。最终核验见`results/static_delivery/followup_20261020/verification_final.json`。
