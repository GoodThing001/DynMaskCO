# 冷链结果分类与证据入口

本目录名沿用已有运行合同；当前主线为静态多车净效益。完整阶段身份以[项目当前状态](../../项目当前状态.md)和[文档版本索引](../../docs/文档索引与版本状态.md)为准。

| 类别 | 目录或文件 | 说明 |
|---|---|---|
| 当前算法开发 | [algorithm_dev_20261021](algorithm_dev_20261021/README.md) | MASKCO四变体、同目标ALNS/VNS、服务器运行、路线审计；训练/验证开发证据 |
| 科研协议预检 | [publication_protocol_v1_preflight_20261021](publication_protocol_v1_preflight_20261021/summary.json) | 原数据、可行证书、几何划分与排程检查 |
| 科研正式封存 | [publication_protocol_v1_scientific_freeze_20261021](publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json) | 72项模型/数据/分析规则，不移动或改写 |
| 早期净效益假设 | [net_benefit_scenario_pilot_20261020](net_benefit_scenario_pilot_20261020)、[smoke](net_benefit_scenario_smoke_20261020) | 旧价值假设的数值情景，不充当当前冻结模型的方法结果 |
| 经典路线预检 | [classic_solomon_preliminary_v2_20261020](classic_solomon_preliminary_v2_20261020/results.json) | 12份经典路由；[最初接口失败](classic_solomon_preliminary_20261020/results.json)保留 |
| 商品品质与价值证据 | [最终品质计算](quality_evidence_demo_final_20261020.json)、[价值单位核验](direct_value_evidence_20261020.json) | 保留被脚本/报告引用的稳定路径，不是路线算法收益 |
| 已替代的品质中间结果 | [history/quality_evidence_20261020](history/quality_evidence_20261020/README.md) | 两份中间版本已归类，原字节与历史身份保留 |
| 旧RRP v3验收 | [followup_20261020](followup_20261020/verification_final.json)、[remote](remote) | 68项本地测试、run06/run07及原目标/检查点身份 |
| 历史审计与探针 | [history](history)、[audit_source_v2](audit_source_v2)、[final_review_probes](final_review_probes)、[verification](verification) | 保留版本与原始结论 |
| 发布记录 | [git_sync_20261020](git_sync_20261020) | 原发布清单和SHA，旧路径是历史记录，不改写成新路径 |

实验结果、源码身份、日志、路线与检查点按同一运行目录归组；不要把不同目标、条件或阶段的结果合成一张方法优胜表。大型权重及payload保留范围见[发布说明](../../../GITHUB_SYNC.md)。
