# 配置分类与适用范围

配置集中在[static_delivery](static_delivery)，目录名沿用原路径以保护来源、运行记录和冻结SHA。

| 文件 | 身份 |
|---|---|
| [publication_model_v1.json](static_delivery/publication_model_v1.json) | 当前已冻结的静态多车品质净效益模型，原100客户、20吨、虚拟U |
| [publication_experiment_freeze_v1.json](static_delivery/publication_experiment_freeze_v1.json) | 已冻结的数据分组、条件、统计与正式执行门槛 |
| [algorithm_candidate_v0.json](static_delivery/algorithm_candidate_v0.json) | 算法开发候选，不能当最终方法冻结 |
| [research_acceptance_20261021.json](static_delivery/research_acceptance_20261021.json) | 用户确认的任务与执行范围 |
| [rrp_source.json](static_delivery/rrp_source.json) | 旧RRP v3/33托盘配置，不用于当前利润单位U的训练 |
| [product_evidence_v1.json](static_delivery/product_evidence_v1.json) | 早期商品品质文献计算配置 |
| [direct_value_evidence_v2.json](static_delivery/direct_value_evidence_v2.json) | 同商品价值/单位核验，非当前路线算法主配置 |
| [net_benefit_scenario_v1.json](static_delivery/net_benefit_scenario_v1.json) | 早期数值假设情景，保留原身份 |

冻结文件保持字节与位置；禁止为方便分类改名或改参数。[科研冻结清单](../results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json)。
