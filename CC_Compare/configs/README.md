# 比较配置

- [methods.json](methods.json)：用户确认的7项及当前运行资格。
- [SOURCE_PINS.json](SOURCE_PINS.json)：6个官方源码快照的commit、来源与原件SHA。
- [publication_model_v1.json](publication_model_v1.json)：现行冻结模型的同字节镜像。
- [publication_experiment_freeze_v1.json](publication_experiment_freeze_v1.json)：现行分析计划的同字节镜像。
- [algorithm_candidate_v0.json](algorithm_candidate_v0.json)：开发候选，非最终方法锁定。
- [OMITTED_VENDOR_ASSETS.json](OMITTED_VENDOR_ASSETS.json)：未复制到活动树的原包数据、无关权重与二进制。

镜像不替代原FREEZE_LOCK中的权威路径；任何新方法或检查点须继续满足原分析计划的正式测试门槛。
