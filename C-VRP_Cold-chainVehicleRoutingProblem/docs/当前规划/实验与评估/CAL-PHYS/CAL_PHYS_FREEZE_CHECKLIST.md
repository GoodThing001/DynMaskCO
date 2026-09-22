# 路线 B：参数设定与最终评测检查表

> 2026-09-13 · 这是一份科研核对表，不是多级实验审批。
> 提交 E 的 P1–P6、仪器注册与 mandatory dynamic holdout 草案已被路线 B 替代，未执行部分无需补做。

## 1. 现在可以做

- [x] 路线 B 已由用户选择；不再等待实物信息或另一轮选路线确认。
- [x] 源码 D 字节基线 sealed；E 已提交，D/E 独立备份据执行人记录完成。
- [x] M0 teacher、M0 诊断、M1 代码和 DEV 迭代可以开始。
- [x] 助手可以立即检索公开资料、登记来源、提取参数，不需要新 seal。
- [ ] 完整 teacher 导出器、训练器和 M1 实现仍需后续实践，不因本表勾选而视为完成。

## 2. 并行完成仿真设定

- [ ] 逐参数区分 literature / derived / assumption / pilot，不编造页码、数值或文件 hash。
- [ ] 原始资料可用性、字段、单位、方程、适用域和许可说明已核实。
- [ ] 目标温度、热参数、品质速率、Q0 与阈值的组合有可解释定义。
- [ ] 坐标/时间/速度、power/energy 的换算一致。
- [ ] 文献零阶指标没有硬塞进当前一阶品质模型；制冷量没有误作输入功率。
- [ ] nominal 和敏感性设置依据公开，不以主模型赢幅挑参数。
- [ ] 合同变化后受影响的 service、scale、teacher/evaluator 回归已完成。

缺某项只限制相应最终参数主张，不禁止在已验证 pilot 下做模型开发。
不强制自有实验、每温区具体商品、仪器证书、独立复核人签名或新源码 bundle。

## 3. 最终论文运行前一次性固定

- [ ] 任务/数据生成与来源清楚；公开数据和合成字段区分。
- [ ] 最终 simulator contract、objective profile、主 nominal 与压力设置有版本。
- [ ] 所有主方法和核心消融用相同最终评价口径，旧 pilot 训练明确区分。
- [ ] 强基线和外部 learned 适配公平，目标/输入差异与预算公开。
- [ ] TRAIN/DEV/FINAL-TEST 无同源泄漏；历史已查看集不冒充新 TEST。
- [ ] 主比较、全部训练 seed、样本数、实例/seed 成组统计和失败处理已写定。
- [ ] checkpoint 选择和模型配置不依赖 TEST。
- [ ] 可以从版本化源码、配置和数据身份重现，不要求每轮开发生成全套 seal。

按正式评测实际风险保存统一 artifact；本目录的 JSON 是规划元数据，不是现有 runner 识别的执行合同。

## 4. 可选 O0-CC ORACLE-VAL

如果论文要独立确认 oracle 结论，再使用既有 archive-only 入口。
它仍需有效 manifest/root_seed、完整 split registry、contract/profile 身份、统计方案和 archive 校验。
本轮未改这些程序锁；不要拿本目录的文献分析 JSON 当正式 statistical_analysis_plan.json。
ORACLE-VAL 的结果不决定是否“允许写模型”；不得拿它调参或回灌 teacher。
