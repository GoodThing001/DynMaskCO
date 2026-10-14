# 配送任务训练独立审计与新完整掩码训练工程入口

本审计只读原 `delivery_traintask` 工件与源码，不改任何封存、当前在跑树或上游。机器复算见 `INDEPENDENT_TRAINTASK_AUDIT.json`，准确范围是存储成本的数值一致性与实现职责；没有重新生成原开发数据或推断原模型。

## 已建立的事实及需更正的解释

原开发面板 32 个唯一实例的均值独立复算为 trained 106.230388、zero-shot 90.139234、discard 和 regret2 均 80.730989。训练臂相对 regret2/discard 的平均相对成本改善为 −32.13549%，95% 配对区间 [−36.75392%, −27.56495%]；相对 zero-shot 为 −18.19143%，区间 [−21.50661%, −15.08456%]。三项均为 0 改善/32 更差，原数值失败保留。原固定 bootstrap 参数复算误差为 0。这不是新的独立确认。

`delivery_trained_guide.py` 的 `DeliveryTrainedGuide` 明确冻结原 encoder/decoder，仅优化双线性 edge head。`features()` 返回缓存 encoder 特征；`logits(features, adjacency)` 付费调用原 decoder 后丢弃结果，再返回仅由 features 得到的 head 分数。训练 head 不读取掩码邻接。因此它是静态节点表示上的插入引导头，不是完整 MaskCO encoder+decoder 掩码邻接重建训练。保存的 base 参数前后哈希相同、head 哈希改变，与该实现一致。

`train_run_v1/result.json` 的 505 是可用认证周期及保真检查周期数；实际优化只用 200 周期、5000 插入记录、100 epoch。保真检查为逐客户选项 12625 次，以及 80 周期前 6 步的完整规则共 480 步，不能写成 505 周期完整客户选择轨迹全部重放。

另有需要显式披露的配方/实现差别：recipe 描述 signed advantage 加权 CE；实际 `loss_fn` 使用 `score = delta/dscale − w*gain` 后普通 CE，`w=clip(delta_cost/scale, −1, 1)`。权重放在分数内部，改变 logits/梯度，不等同于正负权重乘 CE。部署则固定 `guide_weight=1`。本审计不在旧树修正后重训，不把配方文字与实现自动视作同一个科学方法。

`guide_diag` 的 `by_dist=min(scored, key=raw_delta)` 定义在相同当前客户/同一可行选项集合上，故 chosen_delta−min_delta ≥0 是数学恒真；0/400 更便宜不能证明长期完整路线无效，更不能关闭完整 masked 训练。正的额外即时增量与原开发损失方向一致，但不是全局不可能性证据。`min_delta_counterexample.py` 提供固定欧氏几何合成反例：第一步 delta 0.601183 的即时贪心最终路线 2.287803；第一步付 0.610962 的另一个合法插入最终路线 2.115732。该反例只推翻错误推论，没有训练或研究增益声明。

原 DEV `per_instance` 只有成本/部分资格标记，没有完整路线与执行轨迹；本审计不能据此独立重放完整合同硬可行和成本分解。跨主机重跑同一 DEV 属实现复现，不是全新确认。原 `infer_ident` 的两个结果已逐文件 SHA 记录；本审计没有独立重新加载正式 checkpoint，不能声称完成了新的实际权重复核。

## 文献与代码根据

[MaskCO 官方 ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf) §4.1–4.4 支持从完整解构造部分解、条件掩码预测及迭代重构的框架；其一般公式为 BCE。附录的 CVRP 教师与结构配置说明完整网络和教师质量的作用。该论文没有证明当前冷链合同或本项目插入头的收益。实际上游 `training/train_cvrp.py` 使用客户行的正边 log-softmax CE，当前工程入口跟随这一代码选择，明确区别一般 BCE；所有 encoder/decoder 参数进入优化器，没有 frozen base 或替代 head。论文中容量放松和测试 2-opt 的收益也不能直接移植成严格硬可行在线资格。

[Burges 等，Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) 支持以成对分数差构造可微排序损失。新方案的路线偏好 softplus 是借鉴这一形式的项目选择；平均边 log-probability 只是路线评分代理，不是真实路线概率，论文不保证其部署收益。权重 0.2、192 更新、batch 8、64/16 数据量、10 秒均为待审工程预算，不是论文给出的冷链最优参数。

[Beiranvand、Hare、Lucet 的优化算法比较综述](https://arxiv.org/abs/1709.08242) 支持明确问题、控制比较资源和报告复现边界；据此新方案保存所有路线/候选/超时，采用随机臂顺序、同信息和同合同。相同 10 秒上限不表示消耗完全相同；同 CPU 解码调用数也不自动表示全过程墙钟相等。

## 新完整训练版本与资格屏障

`training_engineering.py` 已实现真实 CVRPModel encoder+decoder 的完整参数训练；同初始化的 CE+偏好主臂与 CE-only 控制，以及 seed42 随机初始化同目标臂共享 3 固定步与全部 masks。每臂实际 encoder/decoder 参数哈希改变，固定最后步保存/重载预测一致，改变 partial adjacency 确实改变输出。5 项可移植测试含数据/ID 内容重叠拦截、完整路线约束、掩码子集以及即时 delta 反例。该模型为 1+1 层、16 维合成架构，不是正式 checkpoint 或全规模训练。CLI 只接受 synthetic 工程阶段；正式研究训练目前不能从这个入口意外启动。

`FULL_MASKED_TASK_PROTOCOL_DRAFT.json` 给出下一科学假设、完整网络、10 维可见特征提案、独立 TRAIN/teacher-diagnostic/DEV/确认、固定最终权重，以及同预算 ANY50/GLS 对照。研究种子和源码/合同冻结尚需新总控登记；真实教师完整合同重放、原权重输入投影扩展、真正完整保留边解码、强对照目标一致性仍未实现/未验收。尤其品质原生率与货物域不可默认为已物理校准，rate 字段没有合法合同定义时必须拒绝。不能将此草案或工程 PASS 写成完整研究训练合格、稳定增益或在线增益。

同初始化纯重建控制必须只删偏好项而保留相同 CE、数据、mask、优化器和日程；若主臂也只有完全相同 CE，则二者是同一方法，不能冒称两种消融。随机初始化同目标可区分预训练作用。原未训练预训练模型应单独报告，以判断微调是否改善；真实动态在线确认需要独立动态合同与合格强滚动面板，不能用 ANY50/GLS 静态快照替代。

## 可执行工程命令

```powershell
python research/review_3080_20261010/delivery/training_engineering.py --stage synthetic_engineering_only --upstream MASKCO_code --registration research/review_3080_20261010/delivery/FULL_MASKED_TRAINING_ENGINEERING_REGISTRATION.json --out research/review_3080_20261010/delivery/engineering_qa_NEW
python -m unittest discover -s research/review_3080_20261010/delivery -p test_training_engineering.py
```

执行端需先资源核验；Windows 3080 的 JAX CPU 后端不会因为显卡存在就变成 GPU 训练。以上工程命令不产生科学样本，不上传 MD，也不改旧配送结果。研究执行命令要在待测资格全部关闭并新 JSON 注册后另行形成。
