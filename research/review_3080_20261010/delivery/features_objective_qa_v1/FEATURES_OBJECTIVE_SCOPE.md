# 实际冷链字段与完整掩码目标工程资格

状态：仅工程，旧 TRAIN 首个完整教师样本复核，不生成新科学数据。实际 76M 原 checkpoint 经零扩展保留其余完整 encoder/decoder 权重，两个目标分别固定一次 SGD。CPU 外层上限 300 秒；未完成保留失败，不改旧结果重复挑数。

## 明确字段与单位

输入顺序固定为：可见坐标归一化 x/y，需求除容量，时间窗始末各除 8 小时 horizon，服务时间除 horizon，release/ready 各除 horizon，温区合同目标摄氏温度除 20，类别合同初始品质指标除 9。最后两项是合同 pilot 属性，不是实际舱温或商品货温，类别品质指标本身不可当跨商品统一物理量。

全部输入在 snapshot 决策时已 release 且 ready。原 CVRP 的全节点坐标归一化仅在这个全可见前提下使用。函数拒绝未来订单，而不是将特征清零后宣称 attention 已具备非预知性。8 小时、5 车、0.5 小时波次、20°C 和品质 9 的缩放均为项目工程选择，不由论文给出最优数值。

## 初始化与可执行控制

`init_proj/kernel` 从真实原权重 3×512 增至 10×512，前三行原字节保留，七新行零。其余 233 参数叶保持原身份。实际前向比较原 3D 与新 10D logits，允许浮点矩阵宽度变化误差至 1e-4。初始移除新增属性应保持 logits 等价；更新后记录属性效应和七行梯度，不能用零初始化的不敏感性宣称属性无用。

两臂从同一 mapped 原权重出发：完整 route 条件重建 CE，或同 CE 加 0.2×配对 logistic preference。route score 是 customer row 的正边平均 log-probability，教师标签来自同实例 incumbent 与候选的完整、硬可行 C0 距离/能耗/品质成本差符号；平局不产生偏好项。0.2 为新工程选择，无论文最优保证。训练边 CE 仍包含已知保留边；这与 upstream CVRP 的完整目标一致，未来科学协议应明确是否另测 masked-only loss，不能事后换目标。

每臂记录完整 encoder/decoder 梯度、参数更新、属性移除效应、保存重载 logits、完整 C0 投影路线/交付清单/事件、保留边。工程模型永久不可进入正式候选收益裁决。

## 文献支持与不支持

- [MaskCO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf) §4.1–4.4 支持 partial adjacency 条件完整重构和 encoder/decoder 框架；附录 CVRP 配置与实际 512、16/6 层核对。论文通用 BCE 表述不等于此 upstream CVRP 实际 customer-row 正边 CE；本包采用实际代码目标。该论文不验证我们的冷链字段或动态运营。
- [Burges et al., 2005, RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) 支持 pairwise logistic 排序形式；本项目将 route log-probability 与 C0 成本标签配对属于新的迁移假设，不提供收益保证。
- [Pillac et al., 2013](https://publications.polymtl.ca/13267/) 按信息随时间演化区分动态路由。全可见 dispatch snapshot 与真实在线合同不同，完整 masked QA 不补足动态释放、货物生命周期和强滚动对照资格。
- [Mercier et al., 2017](https://doi.org/10.1111/1541-4337.12269) 讨论冷链时间与温度对商品品质的作用；不能由综述或采用温区类别证明本项目已物理校准。当前 `QualityConfigV2.rate_per_hour` 为固定类别 rate，并未随实际温度轨迹变化，本工程保持旧 C0 教师语义用于归档复核，不冒充新的温度依赖品质模型。

新科学链必须在首次生成 64 TRAIN 之前单独注册数据/教师/诊断/开发确认身份、完整目标与固定训练日程、修正品质合同、ANY50 和适配 GLS 的同合同强对照、耗时与失败全保留条件。原最终留出仍禁用。

## 实际 attention 算子边界

已核 `TransformerLayer` 默认绑定 `MultiHeadAttention.attention_fn`。该实际函数删除传入 `softcap`、sequence-length 和 local-window 参数；不能将源码同名调用视为论文 kernel 一致。`CHEAP_FIXTURE_QA.json` 的真实小型 attention 数值检查显示 `softcap=None` 和 `softcap=1` 输出完全相同。本工程原 checkpoint 的 `softcap=None`，所以零扩展工程保持当前实际算子；它不证明带 finite softcap 的其他模型复现正确，也不证明原论文训练 kernel 与当前 CPU 替代实现完整等价。本包不修改上游。
