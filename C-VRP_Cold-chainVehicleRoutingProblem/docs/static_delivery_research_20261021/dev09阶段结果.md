# dev09阶段结果与缺失验证

阶段更新：dev10已补齐9项并退出0，完整32份验证路线核验通过且封存为v1，见[完整结果](dev09_dev10完整结果.md)。以下保留补跑前阶段记录，不代表当前仍有缺失。

2026-10-21。本次为训练/验证开发实验，未使用正式外层测试。dev09四个模型均完成1440步，但整个任务触及预设2400秒总上限，退出124，验证完成23/32项。原始日志、106份下载报告及SHA、8份留在服务器的权重/优化器文件身份均保留。该任务不能标记为全部成功。

## 已独立核验的内容

- 72项科研冻结文件原SHA未变。
- 新旧教师各228份，共456份原100客户路线重新核验完整硬约束和净效益。
- 新旧教师各训练基础版、方向版，共四模型，每个1440步；重载预测与优化器恢复核验通过。
- 23份已保存验证路线均独立重算通过，目标与约束未改。

完整机器记录：[部分核验结果](../../results/static_delivery/algorithm_dev_20261021/remote/dev09/local_partial_training_study_verification.json)、[下载凭证](../../results/static_delivery/algorithm_dev_20261021/remote/dev09/download_verification.json)。

## 旧教师组：四例完整比较

c205—c208，种子1000，主品质条件和名义P，各方法目标10s。每种方法均在同一四例上求平均。

| 方法 | 平均净效益U | 平均实测搜索秒数 |
|---|---:|---:|
| MASKCO基础任务适配 | 276279.90 | 10.0067 |
| 方向改进D | 276090.42 | 10.0120 |
| ALNS框架适配 | 275986.51 | 10.0092 |
| 自编VNS | 275452.30 | 10.0039 |

基础版平均比ALNS高293.39 U；方向版比ALNS高103.91 U，但比基础版低189.48 U。当前没有方向模块优于基础版的证据。只一个开发种子、一个验证几何且共享服务器负载，不能认定统计显著或正式测试优胜。

与v0相比，同时变化了训练长度、采样顺序和推理实现，收益差异不能归因于某一单项；神经算法本轮使用compiled/highest/vectorized，教师影响应只在dev09内部配对考察。

## 升级教师组：尚不完整

c205四方法已完成；c206已保存基础版、方向版和VNS；c206的ALNS与c207/c208全部方法尚缺，共9项。不能把这些覆盖不同实例的均值放在同一排行榜。

仅对新旧教师都已有结果的c205/c206配对计算：基础版平均增加32.41 U，方向版增加179.01 U。两例结果不足以判断教师改进的完整效果。教师自身净效益提高不保证学生模型搜索收益提高。

## 补全计划

dev10独立目录已准备，仅补上述9项，复用dev09已经保存的1440步检查点及原始执行源码，不重训、不覆盖dev09的退出124与已保存路线。恢复入口拒绝重复结果，核验父任务源码/权重SHA，并把补全结果放在新目录。

首次启动被资源门槛阻止：两张5090利用率分别约94%和93%；这次检查没有启动任务，没有停止其他作业。2026-10-21 15:52（北京时间）复查后，空闲检查通过，已在5090 GPU1启动dev10（CPU核83/85/86/89，900秒上限）；原dev09使用GPU0，恢复使用同型号GPU1，需披露此差异。当前状态为**补验证运行中**，以后以[恢复计划](../../results/static_delivery/algorithm_dev_20261021/recovery_plan.json)及dev10的started.json/run.exit为准。自动化已更新为监控此恢复任务；当前应用保存周期为30分钟，先前文档的5分钟不再代表当前设置。

补全后核验命令：

```powershell
python -B C-VRP_Cold-chainVehicleRoutingProblem/tools/static_profit_remote.py pull --label dev10
python -B C-VRP_Cold-chainVehicleRoutingProblem/tools/verify_static_profit_training_study.py --label dev09 --recovery-label dev10
python -B C-VRP_Cold-chainVehicleRoutingProblem/tools/audit_and_seal_publication_protocol_v1.py
```

成功后追加完整版本，保留每项原始来源、原任务超时及恢复硬件/时刻。正式60s、单CPU对照、多种子和方法/检查点冻结仍待实施。
