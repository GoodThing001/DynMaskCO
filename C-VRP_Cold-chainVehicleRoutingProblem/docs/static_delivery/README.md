# 静态冷链配送：单车RRP与同构多车扩展

文档版本：[全项目文档索引](../文档索引与版本状态.md)。本页为当前v3说明或已完成计划，历史研究身份另行标注。

后续用户明确以最大单次配送净效益研究番茄/冷藏鲻鱼，需论文佐证；最新[证据与初步结果](../static_delivery_research_20261020/README.md)、[静态对比方法源码](../../comparisons/static_coldchain/README.md)独立保存。本页仍描述已实现RRP基准及原年化目标，新目标尚未接入本页CLI，不能用旧结果声称商品净效益。

当前实现为静态、−20°C配送：订单预先已知，每车只装分配给自己的货，逐客卸货，空载返仓；交通和气候保留公开时段变化。已实现单车论文口径评价、多车Split和MaskCO巨型路序重构，尚未完成作者全部成本及LAHC/MIP/CP算法的一致复现。

原数据为Ceschia等的[Extending and Solving the Refrigerated Routing Problem](https://doi.org/10.3390/en13236214)，Energies 2020,13(23),6214；[作者仓库](https://bitbucket.org/sceschia/the-refrigerated-routing-problem/)固定commit `f5c5317a5ecd8fe64df779e79bd0397904bc0f9e`。30个JSON、30份TXT和README原字节/SHA不变。它是公开半真实气候/交通基准，并非企业订单实录；仓库没有单列数据再分发许可。

## 目录与当前结果

| 路径（相对扩展根） | 用途 |
|---|---|
| `scripts/static_delivery/` | 数据、独立评价、真实训练/重构、单/多车求解 |
| `configs/static_delivery/rrp_source.json` | v3来源、公式与时间域身份 |
| `data/public_sources_20261020/refrigerated_routing_rrp/` | 唯一启用的原订单与SHA |
| `docs/static_delivery_followup_20261020/` | 服务器补查、论文逐式审计、多车数据核验、执行记录 |
| `results/static_delivery/followup_20261020/` | 本轮测试、30参考审计、训练、单/多车三臂和60项分车扫描 |
| `docs/static_delivery_migration_20261020/` | 原批准设计和迁移历史 |
| `archive/transition_20261020/` | 取货/负结果关键证据和恢复清单 |

## 唯一主标准

默认`--profile rrp_mip_dimensional_v1 --time-domain source_day_domain`。采用式(20)往返系数2、固定出库、服务前等待、出发槽整弧定速、服务开始槽热学、MIP式(14)全程等待传热，以及表1的0.30 L/kWh乘法。论文的数值例/公式、MIP/CP不完全一致，所以此标准是披露选择的独立实现，不代表逐字复现全文。逐项依据见[参数公式](参数与公式.md)和[论文审计](../static_delivery_followup_20261020/paper_standard_audit.md)。

`source_day_domain`要求返仓早于86400秒，另外报告9h超限；`operational_tour_9h`显式增加总tour≤9h。驾驶、服务、等待和总tour分列，不把驾驶9h文字当作已经证实的总tour约束。`paper`为主公式旧别名；两个archive profile仅作诊断，不根据哪个成本更近来选口径。

参考等待按论文服务前步骤统一为`[0,*TXT_raw[:-1]]`；Table7(b)独立确认13/15份D2，n9/n34保留论文/仓库快照冲突，不拼成作者最优解。D1-n9原8个零值原样保存，完整保存钟的全部等式唯一推出9个零等待，另记**派生标签**；没有找到新原件或修改TXT。保存钟诊断映射与主映射分开，候选钟均独立重算。

## 运行

以下在仓库根运行；输出目录必须不存在，以保留旧结果。CPU默认，后端选择早于JAX导入。

```powershell
python -B C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py audit --out C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/audit_new

# 单车真实训练、重载及共同目标三臂功能检查
python -B C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py smoke --steps 3 --model-profile smoke_cpu --out C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/smoke_new

# 多车，并附加单车审计/求解及30实例×两种时域的有限Split扫描
python -B C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py fleet-smoke --steps 3 --model-profile smoke_cpu --max-vehicles 2 --out C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fleet_smoke_new

# 推理不读取参考解；车数上限必填，原33托盘容量不改
python -B C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py fleet-solve --case D1-n34-np33 --max-vehicles 2 --methods local_search --iterations 3 --seconds 15 --out C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fleet_solve_new
```

`solve/fleet-solve --methods local_search maskco_initial maskco --checkpoint <目录>`比较局部搜索、同架构同初始化种子的随机权重和已训练模型。起点均为同一距离初解与零等待；前向/编译、候选准备、初始Split、DP、独立接受计入搜索预算，加载准备另报。每个学习臂保存实际配置与参数SHA，失败候选保留路线/等待/原因；超时不发布不完整分车为可行。

`--objective with_wage/without_wage`分别对应表7(a)/(b)；D2参考成本为无工资。多车扫描统一显式使用有工资目标，没有拿D2原无工资成本作同目标比较。主成本为各车牵引、传热、渗透和工资的年化欧元，距离另报。

`train --split <JSON> --steps <N>`要求显式train/validation/test名单，并按矩阵/需求SHA分组，D1/D2同几何不得跨角色。教师不符合目标/profile/时域会写明拒绝并终止，不能默默过滤。30例规模很小，尚无正式泛化主表。

`smoke_cpu`为1+1层32维测试模型；`maskco_256`为16层encoder、6层decoder、256维8头。服务器run06已完成当前v3完整骨干3步训练，每步237组参数更新，保存权重/优化器后预测逐值相同，单/多车小例三臂和全量扫描通过；9份输出SHA核验通过。当前检查点为`results/static_delivery/remote/run06/outputs/followup/train/checkpoint`。run07只读加载此检查点，以原n34实例补做三臂实际2车检查，退出0、输出SHA及本地独立重算通过。历史run05为v2，不可用于当前v3。

## 多车文献与范围

方法依据[Prins2004](https://doi.org/10.1016/S0305-0548(03)00158-8)的巨型排列+Split及[Vidal2016](https://doi.org/10.1016/j.cor.2015.11.012)有限车队背景。一般分层DAG/Bellman DP在已给分段成本后为O(Kn²)，分段评价时间另计；只求给定排列和节点等待下最优连续分割，不是连续等待全局最优，也不是Vidal线性算法。

各车独立计钟、装载、卸货位置和返仓；客户全局恰好一次，逐车≤33托盘。K是显式实验上限，报告实际车数。原RRP没有车队数和固定启用费，当前不增加该成本分项，也不声称真实固定费为零。模型预测巨型路序/等待，保留边属于巨型排列；Split可以切断车辆弧，模型没有独立分车头。

原总需求24–33托盘，容量本身不强迫多车；不复制订单、不缩小容量、不伪造冷链单位。其他候选及缺口见[多车数据核验](../static_delivery_followup_20261020/multi_public_data.md)。本轮是原单车公开订单上的**方法扩展**，不是原作者多车基准复现或更大企业数据。

## 参数和验证边界

物理值取原论文/JSON；未启用番茄/鱼品质、可售阈值或多温舱pilot，候选和缺项见[产品/车辆核验](../static_delivery_migration_20261020/参数论文核验.md)。

模型继承[MaskCO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)的partial adjacency、timestep及迭代掩码重构；架构和seed888取只读上游。学习率1e−3、weight decay1e−2取原训练脚本；[AdamW](https://arxiv.org/abs/1711.05101)替代Muon是改动。有向头、气候/速度条件、log1p等待回归和等权合并是本项目设计，须后续消融，不能伪造论文出处。3步、15秒、保留率0.2只作可覆盖的功能预算。

本轮30参考在主一天域均完整可行，显式9h域28份；不代表所有成本一致。60项近邻路序Split检查均可行，两种时域各22例用1车、8例用2车。三步训练三臂未建立正收益或泛化，局部搜索在小例更便宜。逐例残差、失败、旧审计保留，历史取货/NO-GO裁决不变。

## 服务器当前验收（run06/run07）

[run06机器记录](../../results/static_delivery/followup_20261020/server_run06_verification.json)核验完整骨干、237组实际更新、重载、单/多车小例和60项扫描；[run07机器记录](../../results/static_delivery/followup_20261020/server_run07_verification.json)核验原D1-n34的三臂实际2车。后者每臂上限45秒/3轮，实际工作量如下，不据此作正式同计算量效果主表。

| 方法 | 实际车辆 | 年化成本€/年 | 模型前向 | Split调用 | 实际秒数 |
|---|---:|---:|---:|---:|---:|
| local_search | 2 | 110554.447290 | 0 | 95 | 45.001 |
| maskco_initial | 2 | 111344.982695 | 3 | 4 | 7.567 |
| maskco | 2 | 111344.982695 | 3 | 4 | 2.230 |

模型只受过单一4节点实例3步监督；本表说明流程可执行，且本次模型未优于局部搜索。它不证明泛化或冷链多车优化收益。全部客户覆盖、逐车载量/空载返仓和所有成本分项在本地再次独立重算通过；原参考成本残差不因此消失。
