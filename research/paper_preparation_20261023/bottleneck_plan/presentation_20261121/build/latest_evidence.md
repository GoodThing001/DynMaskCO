# 统一外部比较：图表数据与证据口径

更新日期：2026-11-21。配套 `latest_evidence.json` 保存原始数值精度；展示时可保留两位小数。这里只读取已完成实验和审计结果。

## 主结果

统一16法由锁版ORDER_SPLIT、12个外部方法及3项内部参照组成。62个不同案例分为30个已测复测案例、26个前瞻主案例、6个GH同母例压力案例；每例3种子，共2,976个终点，81,251条路线通过独立复算。前瞻主来源为10个荷比配送日、8个新XTW几何和8个商超城市。[论文实验总览](../../unified_external_20261120/论文实验总览.md) · [实例覆盖](../../unified_external_20261120/实例与方法覆盖.md)

| 前瞻来源 | MASKCO U | 外部排名/13 | 全部排名/16 | 最强外部法 | 对最强ΔU | 胜该法单位 | 热图ΔU |
|---|---:|---:|---:|---|---:|---:|---:|
| VRPPDL-p100 | 158,237.76 | 1 | 2 | ALNS | +46.17 | 10/10 | −0.07 |
| 新XTW-p100 | 91,133.09 | 1 | 1 | NDS-Econ | +135.44 | 8/8 | +35.29 |
| RWVRP-p100 | 833,636.86 | 1 | 1 | ALNS | +9.24 | 4/8 | +1.09 |

三个来源平均外部排名均第1；26/26个前瞻单位进入前三，其中19/26第一。三来源等权的ORDER_SPLIT−ALNS为+75.37 U，95%描述区间[43.76,109.44] U；相对同规则ORDER_SPLIT_D为+12.11 U，区间[−3.04,32.74] U。[三来源结果](../../unified_external_20261120/results/三来源前瞻结果.md)

`main_table`为16法×3前瞻来源的U与rank13/rank16；`primary`保留各来源完整配对差、区间、逐种子和逐几何结果；`forest_by_source`含对各自最强外部法及同规则距离的完整森林图数据；`equal_source`保留全部15个对照的来源等权统计。`campaigns`分别保留benchmark01、prospective01、retail01原分析对象，不能相互覆盖。

## 来源与发表身份

VRPPDL对应Bazelmans、ter Braake、Schrotenboer、van Lieshout和Van Woensel的 *The Vehicle Routing Problem with Pickups, Deliveries, and Scheduled Linehauls*，European Journal of Operational Research **330(3):995–1008，2026**，DOI [10.1016/j.ejor.2025.07.070](https://doi.org/10.1016/j.ejor.2025.07.070)。[作者机构记录](https://research.tue.nl/en/publications/the-vehicle-routing-problem-with-pickups-deliveries-and-scheduled/)列明正式发表2026-05-01、早期上线2025-08-23；作者仓库的2025引用属于早期年份。项目原件来自[作者公开实例](https://github.com/siemterbraake/vrppdl-algorithms-instances)，固定提交为7bbbd3e975eb77915aa255217af771e0fdb08e65。

RWVRP为萨格勒布大学FER–LARES的[作者公开基准](https://github.com/unizg-fer-lares/RWVRP_benchmark/tree/2c3a976f2212923db31ac86aea11483930c64b14)。[研究组官方新闻](https://www.lares.fer.hr/lares/about_lares/news?%40=33uop)确认对应 *A Multi-City Benchmark for Vehicle Routing in Real Urban Environments*，IEEE ITSC 2026，由Nikica Perić报告。完整作者表与DOI尚未核实，JSON留空。研究输入采用真实OpenStreetMap门店位置与作者建模需求、时窗，原库道路时间及交通模式未直接用于本统一冷链目标。

前瞻名单在成绩读取前固定。荷比为同运营商编号1的新订单日；新XTW与历史、封存及相互客户坐标重叠不超过10/100；GH未找到新的母客户几何，故第101–200客户作为同母例压力测试，单列。[新面板准入](../../unified_external_20261120/new_panel/准入.md) · [商超准入](../../unified_external_20261120/retail_source/准入.md)

`method_metadata`复用已核对的12法发表资料，`grade_policy`保留CCF2026和中科院2025大类口径。ALNS/Transportation Science与VNS/COR标2区*、PyVRP/IJOC标3区*，星号说明公开逐刊资料支持、官方原表未直接核实；TMLR的NDS与RouteFinder保留未核实。会议ICLR、ICML和NeurIPS均为CCF A。论文年份与等级目录年份分列。[既有等级核对](../../presentation_20261120/build/venue_rank_sources.md) · [近期五法核对](../../presentation_20261120/build/recent5_sources.md)

## 补充实验与JSON接口

| JSON字段 | 内容 | 原始出处 |
|---|---|---|
| `origin` | 8已测例、2种子、四臂64终点的直接消融；7/7几何高于原版；另存26例跨批桥接 | [原版与机制消融](../../unified_external_20261120/原始MASKCO与机制消融.md) |
| `trajectory` | 既有3来源各3几何、10种子、360终点；5/15/30/60秒1,440个实际接受车队 | [轨迹与首次请求](../../unified_external_20261120/results/trajectory01/RESULTS.md) |
| `startup` | r101两掩码六次首次请求测量；原61.45秒、缓存13.02秒；24开发终点首60秒交付3/6→6/6 | [编译复用开发结果](../../maskco_followup_dev_20261120/startup/RESULTS.md) |
| `bounds` | 62例统一经济目标上界；逐来源/方法差距、逐例界及3个五客枚举核对 | [上界结果](../../unified_external_20261120/bounds/RESULTS.md) |
| `sensitivity` | 旧sync01四法360终点固定车队、顺序、时刻；品质/里程/制冷单价±20% | [价格情景](../../unified_external_20261120/economic_sensitivity/RESULTS.md) |
| `cities`、`retail_heterogeneity` | 八城逐城效益、排名、ALNS和热图配对差；去除湾区后七城均值−0.62 U | [逐城结果](../../unified_external_20261120/results/retail01/CITY_RESULTS.md) |

## 科学叙述的实际差异

- 热图稳定门槛要求来源均值>1e−6 U、严格多数单位改善、至少2/3种子为正。仅新XTW通过，但其热图区间[−9.11,96.27] U仍跨零；门槛通过与区间排除零是两件事。
- 商超八城平均超过ALNS，但4胜4负，区间[−4.06,30.20] U跨零。旧金山湾区+78.25 U对均值影响较大；留出结果是描述性影响检查，主面板仍保留八城。
- 各来源的最强外部法来自实测来源均值选择，图中区间是配对描述区间，并未额外校正该选择。三枚搜索种子固定；按来源内几何、日或城整簇重采样20,000次，再给三来源等权。
- 62例不是62个全新独立空间来源。26前瞻例也不是FLEET_PRICE原先封存的32例；原32例继续封存。
- 原版四臂、十种子轨迹、缓存开发和价格敏感性不属于2,976个统一主终点，不计入26例前瞻验证。
- bounds中的既有XTW汇总13个不同案例；主表是10例、低重叠复核另10例且共享7例。两处XTW均值不能直接替换。
- 距乐观上界给出潜在改进空间的上限，并非已知全局最优误差；XTW-n716的全局最优净效益上界为−5,872.24 U，体现强制全覆盖条件下的损失最小化。
- 价格敏感性固定既有路线和时刻，没有按价格重新优化；图中极值横段不是置信区间。首次调用缓存保持热图数值一致，但仍为开发实现，主排名采用原锁版流程。
