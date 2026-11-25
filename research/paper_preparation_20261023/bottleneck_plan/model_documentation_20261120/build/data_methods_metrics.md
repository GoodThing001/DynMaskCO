# 数据、比较方法与结果指标说明

## 一、问题名称与数据单位

### 1. VRP、CVRP、VRPTW 与 VRPTWQ 的关系

VRP 是 Vehicle Routing Problem，即车辆路径问题：决定哪些客户由同一车辆服务、每辆车按什么顺序访问客户，以及何时出发和服务。CVRP 在车辆路径问题中明确加入容量约束；VRPTW 强调客户时间窗，并通常同时限制车辆容量；VRPTWQ 在时间窗车辆路径问题中进一步约束货物交付品质。VRPTWQ 作者仓库将 Q 明确解释为 Quality constraints。它与容量通常使用的符号 Q 在语义上不同，阅读公式时应按上下文区分。[S01][S02]

本项目研究单仓、同质冷藏车、静态订单、全部客户必须服务的闭合配送。静态表示求解前已知本次全部客户、需求和时窗，不涉及行驶过程中动态到单；多车表示车队可以包含多条非空路线。每个客户只由一条路线服务，禁止拆分交付；每辆车只承担一趟，并返回出发仓库。共同数学上界 K≤100 来自最多100条非空路线，不代表研究观察到一个实际拥有100辆车的企业车队。[S02，21–34行]

本项目最终排序指标为全车队净效益 U，越高越好。经典 Solomon/GH 托管页面通常采用“先最少车辆、再最短总距离”的层级目标。两者的目标、品质约束与单位不同，因此本项目的 U 和车辆数不能直接与原网页的最优距离或最佳已知解排名对照。保留原始数据出处，是为了明确输入来源；这里实际测试的是经过固定映射的冷链配送问题。[S03][S04][S02，60–90行]

### 2. 五类硬约束与时间语义

容量约束是每条路线的需求质量之和不超过20,000 kg，即20 t。时间窗约束服务开始时刻处于客户的允许窗口；服务完成可以晚于该客户窗口右端，但仍须满足品质及回仓期限。品质在服务完成时评价，qᵢ≥0.8；覆盖要求每个客户恰好服务一次；回仓要求所有车辆在原仓库允许期限内返回。求解器输出路线后，还必须求出与这些约束一致的出发、等待和服务时刻。[S02，21–58行]

给定一条有向客户顺序，评价器采用最晚可行出库时刻，再向前递推服务时刻。推迟出发可能减少途中等待，使交付货龄与制冷车时不增加。加入一个新客户时，原路线的最晚出发时刻可能前移，故必须重算整条路线中所有客户的品质，不能仅检查新加入的客户。这里的等待是连续时间，不把时刻离散成整分钟或沿用参考解时刻。[S05，32–48行]

主条件采用275 K恒温、零阶品质衰减，qᵢ=1−k(T)(Cᵢ−Dᵣ)，其中 Cᵢ 为服务完成时刻、Dᵣ 为所属车辆出库时刻。k=0.0001848 min⁻¹，因此qᵢ≥0.8对应最大交付货龄约1,082.25分钟。货龄包含车辆出发后的行驶、等待和该客户完整服务时间；客户收货之后的返仓不继续扣减其品质。品质保持率无量纲，需求质量仍按原kg计量，品质下降不等于车上货物质量减少。[S05，25–48行]

单位统一为时间min、距离km、质量kg、温度K和虚拟净效益单位U。参考货值P=5/3 U/kg，里程费率c_d=1 U/km，制冷费率c_f=0.0972 U/(车·min)。U按每公里基准运输费定义，是固定经济情景的计量尺度；P来自有文献依据的货值与里程费用相对比例，不是人民币与欧元的汇率换算。[S02，5行、60–82行；S05，50–71行]

## 二、四类数据来源及固定选取

### 1. VRPTWQ–Solomon：学习和历史开发的数据底座

数据来自Selin Hülagü作者仓库，关联论文为Hülagü、Ciancio、Dabia和Dullaert的 *An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products*，发表于Transportation Science，2026，60(5):675–695，DOI为10.1287/trsc.2025.0262。DOI字符串中含2025，不代表正式发表年份。更早的空间和时窗底座来自Solomon 1987年的车辆路径与调度实例。[S06，5行；S07，38–39行]

作者公开56个实例，每例为100客户加1仓库，字段为ID、X、Y、DEMAND、TW_START、TW_END、SERVICE_TIME。R表示随机分布，C表示聚类分布，RC表示两者混合；类型1与2对应较紧和较宽的时间窗。作者已经完成坐标、需求、时窗和服务时间转换；本项目读取该作者版本中的km、kg、min，不对这一来源再次缩放。客户服务时长90 min，仓库原窗0–1440 min。[S01][S05，9–19行]

在共同速度、容量、服务时长、时间窗和品质规则下，56例中47例满足每个必送客户至少能够独车服务的准入条件。排除的9例为c101–c104、c201–c204及rc205；这是本项目共同约束下的不可行判定，不等于原始Solomon问题无解。不同方法使用同一47例集合，不能按某个算法的求解成绩选择是否保留实例。[S05，21行、94–98行]

47例实际只有4组不同坐标几何：C1、C2、R1+R2、RC1+RC2。同一几何的多个时窗变体不应被当成独立空间样本。当前外部结果使用固定f0_r0模型：fold0的训练集合是R和RC，共38例、2个几何；验证为C2的4例；原折外测试为C1的5例。f0_r0的训练种子2000、掩码种子2000、打乱种子2001，共完成1,440次更新。[S05，79–88行；S08，2–48行]

### 2. 丹麦PDPCDPG-D100：易腐品订单的配送阶段投影

直接来源是Emmanouil Nisyrios在Mendeley Data发布的 *Dynamic PDPCDPG with Rolling-Horizon*，V2，2026-07-14，DOI为10.17632/93w8ynrk59.2。PDPCDPG所指问题包含易腐品取送、交叉转运和时窗；该归档关联的研究题目为 *The dynamic pickup and delivery problem for perishable goods with crossdock and time-windows using an event-based rolling-horizon approach*。其业务数据底座是Wen等发表于JORS 2009的 *Vehicle Routing with Cross-Docking*。直接使用的是2026 V2原件，2009是底座论文年代；关联动态论文的正式期刊卷期在当前材料中未核实。[S06，6、14行；S07，43–46行]

名称中的D100是项目的“100个配送请求”标识。研究从交叉转运仓完成装载后的配送阶段开始，只取请求的配送端点、配送窗口与托盘量，不重现原任务的取货、跨仓同步和滚动到单过程。主面板使用Data200_a至Data200_e五个200请求母实例，每个母实例取相应旧100请求集合之外的全部100请求，形成dk_complement_a至e五个案例、五个几何。按原请求身份匹配补集，保留原行序；共享位置的不同请求仍分别服务。[S09，5、17行；S10，88行；S11，14–30行]

坐标从米除以1,000变为公里。需求按mᵢ=20,000×palletsᵢ/33映射为kg，故保留每个订单占源车辆容量的比例。原仓库工作时段06:00–22:00保留为360–1320 min；源配送时窗不改。配送阶段的时间递推没有另外添加客户服务时间，本项目对应服务时长0 min；转运仓每托盘处理时间属于另一流程。这里的20 t是统一实验容量映射，不是丹麦原件中的实测吨位。[S09，5行；S11，21–29行]

丹麦主面板平均单客需求占车容量30.9%，最近客户距离中位数2.9 km；XTW和GH相应占比约4.9%和5.7%。这使丹麦更容易受到跨车容量组合限制。原丹麦200请求母实例还给定67辆车和550分钟单路线时长；当前共同模型采用K≤100及原仓时窗，未把550分钟作为额外硬约束。历史ORDER_SPLIT的15个丹麦终点中14个满足所有路线≤550分钟，最长路线565.8分钟。该字段是源任务差异说明，不能把这些结果宣称为原PDPCDPG完整问题的可行最优解。[S12，17–19、34、41行；S10，65–72行]

### 3. XTW-p100：多种布局的带时窗扩展基准

XTW由Uchoa、Malheiros、Queiroga、Martinelli和Subramanian在CVRPLIB发布。正式生成说明 *Generation of the XTW and XLTW Instances* 日期为2026-10-31，XTW发布为2026-11-02。名称可理解为X系列加Time Windows；X系列的坐标和需求底座来自Uchoa等EJOR 2017的 *New benchmark instances for the Capacitated Vehicle Routing Problem*。2017与2026分别标识底座和时窗发布，不应互换。[S06，7、12行；S07，40–41行]

主面板按预定顺序使用n351、n359、n367、n376、n384、n393、n401、n411、n420、n429，共10例。n351等是原实例的节点规模标识，包含仓库；p100表示本项目保留该原件的真实仓库及原编号前100客户。原件规模更大，不意味着实验求解了351或429个客户。所有主比较仍是100客户。[S11，12、31–46行；S06，7行]

XTW和GH采用相同的固定尺度投影：令s=1440/b₀，其中b₀是源仓库窗口右端；X、Y、客户时窗和服务时长均乘s，需求乘20,000/Q_source。坐标与时间一同缩放，因而在共同速度1 km/min下保留源距离与行驶时间的相对关系；需求保留占源车容量的比例。该映射产生统一km、min、kg实验场景，不应把投影公里值当作原坐标天然具有的真实道路公里数。[S11，44–45行；S13，60–65行]

低重叠XTW面板保留主面板7例，用n439、n459、n480替换n359、n384、n420。每例去仓库后，与既有历史XTW客户位置并集的重叠最多8/100。它检验较少已见客户位置下的表现，不意味着与主面板独立的另一整组10例：两面板共享7例。因此27例主面板和10例低重叠面板合并只有30个不同案例，不能相加成37例。[S06，10行；S14，3、30行；S15，3–5行]

### 4. GH800-p100：Gehring–Homberger的100客户投影

GH是Gehring–Homberger名称缩写，来源为SINTEF托管的扩展VRPTW基准，原规模分200、400、600、800及1,000客户。关联经典论文为Homberger和Gehring的INFOR 1999论文，以及EJOR 2005论文。当前主面板从800客户组中取C1、C2、R1、R2、RC1、RC2六类各前两例，再取各自原仓库与前100客户，因此是12例GH800-p100，而不是求解12个完整800客户问题。[S04][S06，8行]

以gh_c1_8_1_p100为例，gh标识来源，c1标识空间/时窗类别，8表示800客户原组，最后的1表示组内实例序号，p100表示100客户投影。分组最终依据实际投影坐标而非字符串类别：12个案例经分组只有5个几何。统一变换同XTW；时窗和服务时长仍由对应作者原件决定，不能因为都叫“100客户”就认为与VRPTWQ或丹麦的配送难度相同。[S06，8行；S11，40–46行；S12，17–19行]

## 三、案例、几何、种子与数据角色

**案例/实例（case/instance）** 是一份完整的求解输入，包含仓库、客户坐标、需求、时窗、服务时长以及固定业务参数。**几何（geometry）** 是用于分组的空间结构身份；同一客户位置布局可配不同时间窗，产生多个案例，但这些案例仍共享一个几何。报告先在几何内部平均，目的是避免拥有更多时窗变体的布局获得更高权重。元数据中的长字符串是几何分组标识，不是效益指标。[S05，79–88行；S16，110–115行]

**搜索种子（seed）** 控制客户移除、抽样、首客遍历等求解随机性。最新主面板固定33001、33002、33003，每个案例运行三个种子。三个种子用于描述同一案例的搜索波动，不把空间样本数扩大三倍。**区组（block）** 指一个案例与一个搜索种子的组合；同区组各方法共用初解，sync01还要求在同一物理CPU核上轮换方法顺序。**终点（endpoint/cell）** 则是一个区组中的一个方法完成本次预算后的最终结果，包含路线、效益和计时记录。[S17，13–17行；S15，3–5行；S16，57–104行]

**fold（折）** 是整组几何的训练、验证、测试角色划分；**repeat（重复）** 是同一划分中的独立训练重复；**checkpoint（检查点）** 是某个训练时刻保存的参数状态。f0_r0就是第0折、第0次训练重复；其1,440表示参数更新数，不是推理搜索的轮数，也不是训练使用1,440个独立案例。当前外部比较始终使用这一固定检查点，不会为丹麦、XTW和GH分别选一份成绩更高的权重。[S08，2–48行；S18，111行]

训练数据用于调整网络参数，验证/开发数据用于选择方法和超参数，测试/外部确认数据用于评价已冻结的版本。名称中的reserved_validation是特定历史阶段的预留面板身份：reserve02开封时提供锁版后的新请求/新组合证据；结果公布并用于后续分析后，这些案例已是已测数据。recent5和sync01复用它们作同期扩展与重复评价，不能再次称为全新独立测试。原另外预留的32例仍未运行；最新同期比较没有把它们纳入样本。[S10，3、88行；S18，111行；S19，7行]

主面板包括丹麦5例/5几何、XTW10例/10几何、GH12例/5几何，共27例/20几何。低重叠XTW单列，不参与三个主来源等权的重复计数。这里“外部”表示来源不同于固定模型训练底座；“低重叠”表示位置重复受控；“独立”还取决于是否在选版前锁定、是否已看过结果，三者不是同义词。[S15，3–5、53行；S18，3、70、111行]

## 四、18种方法分别是什么

### 1. 比较身份和发表载体

18法包含6个本项目方法/消融名称与12个外部方法适配。内部方法没有独立发表期刊，不把基础MaskCO的ICLR等级直接填成ORDER_SPLIT自己的发表等级。会议等级统一采用CCF2026第七版；期刊采用中科院2025升级版大类。表中“2区*”“3区*”是公开逐刊资料一致报告、但官方逐刊原表未直核的结果；TMLR分区未核实，不借用JMLR或会议等级。[S07，32、51–72行；S20；S21]

| 方法名称 | 本项目身份与关键机制 | 原方法发表期刊/会议、年份 | 等级口径 |
|---|---|---|---|
| ORDER_SPLIT / MASKCO | 锁版完整方法：掩码边预测、多候选经济重插、访问顺序搜索与经济分车 | 本项目实现；基础MaskCO发表于ICLR 2026 | 本项目无独立发表等级；基础MaskCO为CCF A |
| ORDER_SPLIT_D | 完整方法的同规则纯距离消融 | 本项目实现 | 不适用 |
| HYBRID | 较早锁版：距离/原热图/校准热图与两类插位算子试选 | 本项目实现 | 不适用 |
| HYBRID_D | HYBRID的同规则纯距离消融 | 本项目实现 | 不适用 |
| DIST | 负边距/最短绕行候选的原距离重构基线 | 本项目实现 | 不适用 |
| DIST_SLOT | 距离重构加最短可行绕行位置保护 | 本项目实现 | 不适用 |
| ALNS | 自适应大邻域搜索；使用项目冷链破坏修复算子 | Transportation Science，2006 | 中科院工程技术2区*（2025） |
| VNS | 系统切换邻域的冷链净效益搜索 | Computers & Operations Research，1997 | 中科院工程技术2区*（2025） |
| NDS-Econ | 原NDS学习移除集合，项目经济贪心重插 | TMLR，2025 | 中科院分区未核实（2025） |
| AttentionModel / AM | 注意力网络逐步构造客户序列 | ICLR，2019 | CCF A（2026） |
| POMO | 多起点策略优化及构造 | NeurIPS，2020 | CCF A（2026） |
| DeepACO | 神经启发信息增强蚁群搜索 | NeurIPS，2023 | CCF A（2026） |
| CaR | Construct-and-Refine，构造与轻量约束修复 | ICLR，2026 | CCF A（2026） |
| PyVRP-HGS / PyVRP | 0.11.3版混合遗传搜索、路线交叉和局部搜索 | INFORMS Journal on Computing，2024 | 中科院计算机科学3区*（2025） |
| MVMoE | 多任务混合专家车辆路径网络 | ICML，2024 | CCF A（2026） |
| CaDA | 约束感知局部/全局双注意力 | ICML，2025 | CCF A（2026） |
| RouteFinder | 多任务RF-Transformer路径基础模型 | TMLR，2025 | 中科院分区未核实（2025） |
| RRNCO | 同时读取位置、距离和行驶时间的RRNet | ICLR，2026 | CCF A（2026） |

表格的方法出处与项目身份见[S07，9–32行]、[S21，7–23行]；当前18法名单见[S18，22–41行]。ALNS的2006论文是算法思想来源，实际通用软件另引Wouda和Lan的JOSS 2023论文。PyVRP的2024是软件求解器论文年份，HGS本身有更早的算法研究。CaR使用ICLR2026论文对应实现，与前作PIP的NeurIPS2024身份分开。RouteFinder按本次登记的TMLR2025版本引用。[S07，9–20行；S21，13–17行]

### 2. 六个项目内部方法之间的区别

ORDER_SPLIT是完整组合方法。普通重插阶段先从当前车队遮蔽客户，让冻结MASKCO预测边分数；距离、原始热图和校准热图各配贪心与regret-2重插，形成六种组合，每种从相同状态试选0.04秒，再按实际冷链净效益选择组合继续推进。跨车步骤每三轮触发，选择2–4条相近完整路线，总客户数不超过24；掩码补齐至40客后预测，生成访问序列，对可行连续段用真实路线净效益做动态规划分车，单组预算≤0.30秒。固定顺序的分车最优不等于全问题的全局最优，访问顺序仍由外层搜索提出。[S22，35–37行；S23，19–27行；S07，29–32行]

ORDER_SPLIT_D保持掩码、重构、分车和总60秒限时，只将神经排序信息替换为距离。其普通重插只需试距离下的贪心与regret-2，节约的预测和试选时间可以继续搜索。因此完整方法减此消融衡量的是热图信息连同计算机会成本的净作用，而不是强行固定相同候选数量后的纯打分准确率差。sync01中距离版沿用包含权重加载的同一初始化入口，但在线搜索不执行神经前向，故准备时间仍可能与神经版接近。[S24，373行；S15，49、53行]

HYBRID提供普通重插的六组合经济试选机制，没有ORDER_SPLIT新增的每三轮跨路线顺序分车步骤；HYBRID_D是其同规则距离版本。HYBRID与HYBRID_D比较回答“较早重插机制中的热图增量”；ORDER_SPLIT与ORDER_SPLIT_D比较回答“加入访问顺序与经济分车后的热图增量”。两组差额再相减，才是新增机制是否扩大热图净贡献的交互式比较。[S22，37行；S23，27行；S25，22–28行]

DIST使用负边距作为排序信息，按插入后增加的里程选取前8个候选位置；候选仍交给严格冷链评价，并非以距离代替业务目标。DIST_SLOT在此基础上加入SLOT的最短可行绕行保护：若候选短名单漏掉该位置，就用它替换末位。DIST与DIST_SLOT属于较早实现，重构和候选规则并不等同完整ORDER_SPLIT_D。因此，完整方法优于DIST或DIST_SLOT的全部差额不能直接解释成热图收益。[S26，7–14行；S27，3、17行]

### 3. 传统搜索与学习破坏搜索

ALNS即Adaptive Large Neighborhood Search。它反复移除部分客户，再用修复算子放回，根据算子产生的改进调整使用倾向。原论文为Ropke和Pisinger的 *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows*。本项目使用公开ALNS框架和冷链全净效益破坏/修复算子，保持业务目标与独立评价一致；它是强传统搜索参照，而非原取送论文全部实验的复刻。[S07，9–10行]

VNS即Variable Neighborhood Search，来自Mladenović和Hansen 1997年同名论文。其思想是系统改变局部搜索邻域，使搜索从某个邻域的局部最优继续转移。本项目的VNS为冷链净效益实现。方法表中该论文说明搜索思想的来源，不能把本项目所有操作细节都归到原文。[S07，11行]

NDS-Econ来自Hottung、Wong-Chung和Tierney的 *Neural Deconstruction Search for Vehicle Routing Problems*。网络学习选择移除哪些客户，随后由重建器恢复可行解。本项目复用作者VRPTW100 epoch2000权重，每次选择15个客户，每轮单核抽5个掩码，按学习的移除顺序执行全插位严格经济贪心重建，并采用退火接受。后缀Econ表示经济目标适配，其学习发生在破坏/移除阶段，与MASKCO从残缺路线预测客户连接的学习位置不同。[S07，12、18行；S27，13行]

### 4. 已有四个神经构造/搜索参照

AM的正式题目为 *Attention, Learn to Solve Routing Problems!*，作者Kool、van Hoof和Welling。注意力编码器表示客户，解码器根据当前路径状态依次选择下一个客户；其原方法采用REINFORCE与贪心rollout基线。POMO的正式题目为 *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*，通过多个起点和多种等价最优表示训练和搜索，降低对单一起点的依赖。[S07，15–16行]

DeepACO的正式题目为 *DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization*。网络给出供蚁群使用的启发信息，蚂蚁构造候选并结合蚁群机制继续搜索。CaR的正式题目为 *Towards Efficient Constraint Handling in Neural Solvers for Routing Problems*，将学习式构造与轻量可行性改进联合考虑。它们的原结构分别强调搜索启发和约束处理，而本项目对各自输出统一施加冷链硬约束并用U选择最终车队。[S07，13–14行]

AM、POMO、DeepACO和CaR在本项目使用各自冻结epoch50权重，各完成11,400次冷链适配更新；MASKCO固定权重完成1,440次更新。更新次数不是跨网络训练费用的等价尺度：模型大小、训练步骤工作量与目标不同。这里统一的是在线60秒求解预算，而不是离线训练总成本。[S07，20行；S24，362行]

### 5. 新增五个近期参照

PyVRP采用作者0.11.3版混合遗传搜索，包含种群、选择性路线交换交叉、惩罚管理和默认局部搜索算子。共同初解作为初始种群中的一个个体，其他个体按软件原规则初始化；它不需要神经网络训练和前向。内部目标使用里程、制冷与品质损失的可加下界代理；每个原生可行子代再用完整冷链模型评估，只有60秒内严格改善的车队才能刷新最终纪录。因此主表U是完整目标，HGS内层的代理目标不等于完整非线性冷链目标。[S28，3–9行]

MVMoE的正式题目为 *MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts*。本次采用作者100客户、四专家模型及多起点贪心解码，输入需求和时窗，并在每步检查本项目冷链可行性。混合专家指不同专家模块按输入状态参与处理，不是对五种算法成绩取最优。[S21，8行；S18，107行]

CaDA的正式题目为 *CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention*。网络通过约束提示和局部、全局双注意力表达不同路径约束，本次采用作者100客户多任务权重，容量与时窗提示进入输入，配合八方向等距视图、多起点构造和冷链动作掩码。[S21，9行；S18，106行；S29，6–12行]

RouteFinder的正式题目为 *RouteFinder: Towards Foundation Models for Vehicle Routing Problems*。本次采用作者100客户多任务RF-Transformer，由RL4CO解码器执行多起点构造；需求、时窗、服务时长进入部署模型，逐步屏蔽冷链不可行动作，使用八方向等距视图。RouteFinder与CaDA共同采用按实例跨度规范化的坐标和对应时间尺度，需求按容量归一化；该网络输入归一化与数据映射是不同层次，最终评价仍回到固定物理单位。[S21，10、17行；S29，8–12行]

RRNCO的正式题目为 *RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization*。本次使用作者RCVRPTW权重和RRNet编解码器，同时输入坐标、距离、行驶时间和时窗，按可行动作多起点构造。本面板行驶时间仍由距离和固定车速推算，因此没有额外实测道路拥堵或非对称交通时间信息。[S21，11、15行；S18，108行]

这四项近期神经方法均使用作者公开权重直接迁移，没有进行本轮冷链重训练；MASKCO使用已有冷链适配权重。神经构造方法从首客开始构造新车队，共同初解提供可行保底；MASKCO从共同初解开展反复重构；PyVRP把共同初解加入种群。这种初解使用方式应与成绩一起解释，不能把全部方法描述成完全相同的搜索轨迹。[S18，105–111行；S29，8–12行]

## 五、如何读净效益、差额与统计结果

### 1. U、收益分解与相对百分比

车队净效益定义为U=ΣᵢmᵢPqᵢ−c_dΣᵣDᵣ−c_fΣᵣτᵣ，其中Dᵣ为包含回仓的路线总里程，τᵣ为出发至回仓的总车时。品质价值是交付货物的质量加权价值，不是只把满足阈值的客户算作“成功”；即使两个解均满足q≥0.8，较早交付高货量客户仍可能保留更多价值。制冷费用包含行驶、等待、服务和空载返仓，故减少里程不必然提高U。[S02，38、60–82行]

总满品质价值V₀=PΣᵢmᵢ对同一个案例是常数。质量损失价值为L_q=ΣᵢmᵢP(1−qᵢ)，等价损失费用L=V₀−U=L_q+C_d+C_f。因此最大化U与最小化L在同一案例下等价。报告中的`quality_loss_value_U`越小越好，`net_benefit_U`越大越好；品质损失只是价值分解，不能在已经使用ΣmPq的收入之后再扣一次相同货损。[S02，66–82行；S05，71行]

配对差额ΔU=U_MASKCO−U_对照。正值表示MASKCO更高，负值表示对照更高。差额可以按品质价值改善、里程成本节省和制冷成本节省分解，三个分量之和等于ΔU。例如“里程成本节省”用对照成本减MASKCO成本，因此正值表示费用下降；这个符号方向与“MASKCO里程减对照里程”相反。[S10，59–63行]

相对改善百分比应明确分母。主报告中“相对ALNS的U（%）”表示100×(U_MASKCO−U_ALNS)/U_ALNS；它是相对于该来源汇总净效益的比例，不是订单命中率、品质改善百分比或相比理论最优的gap。若平均的是逐例相对百分比，应另行标明；不能默认其等于先汇总U后相除。由于满品质货值通常远大于可优化差额，几十U的净增量可能仅占总净效益的百分之零点几或更低，解释时应同时保留绝对ΔU。[S10，41–45行；S37，56行；S02，60–82行]

### 2. 三层等权平均

记Uₘ,c,s为方法m在案例c、种子s的结果。先对同例三个种子取均值，得到案例均值；再对一个几何下的所有案例等权平均，得到几何均值；最后对来源内几何等权平均。以GH为例，其12个案例属于5个几何，汇总时每个几何各占1/5，而不是让拥有较多时窗变体的几何获得较大权重。[S15，5行；S16，110–150行]

三个主来源等权是再将丹麦、XTW、GH三个来源均值各赋1/3。它不等于把20个几何全部等权：在来源等权下，丹麦的每个几何占总体1/15，XTW每个几何占1/30，GH每个几何占1/15。这个设计保持每类来源具有相同影响，不让拥有10个几何的XTW压过各有5个几何的另外两类来源。来源均值在同一单位U下比较，但三个来源的绝对货值规模仍不同。[S16，151–159行]

“胜/平/负几何”在每个几何先完成案例和种子汇总，再判断配对ΔU的正负；容差为1×10⁻⁶ U。17/20表示20个主面板几何中17个的平均差额为正，不表示81个案例种子单元中胜17个。跨来源汇总的胜几何数将各来源几何计数相加，是计数描述；与来源等权效益使用不同的权重结构。[S16，139–158行]

“正向种子数”则固定某枚种子，在来源内对几何汇总差额，再判断是否为正。它回答搜索随机性是否改变总体差额方向，不能代替几何胜率。最差几何是几何均值中最小的ΔU，最差单块则是某个案例/种子配对最小的ΔU；两者用于显示收益是否分散及是否存在局部退步。[S10，19–23行]

### 3. 95%描述区间、排名与门槛

当前区间采用几何整簇有放回重采样20,000次。每次抽取一个几何时，保留其内部案例和三枚固定种子的整体结构；不会把同几何时窗变体和种子拆成互相独立的观测。来源区间由该来源重采样均值的2.5%与97.5%分位数给出；三来源等权区间每轮分别在三个来源内重采样，再平均三个来源的结果。[S18，70行；S16，118–154行]

“描述区间”表示当前已测面板的几何间变异。它不包含重新训练不同检查点、换机器、重新选择超参数和从真实世界抽取无限新地区的全部不确定性；因此不能把区间直接解释为所有冷链业务上必胜的95%保证。区间跨零表示当前来源的方向尚不稳定；总体区间为正也不意味着每个来源区间都为正。例如sync01的总体热图增量区间为[5.19,49.00] U，但丹麦、XTW和低重叠XTW的各自区间均跨零。[S15，18、22–36、53行]

名次按同一来源汇总U从高到低排序，容差内平局采用并列组末位。例如两个方法并列最高时，两者均记第2，而非都记第1。同期四法、同期六法和分批十八法是不同名次分母；四法第1不自动推出十八法第1。后者还包含历史批次方法，不具备同一时段全部重跑的含义。[S15，14行；S18，18–43行]

历史预登记联合门槛是逐来源同时检验热图相对各距离参照为正、足够多几何与种子改善、并达到排名要求。它与“来源等权平均正值”是两个不同的问题。reserve02的联合门槛未通过，而平均热图增量为正；sync01是已测案例的同期重复评价，不会追溯改写历史门槛结论。[S10，3、17行；S23，37行；S15，53行]

## 六、原始回执和CSV字段词典

### 1. 输入、终点与经济字段

| 字段 | 读取方式 |
|---|---|
| `batch` | 记录来自哪个执行批次，如reserve02、benchmark01或sync01；决定同期关系。 |
| `case`、`dataset`、`geometry` | 分别为完整案例名、来源名和几何分组标识。 |
| `seed`、`method` | 搜索种子及方法；与case一起确定一个方法终点。 |
| `primary`、`low_overlap` | 是否属于27例主面板/10例低重叠面板；共享7例可同时为1。 |
| `used_in_18` | recent5扩展十八法是否采用该行；当前批MASKCO替代历史MASKCO，避免同方法重复。 |
| `U` / `net_benefit_U` | 最终严格可行车队的净效益。 |
| `initial_U`、`initial_routes` | 本区组共同初解的净效益与有向路线。 |
| `gain_over_initial_U` | `U−initial_U`，表示在共同保底初解之上的收益。 |
| `final_at_initial_U` | 终点与初解净效益在1×10⁻⁶ U内相等；不必然表示路线完全相同。 |
| `final_is_initial_routes` | 原始路线列表是否与初解完全相同；与仅效益相等分开记录。 |
| `routes`、`vehicles` / `used_vehicles` | 最终客户访问序列和非空路线数量。仓库回路由评价器补入；客户序列有方向。 |
| `quality_loss_value_U` | 满品质价值与实际品质价值之差，越小越好。 |
| `distance_km` | 全部路线总距离，包含返回仓库；当前c_d=1时数值等于里程费用U。 |
| `refrigeration_cost_U` | 全部车辆总在途分钟数乘固定制冷费率。 |
| `equivalent_loss_cost_U` | 货币品质损失+里程费用+制冷费用，等于满品质总价值减U。 |
| `minimum_quality` | 终点所有客户的最低交付保持率，必须达到本条件阈值。 |
| `selected_s` | 最终被选车队完成接受/复算时距在线起点的时刻；不是总运行时长。 |
| `status`、`role` | 如AUDITED、external_test表示回执状态与当时数据角色；历史角色不能替代当前已测/未测身份判断。 |

字段提取和面板标记见[S30，179–202行]与[S16，163–174行]，经济分解见[S02，60–82行]。用车数是目标解的结构结果；本目标没有按车固定启用费，不能直接把“少用一辆车”换算成报告中未配置的工资或车辆采购节省。[S02，79–82行]

### 2. 在线工作量与机制字段

| 字段 | 读取方式 |
|---|---|
| `epochs` | 在线重构时段/轮次数，不是神经训练epoch。 |
| `forwards`、`forward_s` | MASKCO实际网络前向次数及该实现记录的前向累计用时；一次前向可供多次重插使用。 |
| `ranking_counts` | 普通重插最终选择距离、原始热图、校准热图的时段计数，顺序按回执定义。 |
| `regret_epochs` | 选择regret-2算子的时段数，与排序类型是交叉维度，不是第四种排序。 |
| `mask_mode_counts` | 不同掩码抽取模式被使用的计数，属于搜索机制记录。 |
| `iterations` | 原生重构迭代计数；不能与其他求解器的完整车队数直接互换。 |
| `improvements` | 内核记录的重构改进事件计数，可能包含试选分支中的改进；最终纪录以严格接受记录为准。 |
| `native_route_requests` | MASKCO原生内核路线评价请求数，包含相应分车调用计数；请求不等于不同路线数。 |
| `task_route_requests` / `requests` | Python路线缓存的命中加未命中请求；不同方法与原生请求的记账层不同。 |
| `cache_misses`、缓存命中 | 未命中表示需要计算的查询，命中表示复用已缓存的路线结果；重复请求会增大请求数而不增加独特路线数。 |
| `split_blocks` | 尝试跨车顺序与经济分车的路线组次数。 |
| `split_orders` | 在这些组内尝试的访问序列总数，多个序列可能得到相同分车。 |
| `split_route_requests` | 分车阶段的路线评价请求数。 |
| `split_offers` | 分车步骤直接提高当时历史最佳纪录的提交次数，不等于分车总次数。 |
| `archive_mips`、`archive_improvements`、`archive_s` | 历史可选路线库组合分支的调用、改进及用时字段；当前锁版未启用时为0，不能据字段存在声称执行了该机制。 |
| `decoded_fleets` | 神经构造器生成的完整车队数，不是每一步神经决策数。 |
| `learned_proposals` | 对应适配器记录的学习候选数；部分构造器与完整车队数相同，仍需按该适配器记录粒度解释。 |
| `forward_calls` | 相应神经适配器记录的网络/解码调用数；与MASKCO一次预测整张边图的粒度不同。 |
| `offspring` | PyVRP作者HGS产生的子代计数，属于遗传搜索工作量。 |
| `native_feasible` | PyVRP在截止前被原生模型判为可行的子代数，之后再接受完整冷链检查。 |
| `strict_feasible` | 适配器记录的通过完整冷链检查的候选数；此字段按具体适配器定义，不是跨法统一采样量。 |
| `audited_in_time` | 完整候选在在线限时内完成严格评价的数目。 |
| `accepted_records` / `accepted` / `accepted_improvements` | 对应实现记录的接受/改进数量；不能仅凭计数推断最终收益大小。 |
| `source` | 某些适配器记录的最终候选产生来源；是机制标签，不是独立数据来源。 |

这些计数的来源见[S31，137–254行]、[S30，179–215行]、[S18，90–99行]、[S32，90–104行]、[S38，62–67行]与[S39，173行]。统计量只在记账口径一致时比较。recent5报告将45,510个RouteFinder完整车队、223,433个PyVRP子代与240次MASKCO分车改善分别展示，它们回答不同操作的工作量，不能用其比值宣称某法搜索效率高若干倍。[S18，90–99行]

热图使用比例按完成试选的普通重插时段统计，包含既定轮换处理的平局。历史丹麦的原始热图33.5%与校准热图27.7%相加约61.2%；XTW约66.4%，GH约71.7%。这说明在线过程实际采用过热图信息，不表示这些时段必然带来相同比例的最终收益。跨路线分车另外使用其对应臂的边分数，未包含在该普通重插比例内。[S25，14–20行]

### 3. 时间与资源字段

| 字段 | 读取方式 |
|---|---|
| `budget_s` | 统一在线预算，本轮60秒；解在截止前完成严格评价才可入选。 |
| `setup_s` | 各适配器已有口径的搜索前准备时间，包括相关加载/准备；不是所有方法完全同构的代码段。 |
| `initialization_s` | 在线计时内的初解、实例或评价初始化分段，由计时器记录。 |
| `search_finished_s` | 主搜索结束时距在线起点的时刻。 |
| `closing_audit_s` | 最后收尾审计所用时间；它可使记录的结束墙钟略大于60秒。 |
| `total_measured_s` / `measured_search_wall_s` / `measured_wall_s` | 实际记录的在线墙钟；MASKCO取搜索时钟总计，其余适配器可能计至额外终点评价结束。 |
| `cpu_s` / `search_cpu_s` | 对应求解阶段进程消耗的CPU时间；单核时常接近墙钟，但资源等待和记录边界会导致差异。 |
| `worker_execute_s` | 工作器从进入方法执行入口到返回结果的墙钟，包含该入口覆盖的准备和求解。 |
| `controller_process_s` | 控制器观察的独立进程总墙钟，含进程启动、准备、搜索、输出等。 |
| `search_overrun_s` | 搜索计时器记录的超时量；与迟到候选是否被接收分别核对。 |
| `rejected_late_candidates` | 因未在截止前完成严格评价而拒绝的候选数。 |
| `core` / `cpu` | 本次求解使用的物理CPU核身份/亲和集合；编号不是线程数量。 |
| `*_available`、`*_mean`、`*_total` | 对应字段有值的终点数、完整可用数据均值与累加量；缺失不按0处理。 |
| 核秒配额 | 终点数×每终点单核预算，例如360×60=21,600核秒。 |
| 控制器墙钟 | 多核并行批次从启动到结束的实际历时，不能与所有方法CPU秒直接相加或比较。 |

时间字段映射见[S16，163–188行]、[S33，15–32行]、[S34，62–73行]和[S18，88–99行]。sync01每个终点仍单核执行，整个批次使用24个独立物理核并行区组，GPU禁用。方法之间“相同60秒”指模型就绪后的在线解选择时限，不是每次独立进程都在相同总时间内结束。[S19，3–7行；S15，49行]

sync01中MASKCO准备均值31.883秒、进程总墙钟均值93.087秒；距离消融32.043秒和93.257秒；ALNS为0.742秒和61.410秒；PyVRP为0.111秒和60.530秒。recent5中MASKCO准备均值64.9秒、PyVRP0.22秒来自另一批运行。不同批次准备时间和短试选截断可能改变搜索轨迹，不能把两批均值拼成同一次端到端计时结论。[S15，42–49行；S18，43、88行]

### 4. 审计与完整性字段

`independent_audit`或`independently_audited=true`表示最终入选路线经过独立复算；复算检查覆盖、容量、合法排程、品质、回仓和金额，而不是直接相信求解器回传的U。`endpoints`是通过的终点数，`routes`是这些车队中被重放的路线总数；同一路线可出现在不同终点中，因此不是互不重复的路线集合规模。`maximum_absolute_U_error`记录原回执U与独立重算U的最大绝对差；`maximum_initial_U_error`相应检查共同初解。[S16，51–106行]

sync01审计为360个终点、7,951条路线，最大终点U误差约4.66×10⁻¹⁰ U、最大初解误差约2.33×10⁻¹⁰ U。PASS表示已报告结果在该模型下计算闭合；它不是全局最优证书，也不单独证明对照训练预算相同、全部来源独立或热图稳定泛化。[S35；S15，53行]

## 七、三个主要结果批次怎样同时使用

| 批次 | 主要问题 | 已完成规模 | 使用位置 |
|---|---|---|---|
| reserve02 | 锁版完整方法与原13法的预留验证 | 27例×3种子×13法=1,053终点；30,899条路线复算 | 历史完整阵容、机制记录、原消融和原预登记门槛 |
| recent5 / benchmark01 | 锁版方法与五项近期对照的同期比较 | 30个不同案例、90区组、6法、540终点；12,239条路线复算 | 近期五法主比较；与12个历史方法组成分批十八法表 |
| sync01 | 锁版、同规则距离、ALNS、PyVRP的同期复核 | 30个不同案例、90区组、4法、360终点；7,951条路线复算 | 同期热图净增量与两个强搜索对照的最新证据 |

规模与身份见[S10，3行]、[S18，3、20行]、[S36]、[S15，3–5行]。表中30个不同案例均由27例主面板加3例低重叠补充构成；低重叠表包含的另7例复用同一批已测结果。[S15，5行]

reserve02的丹麦、XTW、GH完整13法名次为2、1、1；相对ALNS分别为+187.92、+81.35、+43.01 U，相对同规则距离分别为−0.95、+29.35、+30.22 U。它支持完整方法在三来源具备竞争力，同时保留了丹麦热图净增量不正、联合门槛未通过的事实。历史低重叠XTW的同规则热图增量为+3.42 U、改善5/10几何，区间跨零。[S10，3–17行；S14，10行]

recent5中MASKCO在三个主来源的同期六法均为第1，最接近的近期对照为PyVRP。三来源等权领先+2,661.58 U，95%描述区间[1,791.18,3,717.68] U，改善17/20几何。分批十八法主来源均第1，低重叠XTW为第2；这张十八法表将本轮六法与12个历史方法合并，并非18种方法全部同期运行。四个近期神经方法是作者权重直接迁移，因此成绩描述这些具体部署与冷链适配的表现。[S18，5–14、18–43、111行]

sync01对四个方法重做同案例、同种子、同物理核轮换比较。MASKCO在丹麦、XTW、GH和低重叠XTW的同期四法均第1。三个主来源等权相对同规则距离、ALNS、PyVRP分别为+25.17、+88.75、+2,616.84 U；对应改善几何为14/20、16/20、17/20。热图总体区间[5.19,49.00] U支持当前已测面板上的正向净增量，但单来源区间仍需分别解释。[S15，9–36行]

三批次可以共同支持“固定MASKCO掩码重构与经济分车在已测静态冷链面板上具备竞争力”；热图的贡献以同规则距离和相应置乱消融为依据，整套方法减PyVRP、ALNS或四项神经构造器的全部差额不能都归因于热图。结果更新时保留批次标识、相同目标与实际训练身份，不以最新一次较高成绩替换历史同名方法原始回执。[S15，53行；S18，43、111行；S24，373–375行]

## 来源索引

以下本地定位均相对于当前项目证据，链接给出可打开的绝对路径与起始行；正文的行号范围进一步限定对应事实。

- [S01] [VRPTWQ作者数据说明](https://github.com/SelinHulagu/vrptwq-solomon-instances)，Definitions、Instances、File format与Transformations。
- [S02] [固定业务模型配置](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:1)，1–99行。
- [S03] [SINTEF Solomon基准目标说明](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/)。
- [S04] [SINTEF Gehring–Homberger基准说明](https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/)。
- [S05] [组合模型与数据尺度](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_research_20261021/组合模型与数据尺度_v1.md:1)，9–109行。
- [S06] [已核对数据来源](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/presentation_20261120/build/dataset_sources.md:3)，3–16行。
- [S07] [既有方法与数据文献](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/presentation_20261120/references_academic.md:7)，7–72行。
- [S08] [f0_r0训练计划](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fixed_adapt_repeat02_20261022/remote/repeat02/outputs/panel/training/f0_r0/training_plan.json:2)，2–48行。
- [S09] [丹麦来源研究设计](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/研究设计.md:4)，4–18行。
- [S10] [reserve02完整报告](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/RESULTS.md:1)，1–94行。
- [S11] [主面板实际映射代码](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/prepare_reserved.py:12)，12–46行。
- [S12] [来源输入对照](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/来源输入对照.md:3)，3–43行。
- [S13] [跨数据集投影代码](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_cross_dataset_20261119/prepare.py:60)，60–65行。
- [S14] [历史低重叠XTW报告](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/xtw_low_overlap/RESULTS.md:1)，1–34行。
- [S15] [sync01同期四法报告](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/results/sync01/RESULTS.md:1)，1–55行。
- [S16] [sync01审计与统计代码](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/sync_report.py:51)，51–188行。
- [S17] [recent5实验登记](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/实验登记.md:3)，3–17行。
- [S18] [recent5同期六法与分批十八法报告](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/results/benchmark01/RESULTS.md:1)，1–113行。
- [S19] [sync01平台与完成记录](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/README.md:3)，3–7行。
- [S20] [发表载体等级核对](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/presentation_20261120/build/venue_rank_sources.md:3)，3–23行；[CCF2026原文第57页](https://kyc.lsu.edu.cn/_upload/article/files/20/77/2cbaa3754eb9aff9ed74cafed8ff/23a8de02-594c-445f-b084-69b0193c05b3.pdf)。
- [S21] [新增五法论文及等级核对](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/presentation_20261120/build/recent5_sources.md:3)，3–23行；其中保存正式发表URL及等级证据性质。
- [S22] [HYBRID协议](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_heatmap_calibration_20261119/protocol.md:25)，25–43行。
- [S23] [ORDER_SPLIT开发与锁版方案](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/开发方案.md:19)，19–37行。
- [S24] [研究总报告的身份和结论](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/总报告.md:354)，354–377行。
- [S25] [reserve02分车与普通重插机制](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/results/reserve02/分车机制.md:3)，3–28行。
- [S26] [DIST与SLOT定义](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_distance_slot_20261119/执行方案.md:5)，5–14行。
- [S27] [已有比较方法适配与DIST_SLOT消融](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_cross_dataset_20261119/protocol.md:3)，3–19行。
- [S28] [PyVRP冷链适配](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/PyVRP_冷链适配.md:3)，3–13行。
- [S29] [RouteFinder与CaDA部署说明](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/近期神经对照_方法说明.md:3)，3–12行。
- [S30] [recent5字段提取和汇总代码](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/recent_report.py:179)，179–215行。
- [S31] [锁版搜索机制计数代码](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/block_search.py:137)，137–254行。
- [S32] [PyVRP子代计数](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/recent_or.py:90)，90–104行。
- [S33] [sync01工作器计时边界](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/sync_worker.py:15)，15–32行。
- [S34] [sync01控制器进程计时](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/sync_runner.py:62)，62–73行。
- [S35] [sync01独立审计JSON](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/supplement_sync_20261120/results/sync01/independent_audit.json:1)，完整对象。
- [S36] [recent5独立审计JSON](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/results/benchmark01/independent_audit.json:1)，完整对象。
- [S37] [reserve02相对百分比计算](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_report.py:56)，56行。
- [S38] [recent5接受、缓存与计时记录](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/recent_worker.py:51)，51–67行；历史缓存未命中字段见[跨来源回执构建](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_cross_dataset_20261119/compare.py:132)，132–135行。
- [S39] [神经构造计数](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/recent5_20261120/neural_methods.py:173)，173行。
