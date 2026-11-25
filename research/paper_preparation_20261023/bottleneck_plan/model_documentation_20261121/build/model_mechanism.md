# 锁版 ORDER_SPLIT：业务模型、掩码网络与经济分车

## 1. 方法整体在解决什么问题

ORDER_SPLIT 将一次配送决策表示为若干条有序客户路线。所有客户必须服务且恰好出现一次，每条路线由一辆车从仓库出发、依次配送并返回仓库。算法同时改变客户的访问顺序和客户在车辆之间的划分，使整个车队的品质折算交付价值减去行驶、制冷费用后的净效益最大。

方法有三个相互衔接的部分。MASKCO 读取当前路线的残缺邻接关系，预测适合连接的客户；普通重插搜索用这些预测筛选值得精确评价的插入位置；跨车搜索把若干条近邻路线的客户重新排序，再通过动态规划选择经济上最合适的分车边界。可行性和方案优劣始终由统一冷链评价器决定。网络输出是邻接偏好，业务评价器输出才是净效益 U。

当前锁版使用 `ORDER_SPLIT`、`maskco_base`、`repeat02/f0_r0` 的 1440 步检查点。这个检查点的实际元数据记载训练种子为 2000，网络参数量为 19,127,808。[锁版配置](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/locked_method.json:2)；[检查点身份](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fixed_adapt_repeat02_20261022/remote/repeat02/outputs/panel/training/f0_r0/checkpoint/checkpoint.json:2)；[实际训练完成记录](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fixed_adapt_repeat02_20261022/remote/repeat02/outputs/panel/training/f0_r0/completion.json:51)。

## 2. 业务符号、目标和约束

### 2.1 一个实例包含哪些信息

节点集合为 \(V=\{0,1,\ldots,100\}\)，0 为仓库，1–100 为客户。客户输入逐行保存为 `(编号, x, y, 需求质量, 时窗开始, 时窗结束, 服务时长)`。主比较统一使用 kg、km、min、K 和净效益单位 U。

| 符号 | 含义 | 锁版取值或来源 |
|---|---|---|
| \(m_i\) | 客户 \(i\) 的配送质量 | 实例原始/登记投影输入，kg |
| \((x_i,y_i)\) | 节点坐标 | 实例输入 |
| \(d_{ij}\) | 节点间距离 | \(\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}\)，km |
| \([a_i,b_i]\) | 客户服务开始时窗 | 实例输入，min |
| \(s_i\) | 服务时长 | 实例输入，min |
| \(Q\) | 单车载重上限 | 20,000 kg |
| \(v\) | 固定车速 | 1 km/min |
| \([a_0,b_0]\) | 仓库时间窗 | 实例输入 |
| \(\mathcal R\) | 全车队路线集合 | 优化变量 |
| \(K=|\mathcal R|\) | 非空路线/车辆数 | \(1\le K\le100\) |

`horizon_min=1440` 同时用于输入特征的固定数值尺度。实际可行性由当前实例各节点时窗决定。配置中“90 min/customer”描述原 VRPTWQ 数据；外部丹麦、XTW、GH 案例的服务时长应直接读取各自登记输入，不能统一替换成90分钟。[模型配置](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:21)；[距离及输入读取](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:24)；[外部实例接入](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/independent.py:41)。

### 2.2 一条路线怎样计算时间

设有序路线为 \(r=(i_1,\ldots,i_m)\)，该车出库时刻为 \(t_r^0\)。对第一个客户，前驱为仓库且其前驱完成时刻为 \(t_r^0\)；其余客户的前驱为上一客户。时间递推为

\[
A_i=C_{\mathrm{prev}(i)}+d_{\mathrm{prev}(i),i}/v,\qquad
B_i=\max(A_i,a_i),\qquad
C_i=B_i+s_i.
\]

其中 \(A_i\) 是到达时刻，\(B_i\) 是服务开始时刻，\(C_i\) 是服务完成/交付时刻；等待时间为 \(B_i-A_i\)。必须满足 \(B_i\le b_i\)，服务完成时刻可以超过客户时窗上界。最后一客服务完成后，车辆空载返回仓库，回仓时刻为

\[
t_r^{\mathrm{return}}=C_{i_m}+d_{i_m,0}/v\le b_0.
\]

每条路线使用规范化的最晚可行出库时刻。令 \(\pi_k\) 为忽略等待时，从仓库到第 \(k\) 个客户服务开始前累计的行驶与前序服务时间，则程序计算

\[
t_r^0=\min\left\{b_0,\ \min_{1\le k\le m}(b_{i_k}-\pi_k),\ 
b_0-\pi_m-s_{i_m}-d_{i_m,0}/v\right\},
\]

随后按前述 `max(到达,时窗开始)` 的规则正向重算等待、服务和回仓。若出库早于 \(a_0\)，或任一服务/回仓超窗，则该路线不可行。最晚出库使能够留在仓库的等待尽量留在出库之前，降低运输阶段的货龄及制冷时长；本模型只从本车出库开始累计运输品质衰减。

独立复算使用另一套逆向时间递推：从回仓截止 \(b_0\) 开始，按
\(L_i=\min\{b_i,L_{\mathrm{next}(i)}-d_{i,\mathrm{next}(i)}/v-s_i\}\)
反推最晚服务开始时刻，再得到首客之前的出库时刻，用于核对正向评价器。[路线评价与出库时刻](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:80)；[逆向独立复算](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/preflight_publication_model_v1.py:31)。

### 2.3 品质如何随运输时间变化

在恒定车厢温度下，采用参考温度平移的 Arrhenius 速率和零阶运输品质衰减：

\[
\kappa(T)=60k_{\mathrm{ref}}\exp\!\left[-\frac{E_a}{R_g}
\left(\frac1T-\frac1{T_{\mathrm{ref}}}\right)\right],\qquad
q_i=1-\kappa(T)(C_i-t_r^0).
\]

\(k_{\mathrm{ref}}\) 是参考温度下以秒为单位的衰减率；乘60将其换成每分钟速率。\(E_a\) 为活化能，\(R_g\) 为气体常数，\(T\) 为车厢绝对温度，\(q_i\) 为交付时的运输品质保留比例。主条件如下。

| 参数 | 数值 | 含义 |
|---|---:|---|
| \(T=T_{\mathrm{ref}}\) | 275 K | 恒定车厢温度和参考温度 |
| \(T_{\mathrm{ambient}}\) | 293 K | 制冷热负荷中的环境温度 |
| \(R_g\) | 8.3145 J/(mol·K) | 气体常数 |
| \(k_{\mathrm{ref}}\) | \(3.08\times10^{-6}\ \mathrm{s}^{-1}\) | `main_Product1` 参考速率 |
| \(E_a\) | 77,900 J/mol | 主品质参数组的活化能 |
| \(q_{\min}\) | 0.8 | 交付品质硬下限 |

因主条件 \(T=T_{\mathrm{ref}}\)，\(\kappa=0.0001848\ \mathrm{min}^{-1}\)。品质约束等价于每位客户的出库至交付货龄不超过

\[
C_i-t_r^0\le (1-q_{\min})/\kappa\approx1082.2511\ \mathrm{min}.
\]

这个货龄包含行驶、途中等待以及交付前的服务时间。品质影响交付价值，配送质量 \(m_i\) 本身不会按品质减少。训练还包含 `slow_Product3` 和 `fast_Product2` 两组文献动力学参数、以及0.85品质阈值；主外部比较固定使用上述主条件。[动力学参数](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:36)；[速率和最长货龄实现](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:58)。

### 2.4 为什么目标不是最短路

令 \(p\) 为完全保质货物的单位质量参考价值，\(c_d\) 为单位里程费用，\(c_f\) 为每车每分钟制冷费用。路线净效益与车队总目标为

\[
u(r)=\sum_{i\in r}p\,m_iq_i-c_dD_r-c_f(t_r^{\mathrm{return}}-t_r^0),
\qquad \max_{\mathcal R}\ U(\mathcal R)=\sum_{r\in\mathcal R}u(r).
\]

\(D_r\) 包括仓库至首客、客户之间、末客回仓的全部距离。主参数为 \(p=5/3\ \mathrm{U/kg}\)、\(c_d=1\ \mathrm{U/km}\)、\(c_f=0.0972\ \mathrm{U/(车\cdot min)}\)。U 是按基准里程费用归一化的研究货币单位；\(p\) 来自跨文献保留的货值与里程费用比例。

制冷费用由固定壁面热负荷近似给出：

\[
\dot Q=\frac{\lambda A}{\delta}(T_{\mathrm{ambient}}-T),\qquad
c_f=c_Q\dot Q.
\]

\(\lambda=0.36\ \mathrm{W/(m\cdot K)}\) 为导热系数，\(A=30\ \mathrm{m^2}\) 为壁面积，\(\delta=0.1\ \mathrm m\) 为壁厚，\(c_Q=0.00005\ \mathrm{U/(W\cdot min)}\)。代入得 \(\dot Q=1944\ \mathrm W\)、\(c_f=0.0972\)。制冷从出库计至空载回仓，覆盖行驶、等待和服务。

由于所有客户都必须交付，总名义价值 \(p\sum_i m_i\) 是常数。因此最大化 U 等价于最小化“品质损失价值＋里程费用＋制冷费用”：

\[
p\sum_i m_i-U
=\sum_i p\,m_i(1-q_i)+c_d\sum_rD_r+c_f\sum_r(t_r^{\mathrm{return}}-t_r^0).
\]

这解释了多分一辆车可能值得：它增加往返里程与部分制冷成本，却可能减少高价值货物的运输货龄。净收益取决于两者的权衡。主目标没有单独车辆启用费，车辆数通过真实路线费用与品质价值间接决定；品质损失已经反映在交付价值中，目标不重复扣减同一笔损失。[经济参数](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:60)；[路线经济核算](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:101)；[全车队汇总](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_core.py:77)。

容量 \(\sum_{i\in r}m_i\le Q\)、服务开始时窗、交付品质、仓库出入窗及回仓要求均为硬约束。所有客户必须恰好出现一次；不能拆分配送，一辆车只执行一条路线。\(K\le100\) 是100个必服务客户下的非空路线数学上界。[覆盖与路线约束](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:120)。

## 3. MASKCO 看到了什么

### 3.1 101×16 节点输入

一张图有仓库加100位客户，共101行。锁版模型的节点输入为 \(X\in\mathbb R^{101\times16}\)。记固定尺度 \(T_0=1440\)，可见标记 \(z_i=1\) 表示仓库或未被遮蔽客户、\(z_i=0\) 表示本次遮蔽客户。0基列号与实际输入一致。

| 列 | 输入数值 | 直观含义 |
|---:|---|---|
| 0 | \((x_i-x_0)/1440\) | 相对仓库横坐标的固定缩放 |
| 1 | \((y_i-y_0)/1440\) | 相对仓库纵坐标的固定缩放 |
| 2 | \(m_i/20000\) | 占单车容量的比例 |
| 3、4 | \(a_i/1440,b_i/1440\) | 服务开始时间窗 |
| 5 | \(s_i/1440\) | 服务时长 |
| 6 | \(\kappa T_0\) | 在固定时间尺度内的品质衰减强度 |
| 7 | \(q_{\min}\) | 当前任务的品质下限 |
| 8 | \(\log(1+pQ/T_0)\) | 单位货值与任务尺度的对数编码 |
| 9 | \(c_f\) | 制冷费用系数 |
| 10 | \(z_i\) | 本轮客户是否可见 |
| 11–15 | 0 | 锁版 `maskco_base` 中为零的五个通道 |

其中坐标除以1440是程序的固定数值缩放，不能改写成“按每个实例最大坐标归一化”。隐藏客户仍保留其坐标、需求、时窗等已知业务信息；被隐藏的是它在当前解中的路线连接关系。

锁版使用的 `maskco_base` 对应 `quality_context=False, directed=False`。因此，输入中已有品质速率和品质阈值等任务条件，但第11–15列并没有注入客户的货龄、品质余量或载重上下文。[变体定义](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_core.py:24)；[16维特征逐列构造](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_core.py:118)。

### 3.2 残缺邻接矩阵和“时间步”

网络的第二个输入为 \(A^{\mathrm{vis}}\in\mathbb R^{101\times101}\)。原路线中的每条边按两个方向累加；被遮蔽节点对应的整行、整列置零。仓库不被遮蔽。单客路线的客户—仓库边会因往返两次而具有计数2，这与原目标邻接定义一致。

保留的邻接边必须是原路线中两端都可见的边。即使删除客户后搜索器把剩余客户连成一条可行残缺路线，网络输入也不会自动添加跨越遮蔽缺口的捷径边。

第三个输入为

\[
\tau=\frac{\sum_{i=1}^{100}\sum_{j=0}^{100}A^{\mathrm{vis}}_{ij}}{200}.
\]

\(\tau\) 表示客户邻接信息的可见程度。分母200来自完整无向路线中100个客户的两侧邻接计数。它不是业务中的配送时间，也不是搜索已经运行了多少秒。程序将 \(\tau\) 放大1000后做正余弦频率嵌入，再经小型网络映射到256维，加入每个解码节点的表示。[推理输入构造](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_masked_reconstruction_20261118/inference_inputs/base_inputs.py:16)；[时间步嵌入](D:/PyCharm_/MASKCO-Main/MASKCO_code/modules/TimestepEmbedder.py:42)；[解码器条件注入](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/TSPModel.py:143)。

### 3.3 16层编码器、6层解码器如何输出边分

锁版架构使用作者 `CVRPModel` 主干：输入16维先映射到256维，并给仓库表示加可学习偏置；随后经16层编码器，形成实例节点表示。中间投影保持256维，加入上述可见程度嵌入后，进入6层解码器。编码器和解码器每层均有8个注意力头，每头维度为 \(256/8=32\)。Transformer 层使用前置 LayerNorm、注意力残差和 SwiGLU 前馈残差。SwiGLU 隐藏维度是 \(3\times256=768\)；解码结束后的独立最终投影 MLP 则为256→256→256。[主干实际配置](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py:18)；[仓库偏置](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/CVRPModel.py:91)；[网络连接](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/TSPModel.py:85)；[768维与残差层](D:/PyCharm_/MASKCO-Main/MASKCO_code/modules/TransformerLayer.py:31)。

每个注意力头以缩放点积得到节点关系，在进入 softmax 前使用 softcap 30；解码器再加上可见邻接矩阵作为注意力偏置：

\[
W_{ij}^{(h)}=\operatorname{softmax}_{j}
\left[30\tanh\left(\frac{Q_i^{(h)}K_j^{(h)\mathsf T}}{30\sqrt{32}}\right)
+A^{\mathrm{vis}}_{ij}\right].
\]

这里 \(Q^{(h)},K^{(h)}\) 分别为第 \(h\) 个头的查询、键，\(W^{(h)}\) 为该头的注意力权重；编码器使用同样的缩放点积但不加路线邻接偏置。邻接矩阵是正向偏置，不会把所有未知边当作禁止访问。SwiGLU 的核心计算为
\(\mathrm{SiLU}(xW+b_W)\odot(xV+b_V)\)，再投影回256维；\(\odot\) 为逐元素乘法。[实际注意力实现](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py:24)；[SwiGLU](D:/PyCharm_/MASKCO-Main/MASKCO_code/modules/SwiGLU.py:6)。

令最终节点表示经 RMSNorm 后为 \(z_i\in\mathbb R^{256}\)，输出边 logits 为

\[
\ell_{ij}=\frac{16}{256}\,z_i^{\mathsf T}z_j\quad(i\ne j).
\]

自环对角线设为极大的负有限值；最终 logits 不再使用 softcap。锁版使用同一组节点表示的内积，因此 \(\ell_{ij}=\ell_{ji}\)。随后逐行计算

\[
P_{ij}=\frac{\exp(\ell_{ij})}{\sum_k\exp(\ell_{ik})},\qquad
H_{ij}=\log P_{ij}=\ell_{ij}-\operatorname{logsumexp}_k\ell_{ik}.
\]

求解器使用的是 \(H\)，即 log-softmax 后的边分；数值越大表示该行更偏好连接这个邻居。虽然 logits 对称，逐行归一化的分母通常不同，\(H\) 一般不对称。\(H\) 的一行归一化描述邻接偏好，不能把它直接理解成一条完整可行路线的生成概率。[对称输出头](D:/PyCharm_/MASKCO-Main/MASKCO_code/models/TSPModel.py:158)；[实际推理输出与log-softmax](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_distance_slot_20261119/worker.py:44)。

## 4. 网络如何训练

### 4.1 Teacher 是冷链可行路线

训练监督来自已登记、按完整冷链目标评价的教师路线。`f0_r0` 包含38个 R/RC 训练案例，每例覆盖3组品质动力学参数×2个品质阈值，共228份任务条件及对应路线。固定教师先来自旧训练阶段的可行路线，再有一次以完整净效益为目标、每条件1秒的 ALNS 单调改进；训练读取的是已经保存并复算的标签。它们是启发式教师，不具有全局最优证书。[固定教师接入与六条件完整性](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/training_data.py:17)；[教师改进代码](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/teacher_upgrade.py:43)；[已记录的1秒教师改进](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/algorithm_dev_20261021/remote_3080/dev03/outputs/teacher_upgrade/teacher_refinement.json:1)；[当前训练教师数量](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/fixed_adapt_repeat02_20261022/remote/repeat02/outputs/panel/training/f0_r0/training_plan.json:33173)。

### 4.2 从完整教师路线构造交叉熵目标

对教师路线的每条相邻边，包括仓库边，按两个方向构造完整邻接计数 \(A^*\)。逐行归一化得监督分布

\[
Y_{ij}=\frac{A^*_{ij}}{\sum_k A^*_{ik}}.
\]

普通客户的前、后邻居通常各占1/2；单客路线的两个邻接都指向仓库，仓库目标概率为1。训练时随机遮蔽20名客户，只把残缺邻接交给网络，但监督目标仍是完整教师邻接。损失为

\[
\mathcal L_{\mathrm{CE}}
=-\frac{\sum_{i=0}^{100}w_i\sum_{j=0}^{100}Y_{ij}\log P_{ij}}
{\sum_{i=0}^{100}w_i},\qquad w_0=0,\quad w_i=1\ (i\ge1).
\]

损失覆盖全部100个客户行，并非只对20个被遮蔽客户求损失。仓库行权重为0。网络通过恢复邻接学习教师解的局部结构，交叉熵并不直接等于 U，也不通过完整冷链求解器反向传播。[目标邻接与行权重](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_core.py:155)；[交叉熵实际实现](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py:91)。

训练用 AdamW，学习率和权重衰减均为 \(10^{-4}\)，梯度全局范数裁剪阈值为1；每次更新处理一份教师图及其掩码。`f0_r0` 从种子2000初始化，掩码流种子2000，教师顺序打乱种子2001，完成1440次优化器更新，约等于228份教师的6.3158次遍历。每次更新重新使用登记的20客掩码。现有检查点明确记载这些取值；本章节不为未显式记录的优化器默认项补造参数。[训练流](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/fixed_repeat.py:28)；[实际更新循环](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/staged_repeat.py:54)；[优化器](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py:83)。

## 5. 原始热图如何得到距离校准热图

原热图、纯距离分和校准热图分别记为 \(H\)、\(D_{ij}=-d_{ij}\)、\(C\)。对每个节点按距离由近至远得到秩 \(r^D_{ij}\)，按热图分由高至低得到秩 \(r^H_{ij}\)，秩从1开始，排序排除对角自环。定义

\[
R_{ij}=\operatorname{clip}\left(\log\frac{r^D_{ij}}{r^H_{ij}},-3,3\right),\qquad
S_{ij}=-d_{ij}/s,\qquad C_{ij}=S_{ij}+\alpha R_{ij}.
\]

\(R_{ij}>0\) 表示学习热图相比单纯距离更推荐这条边；\(R_{ij}<0\) 表示更不推荐。尺度 \(s\) 为所有距离行中“第9小距离”的中位数，数组以0基第8位取值且包含对角零距离；若该尺度非正，则使用所有正距离的中位数。\(\alpha\) 调节“超出距离所能解释的学习偏好”的权重。

冻结 \(\alpha=0.25\)。候选集合是 \(\{0,0.25,0.5,1,2,4,8,16\}\)，在12个 R/RC 训练案例的固定插位比较中，以几何宏平均插位遗憾选择。校准记录包含120次前向和4320个单客插位标签；这一步选择后处理系数，不更新网络权重。实际插位遗憾以候选集能获得的最好精确净增量与穷举可行插位的最好净增量之差定义。[秩残差与校准损失](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/calibration.py:8)；[冻结系数与训练案例](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/calibrated.json:2)。

## 6. 普通掩码重构：预测一次，完成多次破坏修复

### 6.1 三种不同的“掩码大小”

训练每次遮蔽20客。在线每个外层时段先抽取40客构成候选池，并对当前路线执行一次前向预测。这40客在网络邻接输入中被隐藏，但一次原生重构不必移除全部40客：greedy 重构随机移除6–17客；regret-2 固定移除10客，若候选池较小则取其上限。之后在同一个外层时段内反复破坏修复，共用该次热图。这样将一次网络前向开销分摊给多个经济评价与搜索动作。[40客前向池](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:139)；[实际重构移除数量](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp:235)。

### 6.2 六种短时试选

普通重插使用三种位置排序分 \(D,H,C\)，分别搭配 greedy 与 regret-2，得到六个候选配置。若本时段还剩超过0.30秒，六个配置从同一当前路线状态和相同试选种子出发，各执行完整0.04秒的短搜索。按短搜索得到的最好车队净效益选出本时段继续使用的配置；得分相同时，按外层时段编号轮换优先级。剩余时间不足以完成0.04秒的试选不纳入比较。随后选中的配置运行本时段剩余普通搜索。

| 边排序 | greedy 插回 | regret-2 插回 |
|---|---|---|
| \(-d_{ij}\) | 距离＋greedy | 距离＋regret-2 |
| \(H_{ij}\) | 原始热图＋greedy | 原始热图＋regret-2 |
| \(C_{ij}\) | 校准热图＋greedy | 校准热图＋regret-2 |

外层时段以起点后1.6秒为局部时间边界，网络前向、跨车分车和普通试选共享这个时段，且始终受60秒全局截止约束。1.6秒是调度边界，单个原生调用及评价结束可能略跨出局部边界；正式候选是否计入结果由严格评价完成时间决定。[六项生成与0.04秒试选](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:190)。

### 6.3 热图负责候选筛选，净效益负责选位置

设把客户 \(i\) 插在一条路线的相邻节点 \((a,b)\) 之间。其新增距离和预测边分增量分别为

\[
\Delta d(a,i,b)=d_{ai}+d_{ib}-d_{ab},\qquad
\Delta h(a,i,b)=h_{ai}+h_{ib}-h_{ab},
\]

其中 \(h\) 是本时段选中的 \(D,H,C\) 之一。对保留下来的位置，程序重算整条新路线，得到真实净效益增量
\(\Delta u=u(r\oplus(i,\mathrm{pos}))-u(r)\)。插位的时窗、货龄、制冷和载重影响都进入这个增量。另开一辆单客车也是候选，其净增量为 \(u((i))\)。

greedy 按破坏时的客户顺序依次插回。对一个客户，先保留新增距离最小的6个位置，再从其余位置中保留边分增量最大的6个位置，最多12个位置进行精确比较。若这些现有路线插位均不可行，按距离次序继续检查剩余位置，直到找到可行插位或耗尽；单客车候选始终保留。

regret-2 对每个待插客户选边分增量最高的8个位置，并保护至少一个新增距离最小的可行位置；必要时用它替换所选位置中的末位。将这些位置与单客车一起比较，得到最好和次好的真实增量 \(g_i^{(1)},g_i^{(2)}\)，优先插入遗憾 \(g_i^{(1)}-g_i^{(2)}\) 最大的客户。只有一个有限候选时，程序以最好增量本身作为遗憾值。直观上，它优先处理“错过最佳位置后损失最大”的客户。[插位候选和经济比较](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp:83)；[两种插回顺序](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp:171)。

### 6.4 如何兼顾探索与保留最好解

程序区分三种对象：`state` 是继续搜索的当前状态；`best` 是原生搜索维护的最好候选缓存；`clock.events` 保存已完成严格评价、在预算内接纳的正式最好记录。每4个外层时段把 `state` 重置为 `best`。普通搜索可以接受小幅下降的可行解，以越过局部最优。C++ 可以先更新 `best`，随后 Python 才调用 `clock.offer`；若严格评价结束时已经超时，这个候选不会进入正式记录。因此最终输出以 `clock.events` 为准，不能直接把缓存中的 `best` 等同于最终预算内结果。

外层在两种破坏模式间自适应选择：原40客池的随机破坏，以及围绕当前邻接与预测偏好冲突的聚焦破坏。聚焦池优先选择当前前后邻边排名较差的12客，为其补充预测相近客户，并保留原池前8客。模式使用次数为 \(n_m\)、历史回报为 \(R_m\)；前两个时段和每4个时段优先选择使用较少的模式，其余时段最大化 \(R_m+0.1/\sqrt{1+n_m}\)。回报更新为

\[
R_m\leftarrow0.65R_m+0.35\min\{3,\log(1+\max(0,\Delta U_{\mathrm{best}}))\}.
\]

\(\Delta U_{\mathrm{best}}\) 是这个时段最好净效益的提升。相关破坏模式还会在原生批次的交替迭代中，围绕随机锚点，按双向边分 \(h_{ij}+h_{ji}\) 偏向抽取关联客户。

主搜索温度为 \(\theta(t)=2.5\max\{0.1,1-t/60\}\)；短试选温度固定1.5。若新解不比当前状态差则接受；较差解必须满足 \(U_{\mathrm{new}}>U_{\mathrm{best}}-6\theta\)，再按 \(\exp((U_{\mathrm{new}}-U_{\mathrm{current}})/\theta)\) 的概率接受。\(t\) 为在线已用秒数，温度和阈值均作用于 U 差额。[破坏池与模式选择](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:78)；[模式回报与温度](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:219)；[原生接受规则](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp:255)。

## 7. 跨车访问顺序与 Split 经济分车

### 7.1 哪些路线一起重新安排

每3个外层时段尝试一次跨车分车。随机选一条当前路线为锚点，按“该路线任一客户到另一条路线任一客户的最小距离”选择相近路线。输入组合包含2–4条路线，客户总数不超过24；没有满足条件的至少两条路线则跳过该次跨车动作。

组合内全部客户必须进入本轮40客隐藏池，余下名额从其他客户中抽取。对这个残缺状态生成热图后，用原始 \(H\) 引导候选访问顺序；这一跨车分支不直接使用普通重插试选出的校准热图。纯距离消融在相同位置使用 \(-d\)。[选择路线组合](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/block_split.py:34)；[触发和隐藏池](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:144)。

候选顺序从这些路线的随机排列开始，每条原路线以0.25概率反向，再做1–4次局部扰动。扰动选择某一客户及其双向边分较高的非相邻客户，在最多前8名中偏向高排名抽样，执行片段反转或把客户移至目标客户之前/之后。由于冷链时间是有向递推，即使坐标距离对称，路线反向后的品质和可行性也必须重新评价。[顺序提案](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/block_split.py:45)。

跨车分支的截止点为“全局截止前0.003秒、当前时段起点后1.6秒、分支开始后0.30秒”三者中的最早时刻。它在这段时间内不断产生访问顺序并调用 Split，每个顺序得到该顺序下最好的连续分车。组合价值首先与从当前 `state` 抽出的原组合比较；增加超过 \(10^{-7}\) U 且此刻尚未到全局截止时，程序即更新 `state`。这项当前状态改善不要求超过正式全局最好记录。只有新车队净效益还超过 `clock.events` 中的最好值，并且 `clock.offer` 完成严格评价后在预算内接纳，才同时更新 `best` 并记一次正式跨车改进。[0.30秒分支和三层接纳](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:161)。

### 7.2 给定顺序后怎样求最优分车

给定组合内访问顺序 \(\pi=(\pi_1,\ldots,\pi_m)\)，\(m\le24\)。把连续子段 \((\pi_{i+1},\ldots,\pi_j)\) 作为一辆车的完整仓库往返路线，并用前文同一个冷链模型计算其净效益

\[
g(i,j)=u((\pi_{i+1},\ldots,\pi_j)),\qquad 0\le i<j\le m.
\]

若子段违反容量、时间窗、品质或回仓约束，令 \(g(i,j)=-\infty\)。动态规划状态 \(F(j)\) 表示前 \(j\) 个客户的最大可行分车净效益：

\[
F(0)=0,\qquad
F(j)=\max_{0\le i<j}\{F(i)+g(i,j)\}.
\]

同时记录使其改善的前驱 \(\mathrm{prev}(j)=i\)，从 \(m\) 逆向回溯即可恢复各车辆的连续客户子段。代码将每一子段都完整重算，更新采用 \(10^{-7}\) U 阈值。它等价于在顶点0到 \(m\) 的有向无环图上求最大权路径：一条从 \(i\) 到 \(j\) 的弧代表“用一辆车服务这个子段”，弧权就是其真实净效益。[DP核心](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/split_order.cpp:3)；[前驱回溯](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/block_split.py:21)。

输入组合有2–4辆车，不代表输出必须仍为2–4辆车。分车结果可以增加或减少路线数；在所有子段可独立评价且车辆数只受100客户非空路线界约束的条件下，DP自动选择最有利的分段。单客可行性准入保证至少存在逐客分车的可行方案。

对固定 \(\pi\)，该DP穷举连续分段，因而得到该顺序下的最优分车；访问顺序本身由限时启发式搜索产生。每个顺序至多评价 \(m(m+1)/2\) 个子段，\(m=24\) 时至多300次路线评价。当前实现每次子段评价都遍历该段，因此朴素最坏时间为 \(O(m^3)\)，DP状态空间为 \(O(m)\)。这里的“精确”指固定顺序内的分段优化，不构成全问题全局最优证明。[子段路线评价](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/masked_repair.cpp:23)。

## 8. 在线预算、独立评价与消融身份

在线60秒从实例任务建立和共同可行初解开始计时。模型加载、实例无关准备单独计量。共同初解按最近邻逐客尝试，只接纳完整路线仍可行的追加客户，无法追加时开新车，最终做全车队独立复算。[共同初解](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/preflight_publication_model_v1.py:85)；[求解起点](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:112)。

候选先完成完整硬约束检查和独立经济复算，再读取完成时间。超过60秒的候选不登记为预算内结果，即使它更好；预算内也只保留比已有最好净效益高超过 \(10^{-7}\) U 的候选。最终报告另外重建路线再次评价；末尾复核用时可以使整个进程超过60秒，但它不会将新的迟到解纳入在线终点。[在线接纳时钟](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools/run_dev72_clock.py:14)；[结束与复核时间](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/timed_search.py:48)。

同规则消融 `ORDER_SPLIT_D` 保留外层调度、破坏修复、精确路线经济评价和跨车 Split，使用距离边分替代神经预测；普通试选只剩距离＋greedy、距离＋regret-2两项。因此完整方法与该消融之间的差额包含热图带来的候选质量收益、预测时间以及额外试选的机会成本。它评价的是学习引导在既定求解框架中的净作用。与 ALNS、PyVRP 或其他完整方法的差额则评价整套求解器，不宜全部归因于热图。[距离消融分支](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:106)。

## 9. 模型假设与解释边界

业务侧使用恒温、零阶运输品质衰减和线性品质价值折算，出库前仓储老化、开门温度瞬变、实际电功率动态和额外工资/启用费用未进入主目标。品质阈值0.8是研究接受条件；U是统一研究标尺。这些假设决定了本文“静态冷链净效益”的含义。[冻结模型定义与范围](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:38)。

方法侧使用单个固定训练检查点，学习目标是教师邻接重构，在线目标才是车队净效益。训练掩码20客与在线前向40客属于两个不同阶段的固定设置。普通搜索、访问顺序提案及时间分配均为启发式；固定顺序的连续 Split 和路线评价则按既定数学模型计算。网络能否带来稳定净收益要由同规则消融回答，不能仅凭架构图、前向调用次数或完整方法排名推断。

本章节对数值、边界和流程均采用锁版代码及已保存训练元数据。未显式登记的优化器默认值、教师的全局最优性、现实市场绝对利润和全局最优误差均不赋予推测值。
