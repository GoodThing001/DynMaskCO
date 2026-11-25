# 数学符号速查与手算导读

本节把业务模型、神经网络、搜索和路线主问题放在同一份符号索引中。阅读时先确定公式属于哪一层，再看变量的下标、形状和单位。相同字母在不同领域有时代表不同对象；本节在必要处加上说明性下标，帮助区分这些对象，不改变实际计算。

## 1. 物理冷链与路线时间

| 符号 | 读法或英文 | 含义 | 单位 | 首次出现概念 |
|---|---|---|---|---|
| \(V\) | vertex set | 包含仓库和客户的节点集合；主实例为0–100 | 集合 | 配送图 |
| \(0\) | depot | 仓库节点编号 | 编号 | 配送图 |
| \(n\) | number of customers | 客户总数；正式主实例为100 | 客户数 | 配送图 |
| \(i,j\) | customer/node indices | 节点或客户编号；其他层可能作状态/覆盖项索引 | 编号 | 配送图 |
| \((x_i,y_i)\) | coordinates | 节点位置的横、纵坐标 | 坐标按km解释 | 距离模型 |
| \(d_{ij}\) | distance | \(i\) 到 \(j\) 的欧氏距离 | km | 距离模型 |
| \(v\) | speed | 固定行驶速度，主参数为1 | km/min | 路线时间 |
| \(m_i\) | demand mass | 客户 \(i\) 的配送质量 | kg | 需求与容量 |
| \(Q\) | vehicle capacity | 单车载重上限，主参数为20,000 | kg | 需求与容量 |
| \(r\) | route | 一条有序、非空客户路线，默认含仓库往返 | 有序序列 | 路线表示 |
| \(\mathcal R\) | fleet route set | 车队所有路线组成的集合 | 集合 | 车队方案 |
| \(K=|\mathcal R|\) | number of vehicles | 非空路线数，一条路线对应一辆车 | 辆 | 车队方案 |
| \(a_i,b_i\) | window opening/closing | 客户服务开始时刻的下界和上界 | min | 客户时间窗 |
| \(a_0,b_0\) | depot opening/closing | 仓库出入时间窗边界 | min | 仓库时间窗 |
| \(s_i\) | service time | 客户服务持续时间 | min | 路线时间 |
| \(t_r^0\) | dispatch/departure time | 路线 \(r\) 的出库时刻 | min | 规范化最晚出库 |
| \(A_i\) | arrival time | 到达客户 \(i\) 的时刻 | min | 正向时间递推 |
| \(B_i\) | service start | 开始服务时刻，等于 \(\max(A_i,a_i)\) | min | 正向时间递推 |
| \(C_i\) | service completion | 服务完成即交付时刻，等于 \(B_i+s_i\) | min | 交付事件 |
| \(B_i-A_i\) | waiting time | 到达后因时窗尚未开放而等待的时间 | min | 途中等待 |
| \(t_r^{\mathrm{return}}\) | return time | 最后交付完成后空载回仓的时刻 | min | 回仓约束 |
| \(\pi_k\) | pi, no-wait prefix | 从仓库到第 \(k\) 位客户服务开始前，忽略等待的累计行驶与前序服务时间 | min | 最晚出库公式 |
| \(L_i\) | latest feasible service start | 独立逆向复算中的最晚服务开始时刻 | min | 逆向时间复算 |
| \(D_r\) | route length | 路线的完整仓库往返总距离 | km | 路线费用 |

时间窗约束作用在 \(B_i\)，品质作用在 \(C_i-t_r^0\)，制冷作用在 \(t_r^{\mathrm{return}}-t_r^0\)。这三个时间量分别回答“是否按约开始服务”“交付前货物运输了多久”“车辆制冷系统工作了多久”。[实际时间计算](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/comparisons/static_coldchain/adapters/publication_model_v1.py:80)。

## 2. 品质、热负荷与净效益

| 符号 | 读法或英文 | 含义 | 单位 | 首次出现概念 |
|---|---|---|---|---|
| \(T\) | cabin temperature | 车厢恒定绝对温度，主参数275 | K | Arrhenius品质模型 |
| \(T_{\mathrm{ref}}\) | reference temperature | 参考反应温度，主参数275 | K | Arrhenius品质模型 |
| \(T_{\mathrm{ambient}}\) | ambient temperature | 车外温度，主参数293 | K | 制冷热负荷 |
| \(k_{\mathrm{ref}}\) | reference rate | 参考温度下每秒品质衰减率，主参数 \(3.08\times10^{-6}\) | \(\mathrm{s}^{-1}\) | Arrhenius品质模型 |
| \(E_a\) | activation energy | 活化能，主参数77,900 | J/mol | Arrhenius品质模型 |
| \(R_g\) | gas constant | 气体常数，主参数8.3145 | J/(mol·K) | Arrhenius品质模型 |
| \(\kappa(T)\) | kappa, rate per minute | 温度修正并转成每分钟的品质衰减率 | \(\mathrm{min}^{-1}\) | 品质衰减 |
| \(q_i\) | quality retention | 客户交付时的运输品质保留比例 | 无量纲 | 品质衰减 |
| \(q_{\min}\) | minimum retention | 品质硬下限，主条件0.8 | 无量纲 | 品质约束 |
| \((1-q_{\min})/\kappa\) | maximum delivery age | 从该车出库到交付的最大允许货龄 | min | 品质约束等价形式 |
| \(p\) | reference value per mass | 完全保质货物的参考单位质量价值，主参数 \(5/3\) | U/kg | 品质折算交付价值 |
| \(p\,m_iq_i\) | quality-adjusted value | 客户 \(i\) 的品质折算交付价值 | U | 交付价值 |
| \(p\,m_i(1-q_i)\) | quality loss value | 相对完全保质状态的价值损失 | U | 等价损失表达 |
| \(c_d\) | distance cost | 单位里程费用，主参数1 | U/km | 行驶费用 |
| \(\lambda\) | lambda, conductivity | 车厢壁面导热系数，主参数0.36 | W/(m·K) | 制冷热负荷 |
| \(A\) | wall area | 传热面积，主参数30 | \(\mathrm m^2\) | 制冷热负荷 |
| \(\delta\) | delta, wall thickness | 壁厚，主参数0.1 | m | 制冷热负荷 |
| \(\dot Q\) | Q-dot, heat-load rate | 恒温条件下壁面热负荷，主参数1944 | W | 制冷热负荷 |
| \(c_Q\) | heat-load cost coefficient | 热负荷与时间换算费用的系数，主参数0.00005 | U/(W·min) | 制冷费用 |
| \(c_f\) | refrigeration cost rate | 每车每分钟制冷费，等于 \(c_Q\dot Q=0.0972\) | U/(车·min) | 制冷费用 |
| \(u(r)\) | route net benefit | 一条路线的品质折算价值减行驶、制冷费用 | U | 路线目标 |
| \(U(\mathcal R)\) | fleet net benefit | 全车队路线净效益之和 | U | 车队目标 |
| \(\Delta U\) | delta U | 两方案或两方法的净效益差，必须同时说明相减方向 | U | 配对收益 |

品质保留比例作用于价值，而非把货物重量打折；容量约束仍使用原需求质量 \(m_i\)。\(\dot Q\) 的 W 是热负荷单位，不能直接解释成实际电功率。主目标使用的参考货值、恒温品质和制冷系数来自冻结的研究模型。[主参数定义](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/configs/static_delivery/publication_model_v1.json:36)。

## 3. 网络、注意力与训练目标

| 符号或术语 | 读法或英文 | 含义 | 单位/形状 | 首次出现概念 |
|---|---|---|---|---|
| \(X\) | feature matrix | 节点输入特征，含仓库和100客的16通道 | \(101\times16\) | 网络输入 |
| \(T_0\) | normalization scale | 输入采用的固定1440时间尺度 | 数值尺度，时间通道以min对应 | 特征归一化 |
| \(z_i\)（标量） | visibility indicator | 可见客户/仓库为1，被遮蔽客户为0 | 0或1 | 掩码输入 |
| \(A^{\mathrm{vis}}\) | visible adjacency | 保留原路线可见边的邻接计数矩阵 | \(101\times101\) | 残缺路线输入 |
| \(\tau\) | tau, visible-edge fraction | 客户行可见邻接计数除以200 | 无量纲标量 | 可见程度条件 |
| \(d_{\mathrm{in}}\) | input dimension | 输入通道数16 | 维度 | 网络结构 |
| \(L_{\mathrm{enc}},L_{\mathrm{dec}}\) | encoder/decoder depth | 编码器16层、解码器6层 | 层数 | 网络结构 |
| \(d_{\mathrm{model}}\) | embedding dimension | 节点隐表示宽度256 | 维度 | 网络结构 |
| \(h\)（头编号） | attention head index | 多头注意力中某一个头的索引，共8头 | 编号 | 多头注意力 |
| \(d_{\mathrm{head}}\) | head dimension | 每个注意力头宽度，\(256/8=32\) | 维度 | 多头注意力 |
| \(Q^{(h)},K^{(h)},V^{(h)}\) | query/key/value | 每个头由节点表示投影得到的查询、键、值 | 每矩阵 \(101\times32\) | 缩放点积注意力 |
| \(W_{ij}^{(h)}\) | attention weight | 第 \(h\) 个头中节点 \(i\) 对节点 \(j\) 的注意力权重 | 无量纲；每行和为1 | 注意力汇聚 |
| attention | attention mechanism | 由节点间查询—键匹配决定怎样加权汇总值向量 | 网络运算 | 节点间信息交换 |
| softcap | soft capping | 用 \(30\tanh(\cdot/30)\) 平滑限制注意力打分幅值 | 数值变换 | 注意力数值结构 |
| LayerNorm | layer normalization | 按特征维度使用均值和方差规范化表示 | 网络运算 | Transformer层 |
| RMSNorm | root-mean-square normalization | 按特征均方根规范化表示；锁版输出头启用可学习缩放 | 网络运算 | 邻接输出头 |
| MLP | multilayer perceptron，多层感知机 | 由线性层与激活函数组成的前馈网络；本模型最终投影为256→256→256 | 网络结构 | 解码后投影 |
| SiLU | sigmoid linear unit | 激活函数 \(x/(1+\exp(-x))\) | 数值变换 | 前馈激活 |
| SwiGLU | Swish-gated linear unit | 用SiLU分支与另一投影分支逐元素相乘的门控前馈结构 | 网络结构 | Transformer前馈层 |
| \(d_{\mathrm{ff}}\) | feed-forward width | Transformer内SwiGLU的隐藏宽度768 | 维度 | 前馈网络 |
| \(W,V,W_2\)（SwiGLU中） | projection matrices | 两条门控投影和输出投影的可训练权重 | 相应矩阵形状 | SwiGLU |
| \(b_W,b_V\) | bias vectors | SwiGLU投影的可训练偏置 | 向量 | SwiGLU |
| \(\odot\) | Hadamard product | 两个同形状张量逐元素相乘 | 运算 | SwiGLU门控 |
| \(z_i\)（向量） | final node embedding | RMSNorm之后用于产生边logits的节点表示 | \(\mathbb R^{256}\) | 邻接输出头 |
| \(\ell_{ij}\) | ell, edge logit | softmax之前的原始边分；锁版内积头输出对称 | 无量纲数值 | 邻接输出头 |
| \(P_{ij}\) | row-normalized probability | 对 \(\ell\) 逐行softmax后的邻接偏好 | 无量纲；每行和为1 | 邻接分布 |
| \(H_{ij}\) | log probability / heatmap | \(\log P_{ij}\)，普通搜索使用的原始热图分 | 无量纲对数值 | 搜索热图 |
| softmax | normalized exponential | 指数化并归一化一行分数，得到非负、和为1的分布 | 运算 | 输出分布/注意力 |
| log-softmax | log of softmax | 直接计算归一化分布的对数，提高数值稳定性 | 运算 | 热图与CE |
| logsumexp | log-sum-exp | \(\log\sum_j\exp(\ell_{ij})\)，一行的对数归一化量 | 无量纲 | log-softmax |
| \(A^*\) | teacher adjacency | 完整教师路线的双向邻接计数；星号指教师参照 | \(101\times101\) | 训练监督 |
| teacher | teacher/reference route | 经冷链目标评价并保存为训练监督的可行路线 | 标签路线 | 训练监督 |
| \(Y_{ij}\) | target probability | 对教师邻接计数逐行归一化后的目标分布 | 无量纲 | 训练监督 |
| \(w_i\) | row weight | CE中客户行为1、仓库行为0 | 无量纲 | 损失加权 |
| \(\mathcal L_{\mathrm{CE}}\) | cross-entropy loss | 教师邻接分布与模型预测间的加权交叉熵 | 自然对数尺度 | 训练目标 |
| \(\eta\) | eta, learning rate | AdamW学习率 \(10^{-4}\) | 优化器数值参数 | 参数更新 |
| AdamW | adaptive optimizer with weight decay | 实际使用的优化器，权重衰减 \(10^{-4}\) | 算法 | 参数更新 |
| 梯度裁剪 | gradient clipping | 若整组梯度的全局范数超过1，按比例缩小梯度，再交给优化器更新参数 | 优化步骤 | 稳定参数更新 |

注意力里的 \(V^{(h)}\) 是“值向量矩阵”，与货物的经济价值或配送节点集合 \(V\) 无关。教师邻接 \(A^*\) 上的星号也不代表已经证明全局最优。锁版只有邻接交叉熵训练；完整冷链目标用于教师路线评价和在线搜索评价。[网络与损失](D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_neural.py:40)。

## 4. 热图校准与普通重构搜索

| 符号或术语 | 读法或英文 | 含义 | 单位 | 首次出现概念 |
|---|---|---|---|---|
| \(D_{ij}\)（边分） | distance score | 纯距离排序分 \(-d_{ij}\)，越大越近 | km的负数值 | 距离引导 |
| \(r^D_{ij},r^H_{ij}\) | distance/heatmap rank | 从节点 \(i\) 看，节点 \(j\) 在距离或热图中的排名 | 秩次 | 距离校准 |
| \(R_{ij}\) | rank residual | \(\operatorname{clip}(\log(r^D_{ij}/r^H_{ij}),-3,3)\) | 无量纲 | 距离校准 |
| \(s\)（校准中） | distance scale | 按各行第9小距离取中位数得到的尺度 | km | 距离归一化 |
| \(S_{ij}\) | spatial score | \(-d_{ij}/s\) | 无量纲 | 距离校准 |
| \(\alpha\) | alpha, residual weight | 冻结为0.25的秩残差系数 | 无量纲 | 距离校准 |
| \(C_{ij}\)（边分） | calibrated score | 校准热图 \(S_{ij}+\alpha R_{ij}\) | 无量纲 | 校准候选 |
| \(h_{ij}\)（泛称） | selected edge score | 当前选用的距离、原始或校准边分 | 随所选分数而定 | 插位排序 |
| \(\Delta d(a,i,b)\) | insertion detour | 将客户 \(i\) 插入边 \((a,b)\) 的新增里程 | km | 插位筛选 |
| \(\Delta h(a,i,b)\) | edge-score change | 插入前后边分之差 | 随边分而定 | 插位筛选 |
| \(r\oplus(i,\mathrm{pos})\) | insertion operation | 在路线指定位置加入客户 \(i\) | 路线操作 | 精确插位评价 |
| \(\Delta u\) | marginal route benefit | 插入后的路线净效益减原路线净效益 | U | 插位选择 |
| greedy | greedy insertion | 按当前客户顺序，每次选择精评后的最好插位或单客车 | 算法 | 普通重构 |
| \(g_i^{(1)},g_i^{(2)}\) | best/second-best insertion gain | 客户 \(i\) 最好、次好的可选真实净增量 | U | regret-2 |
| \(g_i^{(1)}-g_i^{(2)}\) | regret-2 | 错过最好位置后会损失多少，优先处理遗憾大的客户 | U | regret-2 |
| `state` | current search state | 可以继续变动的当前可行车队 | 车队 | 搜索状态 |
| `best` | best-candidate buffer | 原生搜索维护的最好候选缓存 | 车队 | 搜索状态 |
| `clock.events` | accepted incumbent history | 严格评价已完成且预算内接纳的正式最好记录 | 记录序列 | 在线接纳 |
| \(n_m\) | mode-use count | 破坏模式 \(m\) 已被使用的次数 | 次 | 自适应破坏 |
| \(R_m\) | mode reward | 破坏模式的平滑收益分数 | 变换后的数值分数 | 自适应破坏 |
| \(t\)（搜索中） | elapsed online time | 从本次在线求解开始计的墙钟时间 | 秒 | 在线预算 |
| \(\theta(t)\) | theta, temperature | 接受下降解的温度，主搜索为 \(2.5\max(0.1,1-t/60)\) | U尺度 | 搜索接受 |
| \(U_{\mathrm{new}},U_{\mathrm{current}},U_{\mathrm{best}}\) | new/current/best benefit | 接受公式所比较的新解、当前解和最好候选净效益 | U | 搜索接受 |

训练掩码20客、每轮网络前向池40客、一次greedy重构移除6–17客、regret-2移除10客，分别描述不同步骤。0.04秒是一个短试选的预算，1.6秒是外层时段调度边界，60秒是整次在线求解预算；它们均使用真实计算秒数，与配送分钟数不同。[搜索时段与六项选择](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/pool_search.py:139)。

## 5. Split动态规划与FLEET_PRICE主问题

| 符号或术语 | 读法或英文 | 含义 | 单位 | 首次出现概念 |
|---|---|---|---|---|
| \(\pi=(\pi_1,\ldots,\pi_m)\) | pi, customer permutation | Split分车前固定的客户访问顺序 | 有序编号 | 访问顺序提案 |
| \(m\)（Split中） | sequence length | 本次分车组合的客户数，最多24 | 客户数 | 局部路线组合 |
| \(i,j\)（DP中） | boundary positions | 访问顺序中的切分边界，0到 \(m\) | 位置索引 | 连续分段 |
| \(g(i,j)\) | segment value | 客户子段 \(\pi_{i+1},\ldots,\pi_j\) 作为一辆车时的净效益 | U | Split弧权 |
| \(-\infty\) | negative infinity | 不可行子段的标记，使其不能赢得最大化选择 | 数学标记 | DP可行性 |
| \(F(j)\) | DP state | 覆盖前 \(j\) 位客户的最佳连续分车净效益 | U | 动态规划 |
| \(\mathrm{prev}(j)\) | predecessor | 达到 \(F(j)\) 时记录的前一切分位置 | 位置索引 | 路线回溯 |
| DP | dynamic programming | 利用子问题状态递推求最优连续分段 | 算法 | 固定顺序分车 |
| DAG | directed acyclic graph | 顶点为切分位置、弧为可行子段的有向无环图 | 图结构 | Split等价图模型 |
| \(\mathcal R_0\) | initial fleet | FLEET_PRICE前缀开始时的共同初解 | 车队 | 全车队前缀 |
| \(M\) | masked customer set | FLEET_PRICE前缀的40位遮蔽客户 | 集合 | 联合重构 |
| \(s_k\)（骨架） | visible skeleton | 从原路线去掉遮蔽客户后保留的可见客户原顺序 | 有序序列 | 骨架保持 |
| \(J\) | number of skeletons | 非空可见骨架条数 | 条 | 骨架保持 |
| \(\mathcal I\) | coverage-item set | 40个遮蔽客户项和 \(J\) 个完整骨架项 | 集合 | 集合划分 |
| \(\mathcal C_0,\mathcal C_1\) | common/expanded column pool | 共同路线列池和扩展后的路线列池 | 集合 | 路线候选池 |
| column | route column | 一条完整可行路线，附带覆盖关系和真实价值 | 候选路线 | 集合划分 |
| \(a_{ir}\) | coverage coefficient | 路线列 \(r\) 是否覆盖项 \(i\)，是为1、否为0 | 0或1 | 覆盖矩阵 |
| \(b_i\)（基准项价值） | item baseline value | 遮蔽客户单客路线或纯骨架路线的净效益 | U | 差额成本转换 |
| \(B\)（基准总值） | total baseline | \(\sum_i b_i\) | U | 差额成本转换 |
| \(u_r\) | route-column benefit | 某条路线列的真实净效益，与 \(u(r)\) 同义 | U | 列评价 |
| \(c_r\) | transformed column cost | \(\sum_i a_{ir}b_i-u_r\) | U | 主问题目标 |
| \(x_r\) | route-selection variable | 整数主问题是否选择路线 \(r\) | 0/1；LP松弛时非负实数 | 主问题变量 |
| LP | linear programming | 将整数选择变量放松后的线性规划 | 优化问题 | 取得对偶价格 |
| MIP | mixed-integer programming | 通称混合整数规划；本主问题实际为0–1整数线性规划 | 优化问题 | 可实施车队选择 |
| \(y_i\)（对偶） | dual variable / shadow price | 当前LP覆盖等式的对偶变量 | U尺度 | 约束边际价值 |
| \(\pi_i=-y_i\)（价格） | stored price | 代码保存的负对偶变量 | U尺度 | 价格符号约定 |
| \(\bar c_r\) | reduced cost | \(c_r-\sum_i a_{ir}y_i\)，负值表示改进当前LP的潜力 | U | 定价 |
| \(g_r=-\bar c_r\) | pricing branch score | 定价束搜索保留分支时使用的反向约化成本分数 | U | 束搜索 |
| \(d^{\mathrm{save}}_{i,a,b}\) | distance-saving proxy | 相对单客往返，把客户插入现有路线所节省的距离分 | km | 定价插位筛选 |
| \(h^{\mathrm{save}}_{i,a,b}\) | learned edge-saving proxy | 相对单客往返的原始logits边分变化 | 无量纲 | 神经定价筛选 |
| \(\rho(\cdot)\) | rho, normalized rank | 升序平均秩除以当前候选数量 | 无量纲 | 定价优先级 |
| \(\mathrm{priority}_{i,a,b}\) | insertion priority | 距离、神经分和价格的归一化排序组合 | 无量纲分数 | 精评候选筛选 |
| pricing | route pricing | 搜索值得加入主问题的新路线列 | 算法步骤 | 启发式列扩展 |
| beam width | beam-search width | 每层继续扩展的分支数，本实现为2 | 分支数 | 有界定价搜索 |
| \(L\)（MIP界） | dual bound on cost | 求解器对当前列池最小差额成本给出的下界 | U | 列池内最优性界 |
| \(U_{\mathrm{chosen}}\) | adopted fleet value | 整数求解最终采用车队的净效益 | U | 整数解接纳 |
| \(\mathrm{gap}_U\) | absolute benefit gap | \(\max(0,B-L-U_{\mathrm{chosen}})\)，只针对当前列池 | U | 列池内差距 |

Split使用固定访问顺序上的连续子段，FLEET_PRICE从显式路线列池中做整数选择。两者的候选空间不同：固定顺序DP可给出该顺序内的最优分段；有限列池MIP的界只适用于该列池。启发式访问顺序和启发式列扩展都不会因此自动获得全问题最优证书。[Split实现](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver/split_order.cpp:3)；[FLEET_PRICE主问题](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:56)。

## 6. 同字母复用时怎样避免读错

| 易混符号 | 区分方法 |
|---|---|
| \(U\)、\(u\)和单位U | \(u(r)\) 或 \(u_r\) 是一条路线的净效益；\(U(\mathcal R)\) 是车队净效益；数值后面的U是研究货币单位，例如“\(U=100\ \mathrm U\)”中的前后两个U角色不同。 |
| \(K\)和K | 公式中的 \(K=|\mathcal R|\) 表示车辆数；“275 K”中的K是开尔文温标单位。注意力 \(K^{(h)}\) 则是键矩阵。 |
| \(Q\)、\(\dot Q\)、\(Q^{(h)}\) | \(Q\) 是kg载重上限；\(\dot Q\) 是W热负荷；\(Q^{(h)}\) 是注意力查询矩阵。上点和头索引不能省略。 |
| \(m_i\)、\(m\)和m | \(m_i\) 是客户质量；Split中的 \(m\) 是局部序列客户数；“0.1 m”中的m是米。 |
| \(A\)、\(A_i\)、\(A^{\mathrm{vis}}\) | 无下标的物理 \(A\) 是壁面积；\(A_i\) 是客户到达时刻；带 `vis` 的 \(A\) 是可见邻接矩阵。 |
| \(B_i\)和 \(B\) | \(B_i\) 是服务开始时刻；FLEET_PRICE中的 \(B\) 是所有覆盖项基准净效益之和。 |
| \(C_i\)、\(C_{ij}\)和 \(\mathcal C\) | 单下标是服务完成时刻；双下标是校准边分；花体是候选路线列池。 |
| \(b_i\) | 时间模型中是客户时窗上界；FLEET_PRICE中是覆盖项基准价值。跨章节推导时可写成 \(b_i^{\mathrm{TW}}\) 与 \(b_i^{\mathrm{base}}\)。 |
| \(x_i,y_i\)与 \(x_r,y_i\) | 配送图中的 \((x_i,y_i)\) 是坐标；主问题中的 \(x_r\) 是路线选择变量、\(y_i\) 是覆盖项对偶变量。先看所在模型。 |
| \(\pi\) | 出库公式的 \(\pi_k\) 是分钟前缀；Split的 \(\pi\) 是客户排列；FLEET_PRICE的 \(\pi_i\) 是U尺度负对偶价格。不能在同一推导中混用。 |
| \(s_i\)与 \(s_k\) | 时间模型中 \(s_i\) 是服务分钟数；FLEET_PRICE中的 \(s_k\) 是可见客户骨架序列；校准中的单独 \(s\) 是km距离尺度。 |
| \(z_i\) | 输入可见标记是0/1标量；输出头中的 \(z_i\) 是256维向量。可分别记 \(z_i^{\mathrm{vis}}\) 与 \(z_i^{\mathrm{emb}}\)。 |
| \(g_i^{(1)}-g_i^{(2)}\)、\(g(i,j)\)、\(g_r\) | 分别为插位遗憾、DP子段净效益、定价的负约化成本；都可能以U表示，但比较对象不同。 |
| \(R_g\)、\(R_{ij}\)、\(R_m\) | 分别为气体常数、边排名残差、破坏模式回报；量纲和下标均不同。 |
| \(L_i\)、\(L\)、\(\mathcal L_{\mathrm{CE}}\) | 分别是最晚服务开始时刻、MIP成本下界和交叉熵损失。 |
| \(t\)与 \(\tau\) | 搜索 \(t\) 以秒计；配送时间 \(t_r^0\) 以min计；网络 \(\tau\) 是无量纲可见比例。 |
| \(\delta\)与 \(\Delta\) | 小写delta \(\delta\) 是壁厚；大写delta \(\Delta\) 表示变化量，如 \(\Delta U\)。 |

公式中的 \(\log\) 默认自然对数，\(\exp\) 是以自然常数为底的指数函数。\(\arg\max\) 返回取得最大值的选择或索引，\(\max\) 返回最大数值。\(|M|\) 对集合表示元素个数，\(|\Delta U|\) 对数值表示绝对值。\(\mathsf T\) 是矩阵转置，既不是温度也不是时间。\(\sum\) 表示求和，\(\in\) 表示属于某集合，\(\forall\) 表示“对每一个”，\(\mathbb R\) 表示实数集合，\(\operatorname{clip}(x,a,b)\) 表示把数值截在区间 \([a,b]\) 内；\(\tanh\) 是双曲正切函数。

## 7. 希腊字母与数学运算怎样读

| 符号或写法 | 中文读法 | 当前上下文中的意思 |
|---|---|---|
| \(\alpha\) | 阿尔法 | 校准热图中的秩残差系数，冻结为0.25。 |
| \(\kappa\) | 卡帕 | 按分钟计的品质衰减率。 |
| \(\tau\) | 陶 | 网络输入的可见邻接比例。 |
| \(\pi\) | 派 | 分别用于时间前缀、客户排列或负对偶价格，须看所属公式；这里通常不是圆周率。 |
| \(\theta\) | 西塔 | 搜索接受下降解时使用的温度。 |
| \(\lambda\) | 兰姆达 | 壁面导热系数。 |
| \(\delta\) | 小写德尔塔 | 车厢壁厚。 |
| \(\rho\) | 柔（rho） | 定价筛选中的归一化平均秩。 |
| \(\Delta\) | 大写德尔塔 | 变化量或差额，例如“德尔塔U”表示净效益差。 |
| \(\Sigma\)、\(\sum\) | 大写西格玛；求和 | \(\Sigma\) 是字母；公式中的 \(\sum_i\) 是由该字母形状形成的求和运算符，读“对所有i求和”。 |
| \(\in\) | 属于 | 如 \(i\in M\) 表示客户i属于本轮遮蔽集合。 |
| \(\forall\) | 对任意一个／对所有 | 表示约束对每个覆盖项或客户都成立。 |
| \(\odot\) | 逐元素乘／哈达玛积 | SwiGLU两条同形状分支对应位置相乘。 |
| \(\oplus\) | 插入操作（本文约定） | \(r\oplus(i,\mathrm{pos})\) 表示在路线r指定位置插入客户i。 |
| \(|M|\)、\(|x|\) | 集合M的大小；x的绝对值 | 竖线包围集合时数元素，包围实数时取绝对值。 |
| \(m_i\)、\(a_{ir}\) | m下标i；a下标i、r | 下标区分客户、覆盖项、路线或位置；本身不是乘法。 |
| \(t_r^0\)、\(Q^{(h)}\)、\(m^3\) | t下标r上标零；第h头的Q；m的三次方 | 上标可能是标签，也可能是幂；出库的0和注意力头的h是标签，3才是指数。 |
| \(A^*\) | A星 | 指教师参照邻接，并不表示已证全局最优。 |
| \(K^{\mathsf T}\) | K的转置 | 把矩阵的行与列交换；上标T不表示温度。 |
| \(\log x\) | x的自然对数 | 正文使用以自然常数e为底的对数。 |
| \(\exp(x)\) | e的x次方／指数函数 | 用于Arrhenius速率、概率归一化和搜索接受概率。 |
| softmax | 软最大／归一化指数 | 将一行分数变成非负且总和为1的权重。 |
| log-softmax | 软最大后的对数 | 直接计算逐行归一化权重的对数，得到热图H。 |
| logsumexp | 对数—指数和 | 先对各分数取指数、求和、再取对数；它是一行的对数归一化量。 |
| \(\arg\max\) | 使值最大的自变量或索引 | 返回“选哪一个”，而不是最大值本身。 |
| \(\operatorname{clip}(x,a,b)\) | 把x截在a到b之间 | 小于a的取a，大于b的取b，中间保留原值。 |
| \(O(m^3)\) | 大O，m的三次方／立方量级 | 当前朴素Split实现的最坏时间增长上界；m是局部客户数，不是运行秒数。 |

## 8. 两客户路线的完整手算

**本节数值专供手算教学，不属于任何实验数据或案例。**这里只在文稿中给出一个缩小到两客户的配送问题，用来解释公式；它不对应正式100客输入，也不用于评价任何算法。动力学、车速、容量和费用采用项目实际主参数。

### 8.1 已知条件

主参数为 \(Q=20000\) kg、\(v=1\) km/min、\(p=5/3\) U/kg、\(c_d=1\) U/km、\(c_f=0.0972\) U/(车·min)、\(q_{\min}=0.8\)。恒温 \(T=T_{\mathrm{ref}}=275\) K，所以

\[
\kappa=60\times0.00000308=0.0001848\ \mathrm{min}^{-1}.
\]

仓库开放区间取 \([0,800]\) min。两位客户的教学条件如下。

| 客户 | 配送质量 | 服务开始时窗 | 服务时长 |
|---|---:|---:|---:|
| 1 | 4,000 kg | [100,300] min | 90 min |
| 2 | 6,000 kg | [500,600] min | 90 min |

给定对称距离 \(d_{01}=40\) km、\(d_{12}=30\) km、\(d_{20}=50\) km。考虑既定路线 \(0\rightarrow1\rightarrow2\rightarrow0\)，只进行数值评价。载重为10,000 kg，低于20,000 kg。

### 8.2 求规范化出库时刻

忽略等待的服务开始前缀为

\[
\pi_1=40,\qquad \pi_2=40+90+30=160.
\]

由客户1时窗得到 \(t_r^0\le300-40=260\)；由客户2时窗得到 \(t_r^0\le600-160=440\)；由回仓上界得到
\(t_r^0\le800-(160+90+50)=500\)。仓库本身要求 \(t_r^0\le800\)，于是

\[
t_r^0=\min\{800,260,440,500\}=260\ \mathrm{min}.
\]

260分钟位于仓库开放区间内。这个最晚出库由客户1的最迟服务开始时刻决定。

### 8.3 正向重建到达、等待和交付

| 事件 | 计算 | 时刻或时长 |
|---|---|---:|
| 出库 | 上一步结果 | 260 min |
| 到达客户1 | \(260+40/1\) | 300 min |
| 开始服务客户1 | \(\max(300,100)\) | 300 min |
| 客户1等待 | \(300-300\) | 0 min |
| 完成客户1交付 | \(300+90\) | 390 min |
| 到达客户2 | \(390+30/1\) | 420 min |
| 开始服务客户2 | \(\max(420,500)\) | 500 min |
| 客户2等待 | \(500-420\) | 80 min |
| 完成客户2交付 | \(500+90\) | 590 min |
| 回仓 | \(590+50/1\) | 640 min |

两位客户的服务开始时刻300、500均在各自时窗中，回仓640≤800。客户1完成服务时刻390超过其窗口上界300仍然可行，因为这里限制的是服务开始时刻。

路线总距离为 \(40+30+50=120\) km；制冷持续时间为 \(640-260=380\) min，也等于120分钟行驶＋180分钟服务＋80分钟等待。

### 8.4 计算交付品质

两位客户相对本车出库的交付货龄分别为

\[
\mathrm{age}_1=390-260=130,\qquad
\mathrm{age}_2=590-260=330\quad(\mathrm{min}).
\]

因此

\[
q_1=1-0.0001848\times130=0.975976,
\qquad q_2=1-0.0001848\times330=0.939016.
\]

两个品质均超过0.8；相应货龄也均小于主参数允许的约1082.2511分钟。客户2的80分钟等待已经计入其330分钟货龄。

### 8.5 计算价值、费用和路线净效益

品质折算交付价值为

\[
\begin{aligned}
\mathrm{Value}
&=\frac53\left(4000\times0.975976+6000\times0.939016\right)\\
&=\frac53(3903.904+5634.096)\\
&=15896.666666\ldots\ \mathrm U.
\end{aligned}
\]

行驶费用为 \(120\times1=120\) U，制冷费用为
\(380\times0.0972=36.936\) U。于是

\[
u(1,2)=15896.666666\ldots-120-36.936
=\boxed{15739.730666\ldots\ \mathrm U}.
\]

也可从等价损失方向核对：完全保质的名义价值为
\(10000\times5/3=16666.666666\ldots\) U，品质损失为770 U。扣除品质损失770 U、行驶120 U和制冷36.936 U，得到相同结果。品质损失不再额外从已折算过品质的交付价值中扣第二次。

### 8.6 同样两位客户分成两辆车时会怎样

另取两条给定路线 \(0\rightarrow1\rightarrow0\) 和 \(0\rightarrow2\rightarrow0\)，逐条按同样规则计算；本小问题的覆盖条件是两位客户各出现一次。

| 指标 | 客户1单客车 | 客户2单客车 |
|---|---:|---:|
| 最晚出库时刻 | \(\min(300-40,800-40-90-40)=260\) | \(\min(600-50,800-50-90-50)=550\) |
| 服务开始/交付时刻 | 300 / 390 min | 600 / 690 min |
| 回仓时刻 | 430 min | 740 min |
| 交付货龄 | 130 min | 140 min |
| 交付品质 | 0.975976 | 0.974128 |
| 路线距离 | 80 km | 100 km |
| 制冷持续时间 | 170 min | 190 min |

两辆车的质量、品质、服务开始时窗和回仓都满足约束。客户2单客车虽到600分钟才开始服务，其货物也到550分钟才出库，因此货龄反而较短。

两车合计交付价值为

\[
\frac53\left(4000\times0.975976+6000\times0.974128\right)
=16247.786666\ldots\ \mathrm U.
\]

合计里程费用为180 U，制冷费用为
\((170+190)\times0.0972=34.992\) U，所以

\[
U_{\mathrm{two}}=16247.786666\ldots-180-34.992
=16032.794666\ldots\ \mathrm U.
\]

相对一车方案，净效益差为

\[
\Delta U=16032.794666\ldots-15739.730666\ldots
=\boxed{293.064\ \mathrm U}.
\]

这个差额可以逐项解释：更好的客户2品质增加交付价值351.12 U，额外里程增加费用60 U，而减少等待使总制冷时间少20分钟、节省1.944 U，合计 \(351.12-60+1.944=293.064\) U。该教学对照展示了本模型的分车取舍：车多不必然总制冷时间更长；是否合算应由交付价值、全部路线距离和出库后持续时间共同决定。

## 9. 五个核心概念问答

**1. 交叉熵更低，就意味着冷链净效益更高吗？**

不意味着。交叉熵衡量模型恢复教师邻接的程度；净效益还取决于候选路线怎样构造、约束怎样影响可行性，以及网络计算占用了多少搜索时间。最后应看统一预算下的 U 和同规则消融。

**2. 热图是不是“预测哪条路线最赚钱”的概率图？**

热图 \(H\) 是逐行邻接偏好的对数分数，\(\exp(H_{ij})\) 才是该行归一化后的概率。它不直接预测整条路线的净效益，也不保证容量和品质可行。候选路线要经过完整冷链评价后才决定是否采用。

**3. 为什么有20客、40客和6–17客/10客几种掩码数量？**

20客是训练时隐藏邻接的数量；40客是在线一次网络前向隐藏的候选池；6–17客或10客是一次普通重构实际从路线中移除并插回的数量。同一张40客热图被多次较小重构共用，从而分摊预测费用。

**4. 有DP和MIP，是否说明整个问题已经精确求解？**

DP精确处理的是给定访问顺序的连续分车。MIP优化的是已经生成的有限路线列池，而且还受实际时间和节点上限约束。访问顺序、候选列均由启发式生成，这些局部精确组件不能直接推出整个100客问题的全局最优性。

**5. 为什么不直接追求车辆最少或总距离最短？**

主任务最大化品质折算交付价值减行驶和制冷费用。过度合车可能延长高价值货物的货龄；适当分车也可能通过减少等待降低总制冷时长。反过来，分车过多会增加仓库往返里程。手算中的293.064 U来自具体权衡，不代表所有实例都会从增加车辆中受益。
