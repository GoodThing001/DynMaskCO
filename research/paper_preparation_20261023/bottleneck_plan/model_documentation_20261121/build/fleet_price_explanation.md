# FLEET_PRICE：全车队候选路线定价扩展

## 1. 与锁版 ORDER_SPLIT 的关系

FLEET_PRICE 是在锁版 ORDER_SPLIT 之前增加一次全车队路线组合优化的相关扩展。两者使用相同的静态冷链业务目标和硬约束，区别在于如何组织一次较大范围的重构。

ORDER_SPLIT 在搜索过程中反复选取2–4条近邻路线、合计不超过24位客户，先生成访问顺序，再对固定顺序做连续分车。FLEET_PRICE 则在共同初解得到后，均匀随机遮蔽40位客户，允许这些客户进入车队中所有保留下来的路线骨架或组成新的路线；它把候选路线交给一个整数选择问题，同时决定应保留哪些路线。该前缀只执行一次，之后使用剩余时间继续 ORDER_SPLIT。前缀与后续搜索共用单物理核60秒在线预算。

| 方法 | 开局路线组合前缀 | 前缀之后的搜索 |
|---|---|---|
| ORDER_SPLIT | 无 | ORDER_SPLIT |
| FLEET_PRICE | 距离、MASKCO边分及LP价格引导 | ORDER_SPLIT |
| PREFIX_DISTANCE | 距离及相同LP价格引导 | ORDER_SPLIT |
| FLEET_PRICE_D | 距离及相同LP价格引导 | ORDER_SPLIT_D |

`PREFIX_DISTANCE` 用于分离前缀中的神经边分作用，因为它与 FLEET_PRICE 的后续搜索相同。`FLEET_PRICE_D` 同时移除前缀和后续搜索中的神经引导，其差额评价的是整个扩展中的学习引导净作用。[实际四法分支](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_timed_worker.py:58)。

## 2. 骨架、覆盖项与“列”是什么意思

给定初解车队 \(\mathcal R_0\)，设本轮遮蔽客户集合为 \(M\)，\(|M|=40\)。从每条原路线中删去 \(M\) 内客户，剩余可见客户的原有顺序称为一条**骨架**，记为 \(s_1,\ldots,s_J\)；空骨架不保留。这里“删去”只用于构造候选路线，客户仍必须在最终车队中恰好配送一次。

主问题的每个覆盖项是“一个遮蔽客户”或“一整条非空骨架”。因此共有 \(40+J\) 个覆盖项，记集合为 \(\mathcal I\)。把整条骨架当成一个覆盖项，意味着其可见客户不能在本轮拆散，也不能改变相对顺序；遮蔽客户可以插在骨架的各个空隙中。一个候选路线最多包含一条完整骨架，也可以仅包含遮蔽客户。[骨架与覆盖项定义](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:9)。

**列（column）**在这里就是一条通过冷链可行性评价的候选路线 \(r\)。它不是表格中的任意数据列，而是整数主问题可选择的一个完整配送方案片段。每列带有三项信息：路线的有序客户序列、覆盖哪些项、这条路线的真实净效益 \(u_r\)。定义覆盖系数

\[
a_{ir}=\begin{cases}1,&\text{路线 }r\text{ 覆盖项 }i,\\0,&\text{否则。}\end{cases}
\]

对于遮蔽客户，覆盖指这位客户在路线中出现；对于骨架，覆盖指其所有可见客户以原顺序完整出现。两条路线若覆盖完全相同的项，则只保存净效益较大的那一个。它们对主问题约束的作用相同，保留较差路线不会改善目标。[覆盖映射与同覆盖去重](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:23)。

## 3. 共同列池：神经扩展之前有什么

所有前缀变体先构造相同的共同列池 \(\mathcal C_0\)，包括：原车队路线、各条纯可见骨架、各遮蔽客户的单客路线、所有遮蔽客户有序两客路线、以及每位遮蔽客户插入每条骨架各位置形成的路线。候选必须经过统一路线评价，违反容量、时窗、品质或回仓要求的路线不加入列池。同一覆盖集合只保留最好访问顺序。

共同列池本身已经包含大量全车队重分配选择，而且保留原车队作为可行基准。MASKCO 在随后扩展多客路线时发挥引导作用；它不改变已经登记路线的真实收益，也不替代共同列池。[共同列构造](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:42)。

## 4. 主问题为什么用“成本”，却仍在最大化净效益

为每个覆盖项定义基准价值 \(b_i\)：遮蔽客户的 \(b_i\) 是其单客路线净效益；骨架项的 \(b_i\) 是该骨架单独作为路线的净效益。令

\[
B=\sum_{i\in\mathcal I}b_i,\qquad
c_r=\sum_{i\in\mathcal I}a_{ir}b_i-u_r.
\]

\(B\) 是“每个遮蔽客户单独配送、每条骨架独立配送”的基准总净效益，\(c_r\) 是选择路线 \(r\) 相对其所覆盖项基准价值的差额成本。\(c_r\) 可以为负：负值表示把这些项放在同一条路线里，比其分别配送更有利。

给定当前候选列池 \(\mathcal C\)，令二元变量 \(x_r\in\{0,1\}\) 表示是否选择路线 \(r\)。整数主问题为

\[
\min_x\ \sum_{r\in\mathcal C}c_rx_r,
\qquad
\sum_{r\in\mathcal C}a_{ir}x_r=1\quad(\forall i\in\mathcal I),
\qquad x_r\in\{0,1\}.
\]

每条等式要求一个客户或一条骨架被恰好覆盖一次。所有覆盖项都选满后，

\[
\sum_r c_rx_r=B-\sum_r u_rx_r.
\]

因此最小化差额成本，与最大化被选路线的车队净效益完全等价。这是**集合划分模型**：从许多可行路线中选出一组互不重复、完整覆盖所有客户的路线。[矩阵和成本构造](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:56)。

**MIP**指混合整数规划；本实现的这个具体主问题所有选择变量都是0/1，所以更具体地说是一个0–1整数线性规划。求解器使用 HiGHS，单线程，最多256个分支节点、5秒，目标相对间隙参数设为0。设为0表示希望找到该有限列池的精确最优解；时间或节点上限仍可能先到达。程序只有在得到的整数可行车队优于传入的可行基准时才采用它；没有更好整数解时保留基准。[整数求解和基准保留](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:72)。

## 5. LP、对偶价格和约化成本

### 5.1 LP 是用于取得价格的连续松弛

**LP**即线性规划。把上面的 \(x_r\in\{0,1\}\) 放松成 \(x_r\ge0\)，保留相同的目标与覆盖等式，得到当前列池的 LP 松弛。LP 可以用0.3条某路线加0.7条另一条路线的数学组合覆盖客户，所以它的解一般不能直接作为真实配送车队。它提供的是当前有限列池的连续最优目标和覆盖约束的对偶信息。实现对共同列池只求解一次 LP。[LP调用](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:64)。

### 5.2 “价格”是覆盖约束的边际价值

令 LP 覆盖等式的对偶变量为 \(y_i\)。对应的对偶问题为

\[
\max_y\ \sum_i y_i,\qquad
\sum_i a_{ir}y_i\le c_r\quad(\forall r\in\mathcal C_0).
\]

\(y_i\) 可理解为在当前列池和当前最优基附近，覆盖要求右端发生小幅变化时，最小差额成本怎样变化的影子价格。它是数学模型中的边际量，不是客户真实支付的报价。等式对偶变量不要求非负。

SciPy 返回的 `lp.eqlin.marginals` 对应这里的 \(y\)，代码将其取负后保存为

\[
\pi_i=-y_i.
\]

所以阅读 `prices` 变量时必须保留这个负号；直接把它当成 \(y_i\) 会把后续定价方向解释反。[价格符号](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:70)。

### 5.3 约化成本衡量新路线对 LP 的改进潜力

对新候选路线 \(r\)，**约化成本（reduced cost）**为

\[
\bar c_r=c_r-\sum_i a_{ir}y_i
=\sum_i a_{ir}b_i-u_r+\sum_i a_{ir}\pi_i.
\]

对于最小化问题，负的 \(\bar c_r\) 表示该列具有改进当前 LP 的潜力。实现以相反方向的分数排序束搜索分支：

\[
g_r=u_r-\sum_i a_{ir}b_i-\sum_i a_{ir}\pi_i=-\bar c_r.
\]

因此 \(g_r\) 越大，代表约化成本越小。这个分数与最终的真实路线净效益 \(u_r\) 不同：它还考虑了路线所占用覆盖项的基准价值及对偶价格。[真实价值与定价分支分数](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:123)。

## 6. 本实现怎样产生新列

### 6.1 定价在这里是有界启发式搜索

**定价（pricing）**指在尚未加入主问题的路线中寻找值得加入的新列，并不是给货物确定售价。本实现分别从每条可见骨架、每位遮蔽客户的单客路线出发，向其中逐步插入其他遮蔽客户。

每个锚点最多向下扩展4层；每个当前分支只精确尝试优先级最高的3个“客户—插位”组合，每层仅保留分数最高的2条分支继续扩展。因此新增骨架列最多插入4名遮蔽客户；纯遮蔽客户列从1客锚点开始，最多形成5客路线。共同列池保留的原始路线不受这个新增列深度限制。纯遮蔽路线只加入编号大于锚点的客户，使该锚点成为路线中的最小编号客户，减少等价起点的重复。

容量先作快速筛选，完整路线可行性和价值仍由统一冷链评价器计算。所有精确尝试后可行的候选都可加入列池；\(g_r\) 用来选择后续束分支，不是“只有负约化成本列才可入池”的硬过滤。[锚点、深度、候选和束宽](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:97)。

### 6.2 距离、神经边分和价格如何共同选插位

设当前路线的插入空隙为 \((a,b)\)，拟插入客户为 \(i\)。距离优先分为

\[
d^{\mathrm{save}}_{i,a,b}=2d_{0i}-d_{ai}-d_{ib}+d_{ab}.
\]

它比较“客户 \(i\) 单独往返仓库”与“把它插入现有路线”所需里程的差，数值越大表示潜在节省越多。

神经前缀直接调用 `predict` 取得原始 logits \(\ell\)，构造

\[
h^{\mathrm{save}}_{i,a,b}
=\ell_{ai}+\ell_{ib}-\ell_{ab}-\ell_{0i}-\ell_{i0}.
\]

这个量比较“插入现有路线”与“保留单客仓库往返”在预测边分上的差。它没有使用 ORDER_SPLIT 普通重插中的校准热图。令 \(\rho(\cdot)\) 表示在当前候选数组内的升序平均秩除以数组长度，值大代表排名靠前：相等值共享平均秩。对全部当前可选客户和插位展平排序后，优先级为

\[
\mathrm{priority}_{i,a,b}=
\begin{cases}
\tfrac12\rho(d^{\mathrm{save}}_{i,a,b})+\tfrac12\rho(h^{\mathrm{save}}_{i,a,b})
+\tfrac12\rho(-\pi_i), &\text{神经前缀},\\
\rho(d^{\mathrm{save}}_{i,a,b})+\tfrac12\rho(-\pi_i), &\text{纯距离前缀}.
\end{cases}
\]

价格项只按候选客户排序，再复制到该客户的所有可选插位。由于 \(-\pi_i=y_i\)，更大的这个项倾向于优先处理 LP 对偶意义下更有边际改进价值的覆盖要求。以上分数用于把众多插位缩减到3个精确尝试，不等于真实净效益，也不等于完整新路线的约化成本；完整路线形成后再计算 \(u_r\) 和 \(g_r\)。[原始网络输出](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_timed_worker.py:27)；[三类优先级](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:93)。

## 7. 一次前缀的完整执行顺序与可证明范围

一次前缀先构造共同列池 \(\mathcal C_0\)，求其 LP 得到固定价格，再求共同列池的整数主问题，获得基准车队。接着用这些价格和相应边分扩展路线列，得到 \(\mathcal C_1\supseteq\mathcal C_0\)，再求一次扩展后的整数主问题，以共同列池的车队作为可行保底。只有更好的整数车队才替代保底。[前缀执行顺序](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_timed_worker.py:41)。

该方法是“LP价格引导的有限路线扩展＋整数集合划分”。标准精确定价式列生成通常反复执行“解受限主问题—求解定价问题—加入负约化成本列”，直到证明不存在改进列。本实现只计算一轮共同列池价格，束搜索也不会穷举全部可行路线；扩展后没有再次求价格。因此不能把它描述为完成了收敛证明的列生成，也不能声称获得全车队所有路线空间的全局最优解。

若整数求解器给出差额成本的对偶下界 \(L\)，代码记录的绝对净效益间隙为

\[
\mathrm{gap}_U=\max\{0,B-L-U_{\mathrm{chosen}}\}.
\]

其中 \(U_{\mathrm{chosen}}\) 为最终采用车队的真实净效益，\(B-L\) 是这个有限列池内净效益的上界。该量只针对当前列池，不是全体可能配送路线的全局最优误差；缺少求解器界时记为空值。[列池内间隙](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_pricing.py:87)。

前缀整数求解分别具有5秒局部上限，LP及候选扩展没有逐步接入60秒截止回调；是否成为正式解仍由统一在线接纳器决定。前缀完成后调用 `clock.offer`，只有严格复算完成时间处于60秒内且净效益超过当前正式最好记录，才同时替换搜索的 `state` 和 `best`。随后在原始全局截止时间内继续 ORDER_SPLIT。这个规则与搜索过程中“改善当前状态即可推进、但未必更新正式最好记录”的跨车局部动作不同。[前缀正式接纳](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/fleet_timed_pool.py:127)。

## 8. 已完成的等时结果与当前身份

已完成的等时开发筛查覆盖11例：丹麦5例、XTW3例、GH3例，使用2个种子和6种方法，共132个终点。所有终点通过独立复算，涉及3,837条终点路线及66个定价前缀提出的车队。各法均使用单物理核60秒在线预算。主要差额如下，正数表示 FLEET_PRICE 净效益较高。

| 来源 | 对 FLEET_PRICE_D ΔU | 对 PREFIX_DISTANCE ΔU | 对 ORDER_SPLIT ΔU |
|---|---:|---:|---:|
| 丹麦 | +15.82 | −6.84 | −15.51 |
| XTW | −31.42 | −41.24 | −1.35 |
| GH | +42.99 | +17.33 | +20.37 |
| 三来源等权 | +9.13 | −10.25 | +1.17 |

前缀自身能较明显改善共同初解，但开局优势并不等于60秒终态优势。两种前缀之后都继续 ORDER_SPLIT 时，神经前缀相对纯距离前缀在丹麦、XTW、GH 的首次接纳车队差额分别为 +42.52、+2.86、+44.69 U；到60秒终点时变为 −6.84、−41.24、+17.33 U。这同时反映了前缀计算占时及后续搜索轨迹变化。

目前，FLEET_PRICE 是已完成等时筛查的相关开发扩展；跨来源和逐种子的预设收益要求尚未同时满足。主方法仍采用锁版 ORDER_SPLIT。以上11例结果用于说明扩展的机制与取舍，不并入锁版主方法的统一16法、62例及三个前瞻来源成绩；原为该扩展另留的32例保持封存。[FLEET_PRICE 等时筛查完整结果](D:/PyCharm_/MASKCO-Main/research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/results/fleetscreen01/RESULTS.md:1)。
