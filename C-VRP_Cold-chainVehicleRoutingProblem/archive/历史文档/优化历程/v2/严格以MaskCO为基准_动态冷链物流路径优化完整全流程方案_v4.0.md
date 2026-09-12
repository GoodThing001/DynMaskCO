# DynMaskCO：严格以 MaskCO 为基准的动态冷链物流路径优化完整方案

> **版本：v4.0 — MaskCO-Centric Master Plan**  
> **日期：2026-08-31**  
> **定位：项目后续唯一方法与实验执行主文档**  
> **核心原则：任何新增模块都必须能回答“它如何从 MaskCO 的 masked reconstruction 范式自然扩展而来”，不能把 MaskCO 降级成无关插件，更不能另起一个与 MaskCO 无关的独立求解器作为最终主方法。**

---

# 0. 文档目的与最高原则

本项目的研究问题不是：

> “给 CVRP 加几个冷链字段”，也不是“在 MaskCO 外面套一个 rolling horizon”，更不是“最后用一个启发式 joint assignment 替代 MaskCO”。

本项目真正要解决的是：

\[
\boxed{
\text{如何把 MaskCO 的静态 masked-generation / reconstruction 范式，
扩展为一个在动态订单揭示、多车辆并行、时间窗/容量/回仓约束和冷链资源条件下，
严格 non-anticipatory、feasibility-preserving、可持续 recourse 的神经组合优化框架。}
}
\]

因此，最终方法必须始终保留 MaskCO 的核心语义：

\[
\boxed{
\text{当前可见 incumbent}
\rightarrow
\text{部分结构 mask}
\rightarrow
\text{MaskCO 表示与重建}
\rightarrow
\text{学习到的 preference}
\rightarrow
\text{资源可行搜索}
\rightarrow
\text{接受更好的可行 recourse}
}
\]

MaskCO 不是一个额外打分器，而是整个方法的**神经组合优化底座和表示学习范式**。

---

# 1. 父方法 MaskCO：必须保留的核心

## 1.1 MaskCO 的基本范式

MaskCO 的核心不是 from-scratch autoregressive construction，而是：

\[
\boxed{\text{Mask-and-Reconstruct}}
\]

给定一条已有高质量解：

\[
R
\]

先转换为解结构（如邻接矩阵）：

\[
A(R)
\]

再保留一部分结构、遮掉另一部分：

\[
\widetilde A
=
Mask(A(R),\tau)
\]

其中 \(\tau\) / keep rate 描述 MaskCO 的重建难度，而不是现实物流时间。

模型根据：

- 节点特征；
- 边/图结构；
- 保留下来的部分 adjacency；
- mask timestep / keep rate；

预测缺失结构：

\[
L^{route}_{ij}
=
Decoder(
Encoder(X,E),
\widetilde A,
\tau
)
\]

最终得到 edge logits / edge preference。

---

## 1.2 MaskCO 的训练逻辑

基本闭环：

```text
参考路线
  ↓
转换邻接结构
  ↓
部分遮边
  ↓
Encoder
  ↓
Decoder
  ↓
edge logits
  ↓
与真实被遮边计算 loss
  ↓
backprop
  ↓
更新模型
```

即：

\[
Mask
\rightarrow
Predict
\rightarrow
Loss
\rightarrow
Backprop
\rightarrow
Update
\]

这套结构必须保留到 DynMaskCO。

---

## 1.3 MaskCO 的推理逻辑

推理时没有 label。

MaskCO 从当前 incumbent 出发：

```text
Current incumbent
    ↓
mask 部分可修改结构
    ↓
MaskCO reconstruct
    ↓
得到 learned edge/structure preference
    ↓
生成 candidate
    ↓
评价 candidate
    ↓
保留更优结构
```

因此本项目所有动态化扩展，都必须保留：

\[
\boxed{
incumbent
\rightarrow
partial solution
\rightarrow
MaskCO reconstruction
}
\]

不能把 decoder adjacency 全置零后从空图构造，否则任务语义已经偏离父方法。

---

# 2. 从 MaskCO 到 DCC-VRP 的四个根本缺口

Static MaskCO 不能直接用于动态冷链路径优化，原因不是“模型不够大”，而是存在结构性缺口。

## 2.1 缺口一：未来信息泄漏

静态问题默认全部客户已知。

动态问题中，在时刻 \(t_e\)：

\[
r_i>t_e
\]

的客户不允许影响当前决策。

需要：

\[
\boxed{\text{Causal MaskCO}}
\]

---

## 2.2 缺口二：历史不能被重写

静态 MaskCO 可以重新构造整条解。

真实车辆执行后：

- 已服务客户；
- 已走过路径；
- 正在行驶的 committed leg；

都不能撤销。

因此需要：

\[
\boxed{\text{Frozen Prefix + Committed-Leg Invariance}}
\]

---

## 2.3 缺口三：edge preference 不是硬约束

MaskCO 输出：

\[
L_{ij}
\]

只能表示：

> 哪些连接更值得考虑。

它不能证明：

- capacity 可行；
- TW 可行；
- depot return 可行；
- exact-once；
- complete service。

因此需要：

\[
\boxed{\text{Feasibility-Preserving Search}}
\]

---

## 2.4 缺口四：动态多车的关键不是只有 route sequencing

最新诊断已证明：

\[
G_{fleet}\gg G_{seq}
\]

即当前问题主要瓶颈是：

> **visible mutable customers 在多辆车之间如何重新分配。**

因此 Static MaskCO 的：

\[
\text{node}\rightarrow\text{edge}
\]

决策粒度必须进一步升级为：

\[
\boxed{
\text{MaskCO representation}
\rightarrow
\text{vehicle-request preference}
}
\]

但这不是抛弃 MaskCO，而是增加一个与动态 fleet task 对齐的新 decoder head。

---

# 3. DCC-VRP 严格问题定义

## 3.1 客户

客户集合：

\[
V=\{1,\ldots,N\}
\]

depot：

\[
0
\]

客户 \(i\) 至少包含：

\[
x_i,\;
d_i,\;
[a_i,b_i],\;
s_i,\;
c_i,\;
r_i
\]

分别表示：

- 坐标；
- demand；
- time window；
- service duration；
- temperature class；
- reveal time。

当前基础节点表示可写为：

```text
[x, y, demand, tw_start, tw_end, temp_class, reveal_time]
```

---

## 3.2 车辆

当前 canonical 50-node benchmark 使用：

\[
K=25
\]

同质车辆上限。

车辆容量：

\[
Q
\]

注意：

\[
K
\]

必须在 episode 开始前固定，不能根据全体客户 demand 动态计算，否则会泄露未来总需求。

---

# 4. FleetState：动态方法的真实状态

每辆车维护：

```text
VehicleState
├── vehicle_id
├── status
│    ├── idle
│    ├── active
│    ├── committed
│    └── closed
├── current_node
├── ready_time
├── current_load
├── committed_next
├── committed_arrive
├── committed_finish
├── mutable_suffix
└── served_route
```

全局：

```text
FleetState
├── vehicles[0:K]
├── served_mask
├── reserved/committed_mask
├── pending_visible_mask
└── current_time
```

---

# 5. 四种车辆状态的正式语义

## idle

车辆还在 depot，尚未 dispatch。

当前 event 时刻可以被首次派出。

---

## active

车辆当前没有不可修改的 committed leg，可以重新规划 mutable suffix。

---

## committed

车辆正在执行已承诺动作。

该 leg：

\[
\boxed{\text{绝对不能取消}}
\]

后续只能从 committed 完成后的 anchor state 继续规划。

---

## closed

车辆已经返回 depot 并结束 route。

正式 single-route-per-vehicle 协议下：

\[
\boxed{\text{不能再次 dispatch}}
\]

避免问题偷偷变成 Multi-Trip VRPTW。

---

# 6. 在线信息结构：Filtration

事件 \(e\) 时：

\[
t_e
\]

当前可用信息：

\[
\mathcal F_e
\]

只包含：

- 已 reveal 客户；
- 这些客户已知属性；
- 当前 FleetState；
- 已执行动作；
- committed actions；
- 事先已知外生信息。

策略必须满足：

\[
\boxed{
a_e\sim\pi_\theta(\cdot|\mathcal F_e)
}
\]

如果两个未来世界当前：

\[
\mathcal F_e^A=\mathcal F_e^B
\]

那么当前：

- visible node representation；
- assignment logits；
- route logits；
- feasible candidate ordering；
- executed action；

都必须一致。

---

# 7. 事件驱动 Strict-Online Simulator

物理事件：

\[
\mathcal E
=
\{
Reveal,\;
ServiceCompletion,\;
ReturnCompletion
\}
\]

recourse triggers：

\[
\mathcal D
=
\{
Initial,\;
NewInformation,\;
PlanInvalidation,\;
PlanExhaustion
\}
\]

---

## 7.1 Reveal

当：

\[
r_i=t
\]

customer 进入：

```text
visible
pending
```

然后重新规划 mutable part。

---

## 7.2 Service Completion

车辆完成 committed service：

- 更新 current node；
- 更新 ready time；
- 更新 current load；
- customer → served；
- 若原 suffix 仍有效，可以继续，不强制每次都重新调用 solver。

---

## 7.3 Return Completion

车辆完成 depot return：

```text
status = closed
```

不允许再次使用。

---

# 8. DynMaskCO 总架构

最终主架构：

```text
┌─────────────────────────────────────────────┐
│ DCC-VRP Strict-Online Environment          │
│ visible orders + FleetState + incumbent    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ 1. Causal Preprocessing                    │
│ Feature Mask                               │
│ Visible-Only Normalization                 │
│ Visibility Attention Mask                  │
│ Edge/Adjacency Causalization               │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ 2. Causal MaskCO Encoder                   │
│ shared node/graph representation H         │
└──────────────┬───────────────────┬──────────┘
               │                   │
               ▼                   ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ Route Reconstruction │  │ Fleet Assignment Head    │
│ Head                 │  │ S(k,j | FleetState)      │
│ L_route(i,j)         │  │ MaskCO + dynamic state   │
└──────────┬───────────┘  └────────────┬─────────────┘
           │                           │
           │                           ▼
           │                Sound Feasibility Mask
           │                           │
           │                           ▼
           │                Joint Mutable Assignment
           │                           │
           └──────────────┬────────────┘
                          ▼
┌─────────────────────────────────────────────┐
│ 3. Masked Fleet Recourse                   │
│ incumbent → mask → reconstruct             │
│ frozen prefix / committed legs unchanged   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│ 4. Per-Vehicle Resource-Feasible Routing   │
│ Resource Beam / feasible sequencing        │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│ 5. Fleet Candidate                         │
│ complete mutable suffix plan               │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│ 6. Service-First Fleet Guard               │
│ incumbent vs candidate                     │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│ 7. Commit Actions Before Next Event        │
└─────────────────────┬───────────────────────┘
                      ▼
             StrictOnlineEnv advances
```

---

# 9. 第一层：Causal Feature Gating

future customer：

\[
r_i>t_e
\]

不能保留其真实属性。

必须替换为固定 unknown/dummy representation。

例如：

```text
coords      → fixed placeholder
demand      → fixed constant
tw_start    → fixed constant
tw_end      → fixed constant
temp_class  → future type
reveal      → 不暴露 realized future identity
```

目标：

> 改变 future customer 的真实坐标、需求、TW、温度，不允许影响 visible representation。

---

# 10. 第二层：Visible-Only Normalization

禁止：

> 使用全部 50 个节点计算坐标均值/尺度。

只能使用当前 visible nodes：

\[
\mu_e
=
\frac{1}{|V_e|}
\sum_{i\in V_e}
x_i
\]

\[
\sigma_e
=
Scale(\{x_i:i\in V_e\})
\]

visible node：

\[
\hat x_i
=
\frac{x_i-\mu_e}{\sigma_e}
\]

future dummy 不参与统计。

---

# 11. 第三层：Visibility-Gated Attention

对于 visible query \(i\) 和 future key \(j\)：

\[
B_{visibility}(i,j)=-\infty
\]

因此：

\[
Attention(i,j)=0
\]

attention：

\[
S_{ij}^h
=
\frac{Q_i^h(K_j^h)^T}{\sqrt{d_h}}
+
B_{visibility}(i,j)
+
B_{edge}(i,j,h)
\]

这样 future value 不会流入 visible representation。

---

# 12. 第四层：Decoder Context Causalization

MaskCO decoder 不能看到 future target adjacency。

正式路径：

```text
Full target routes
    ↓
Visible Route Projection
    ↓
Projected adjacency
    ↓
MaskCO solution mask
    ↓
× visible × visible
    ↓
Decoder context
```

---

# 13. Visible Route Projection

例如完整路线：

```text
0 → 1 → 7 → 2 → 0
0 → 4 → 9 → 3 → 0
```

当前：

```text
visible = {1,2,3,4}
future  = {7,9}
```

投影：

```text
0 → 1 → 2 → 0
0 → 4 → 3 → 0
```

规则：

1. future customer 删除；
2. 同一 vehicle route 内重连 visible predecessor/successor；
3. 绝对不能跨 depot separator 重连。

然后再：

\[
A_{projected}
\rightarrow
MaskCO\ solution\ mask
\rightarrow
A_{causal}
\]

---

# 14. 第五层：Visible-Only Objective

route reconstruction loss：

\[
L_{route}
\]

只能在 visible-visible target edge 上计算。

future target edge 不参与当前梯度。

必须通过：

- future target adjacency perturbation；
- visible logits invariance；
- visible loss invariance；

测试。

---

# 15. MaskCO solution mask 与 Causal mask 必须区分

## MaskCO solution mask

遮的是：

\[
\boxed{\text{解结构}}
\]

回答：

> 哪些已有路线边保留、哪些需要模型重建？

---

## Causal mask

遮的是：

\[
\boxed{\text{未来信息}}
\]

回答：

> 哪些客户现在根本不允许模型知道？

两者不是一个 mask。

---

# 16. MaskCO timestep 与物流时钟必须区分

三种时间概念：

### Reality time

\[
t_e
\]

车辆现实世界时钟。

### reveal time

\[
r_i
\]

客户订单何时可见。

### MaskCO timestep / keep rate

表示：

> 当前 reconstruction corruption 程度。

三者严禁混用。

---

# 17. MaskCO 基础表示层

当前 backbone 可继续保持：

```text
hidden_dim = 256
heads = 8
Transformer-style Encoder
```

节点输入：

```text
7D typed node feature
→ embedding
→ causal MaskCO Encoder
→ H = {h_i}
```

需要保留 type embedding：

- depot
- normal
- chilled
- frozen
- future/unknown

---

# 18. 双 Head：严格以 MaskCO 为核心

## 18.1 Route Reconstruction Head

保留父方法：

\[
L^{route}_{ij}
\]

它继续负责：

- masked route reconstruction；
- route structure learning；
- auxiliary combinatorial supervision；
- per-vehicle feasible sequencing中的 preference。

---

## 18.2 Fleet Assignment Head

新增：

\[
L^{fleet}_{kj}
=
S_\theta(k,j|\mathcal F_e,FleetState)
\]

它不是独立于 MaskCO 的网络。

必须以：

\[
h^{MaskCO}_{anchor_k}
\]

和：

\[
h^{MaskCO}_{j}
\]

为核心输入。

因此主方法是：

\[
\boxed{
\text{shared Causal MaskCO Encoder}
+
\text{task-specific Route Head}
+
\text{task-specific Fleet Head}
}
\]

---

# 19. 为什么不能直接使用旧 Edge Logits 做 Fleet Assignment

旧：

\[
L_{ij}
\]

训练任务：

> “从节点 \(i\) 出发，下一个 route node 更像是哪个 \(j\)？”

Fleet assignment 任务：

> “在当前整个 FleetState 下，customer \(j\) 应该由 vehicle \(k\) 负责吗？”

两者存在：

- semantic mismatch；
- vehicle dynamic-state 缺失；
- cross-source score calibration 问题；
- TW opportunity cost 缺失；
- assignment cascade。

因此 JF1-R 负结果不能解释成 MaskCO 没用，而应该解释成：

\[
\boxed{
\text{旧 decoder head 不适用于新任务；
MaskCO representation 仍需保留并重新对齐。}
}
\]

---

# 20. Vehicle Anchor State

ready vehicle：

\[
a_k=currentNode_k
\]

\[
t_k=readyTime_k
\]

committed vehicle：

\[
a_k=committedNext_k
\]

\[
t_k=committedFinish_k
\]

closed：

> 不进入候选车辆集合。

---

# 21. MaskCO Fleet Assignment Head

车辆表示：

\[
v_k
=
MLP_v(
[
h^{MaskCO}_{a_k},
t_k/T,
load_k/Q,
remainingCap_k/Q,
status_k
]
)
\]

客户表示：

\[
c_j
=
MLP_c(
[
h^{MaskCO}_j,
x_j
]
)
\]

pair：

\[
\psi_{kj}
\]

包含动态兼容信息。

---

# 22. JF2 Pair Features

第一版核心：

\[
\boxed{
TW\ slack
+
fleet\ scarcity
+
opportunity\ cost
}
\]

而不是以容量为主。

当前 canonical K=25 数据显示 fleet-level capacity 很宽松，因此 canonical regime 主要是 TW-driven。

但 capacity feature 仍保留，不能删除。

---

# 23. Motion Features

\[
travel_{kj}
\]

\[
ETA_{kj}
=
t_k+travel(a_k,j)
\]

\[
wait_{kj}
=
\max(0,a_j-ETA_{kj})
\]

---

# 24. Time-Window Features

\[
start_{kj}
=
\max(ETA_{kj},a_j)
\]

\[
TWSlack_{kj}
=
b_j-start_{kj}
\]

\[
TWWidth_j=b_j-a_j
\]

\[
RelativeUrgency_{kj}
=
\frac{TWSlack_{kj}}{TWWidth_j+\epsilon}
\]

\[
DeadlineGap_{kj}=b_j-t_k
\]

---

# 25. Return / Capacity Features

\[
CapSlack_{kj}
=
Q-(load_k+d_j)
\]

以及：

\[
ReturnSlack_{kj}
\]

ReturnSlack 第一版优先作为 scorer feature。

只有 Pair Feasibility Audit 证明某个 return check 是 sound necessary condition 时，才能 hard-mask。

---

# 26. Fleet Scarcity

customer \(j\) 当前可行车辆数：

\[
m_j
=
|\mathcal V_j^{feasible}|
\]

normalized：

\[
Scarcity_j
=
m_j/|\mathcal V_e|
\]

\(m_j\) 越小：

> customer 越稀缺。

---

# 27. Vehicle Coverage

vehicle \(k\) 当前可服务客户数：

\[
n_k
=
|\mathcal C_k^{feasible}|
\]

表示：

> 该车辆当前资源覆盖能力。

---

# 28. Opportunity / Regret Features

对于 customer \(j\)：

最小 travel：

\[
d_1(j)
\]

第二小：

\[
d_2(j)
\]

定义：

\[
TravelRegret_j=d_2-d_1
\]

类似定义：

\[
SlackRegret_j
\]

帮助模型学习：

> 如果不用最佳/稀缺 vehicle，机会成本是多少。

---

# 29. Fleet Assignment Residual Scorer

JF1-H 已证明 min-travel joint assignment 是强 baseline。

因此不从零学习。

定义：

\[
S_H(k,j)
=
-NormTravel(a_k,j)
\]

learned MaskCO residual：

\[
r_\theta(k,j)
=
MLP(
[
v_k,
c_j,
v_k\odot c_j,
\psi_{kj},
g_{fleet}
]
)
\]

最终：

\[
\boxed{
S_{fleet}(k,j)
=
S_H(k,j)
+
\alpha r_\theta(k,j)
}
\]

初始化：

\[
\boxed{\alpha=0}
\]

这样：

\[
JF2_{\alpha=0}
=
JF1-H
\]

---

# 30. JF2 Gate-0

在任何训练之前：

\[
\alpha=0
\]

必须精确复现：

```text
JF1-H distance = 24.5019
completion = 100%
```

若不复现：

> 说明 candidate/search semantics 被悄悄改变，禁止进入正式 JF2。

---

# 31. Sound Feasibility Mask

硬约束不能交给网络学习。

先：

\[
\boxed{\text{Feasibility Filter}}
\]

再：

\[
\boxed{\text{MaskCO learned ranking}}
\]

---

# 32. Hard Mask 第一版

必须包括：

### Visibility

\[
j\in Visible
\]

### Service state

- 未 served；
- 未 committed；
- mutable。

### Vehicle

- 非 closed；
- 有合法 anchor。

### Capacity necessary condition

\[
load_k+d_j\le Q
\]

### Direct arrival TW necessary condition

\[
start_{kj}\le b_j
\]

---

# 33. Pair Feasibility Audit

任何更强 hard filter，尤其：

```text
return-depot
future suffix estimate
```

必须先验证 false-negative rate。

对：

\[
pair\_feasible(k,j)=False
\]

的 pair：

> 临时分配给 vehicle k，再调用正式 route planner，验证是否确实无合法 suffix。

输出：

| reason | rejected | true infeasible | false negative |
|---|---:|---:|---:|
| capacity | | | |
| TW | | | |
| return | | | |

只有 false-negative=0 或严格证明 sound 的规则，才能成为正式 hard mask。

---

# 34. 禁止 Silent Drop

如果：

```text
feasible_vehicle_set(j) = empty
```

不能直接丢客户。

必须：

```text
unresolved
   ↓
repair
   ↓
fallback
```

若 candidate 无法完整服务：

> guard 回退 incumbent。

---

# 35. Mutable Ownership

event \(e\) 中：

### 冻结

- executed customers；
- committed customers；
- committed legs。

### 释放

所有：

\[
visible
\cap
unserved
\cap
mutable
\]

customer ownership。

这就是 Joint Fleet Recourse 的核心自由度。

---

# 36. JF2 第一版 Joint Assignment

为了只测 learned scorer，第一版保持：

\[
B=1
\]

即 greedy joint assignment。

当前 JF1.5 已证明基于 separable anchor-travel surrogate 的 assignment beam width 没有有效作用。

因此 JF2 不混入 beam-depth 变量。

---

# 37. Customer Assignment 顺序

推荐：

\[
\boxed{\text{MRV + TW urgency}}
\]

先选择：

> feasible vehicle 数最少的 customer。

tie：

> earliest `tw_end`。

作用：

> 先处理最难分配的客户。

---

# 38. Per-Vehicle Routing：继续使用 MaskCO Route Preference

完成 fleet assignment：

\[
A_k
\]

之后，每辆车并不是纯 greedy 随便排序。

仍然使用：

\[
\boxed{
MaskCO\ Route\ Reconstruction\ Head
+
Resource\ Feasible\ Search
}
\]

即：

```text
assigned customer set A_k
       ↓
current vehicle incumbent
       ↓
MaskCO mask-and-reconstruct
       ↓
route edge logits
       ↓
Resource Beam / feasible route search
       ↓
mutable suffix R_k
```

因此 MaskCO 在最终方法中承担两个学习层：

1. fleet assignment representation / preference；
2. per-vehicle route reconstruction preference。

---

# 39. Resource Beam State

单车资源状态：

\[
z=(i,t,q,V)
\]

其中：

- current/anchor node \(i\)
- current time \(t\)
- load \(q\)
- remaining candidate set \(V\)

每一步扩展 customer \(j\)：

先检查：

- visible；
- unserved；
- capacity；
- TW；
- return feasibility；
- commitment。

再用 MaskCO logits排序合法动作。

---

# 40. Resource Beam 的职责边界

MaskCO：

> 在合法动作中提供 preference。

Resource Beam：

> 决定哪些动作根本允许进入搜索。

严禁：

> 模型高分覆盖 hard constraint。

---

# 41. Fleet Candidate

所有车辆生成 mutable suffix：

\[
R_1,\ldots,R_K
\]

组合成：

\[
P^{cand}
\]

必须满足：

- exact-once；
- no duplicate；
- frozen prefix unchanged；
- committed unchanged；
- capacity；
- TW；
- depot return；
- visibility。

---

# 42. Service-First Fleet Guard

incumbent：

\[
P^{inc}
\]

candidate：

\[
P^{cand}
\]

比较优先级：

### 第一层

completion / unserved。

### 第二层

TW / capacity / return / duplicate。

### 第三层

服务等价后才比较：

\[
distance
\]

后续冷链扩展再比较 quality/energy。

---

# 43. 当前 Guard 的定位

最新 long-horizon audit 已表明：

- service safety 当前没有暴露 accepted service-worse；
- 但 local cost guard 存在 myopia；
- 并存在 rejected beneficial opportunities。

因此当前 guard 定义为：

\[
\boxed{\text{Safety Guard}}
\]

不是最终 value predictor。

后续可升级：

\[
\boxed{\text{Value-Aware Fleet Guard}}
\]

但不阻塞 JF2。

---

# 44. 在线推理完整循环

```text
while episode not done:

    1. simulator advance to next physical event

    2. update FleetState

    3. reveal newly available orders

    4. freeze:
         executed prefix
         committed legs

    5. build visible mutable customer pool

    6. construct current feasible incumbent

    7. project incumbent into MaskCO partial structures

    8. causal feature gating

    9. visible-only normalization

   10. Causal MaskCO Encoder
         → H

   11. Fleet Assignment Head
         → S_fleet(k,j)

   12. sound hard feasibility mask

   13. joint mutable assignment

   14. per vehicle:
         incumbent suffix
         → MaskCO route mask
         → route reconstruction logits
         → Resource Beam
         → feasible suffix

   15. assemble fleet candidate

   16. authoritative service-first evaluation

   17. fleet guard:
         candidate vs incumbent

   18. install selected mutable suffixes

   19. commit only actions that must execute
       before next event

   20. continue
```

---

# 45. 训练体系总览

最终训练不能只训练一个 fleet MLP。

完整 DynMaskCO 训练应逐步包含：

\[
\boxed{
L
=
L_{MaskCO-route}
+
\lambda_aL_{assign}
+
\lambda_pL_{fleet-pref}
+
\lambda_cL_{cold-chain}
}
\]

不是第一天全部打开，而是阶段化加入。

---

# 46. Stage T0：MaskCO Causal Route Reconstruction

第一层训练仍严格基于 MaskCO。

每 optimizer step 模拟：

\[
K_{seq}=5
\]

个 visibility stages。

注意：

> K=5 不是现实中五次运输事件。

---

# 47. T0 训练流程

对每个 visibility stage \(k\)：

\[
vis_k
=
1[r_i\le threshold_k]
\]

然后：

\[
x_k
=
causalMaskFeatures(x,vis_k)
\]

\[
y_k
=
VisibleRouteProjection(y,vis_k)
\]

\[
A_k
=
MaskCOAdjacency(y_k,\tau_k)
\]

\[
A_k
=
Causalize(A_k,vis_k)
\]

\[
H_k
=
Encoder(x_k,vis_k)
\]

\[
L^{route}_k
=
DecodeRoute(H_k,A_k,\tau_k)
\]

loss：

\[
\mathcal L_{route}
=
\sum_k w_k
CE(
L^{route}_k,
VisibleEdges(y_k)
)
\]

---

# 48. T0 基础训练配置

当前已冻结基础配置可继续使用：

```text
optimizer = AdamW
batch_size = 64
peak_lr = 1e-3
full steps = 50,000
online_seq stages = 5
```

正式 checkpoint：

```text
seed42
seed123
seed999
seed2025
seed2026
```

但新模块 idea screening 仍从 seed42 开始。

---

# 49. T1：收集真实 JF1-H Student States

JF2 teacher 数据必须来自真实在线 state distribution。

当前部署基线：

\[
JF1-H
\]

因此：

```text
train instances
     ↓
JF1-H strict-online rollout
     ↓
recourse states
     ↓
save FleetState + mutable pool
```

第一目标：

\[
5000\sim10000
\]

有效 assignment states。

---

# 50. T1 数据分层

必须记录：

- initial / reveal / exhaustion / invalidation；
- early/mid/late episode；
- n_pending；
- n_active vehicles；
- feasible vehicle count；
- TW slack；
- scarcity；
- high branching。

因为 R1.7 已证明：

> early / high pending / high branching 是高价值也高风险状态。

---

# 51. T2：OR-Joint Assignment Teacher

OR teacher 只能使用当前：

\[
\mathcal F_e
\]

不能看真实未来。

冻结：

- executed；
- committed；
- current FleetState。

允许：

- visible mutable customer ownership 在 active vehicles 之间重分配。

输出：

\[
A^*
\]

作为 expert fleet assignment。

---

# 52. Teacher Budget Calibration

抽：

\[
200
\]

representative train states。

测试：

```text
50ms
100ms
250ms
500ms
1000ms
```

比较：

- completion；
- distance；
- assignment stability；
- runtime；
- solver gap（如可用）。

选择 offline expert quality-time knee。

---

# 53. T3：JF2-MaskCO Assignment Imitation

对 customer \(j\)：

sound-feasible vehicles：

\[
\mathcal V_j^F
\]

OR teacher：

\[
k_j^*
\]

assignment loss：

\[
\mathcal L_{assign}
=
-\log
\frac{
\exp S(k_j^*,j)
}{
\sum_{k\in\mathcal V_j^F}\exp S(k,j)
}
\]

softmax 只在 feasible vehicles。

---

# 54. T3 第一阶段：冻结 MaskCO Encoder

先：

```text
freeze Causal MaskCO Encoder
train FleetAssignmentHead only
```

目的：

> 检测已经训练好的 MaskCO representation 是否可以迁移到 fleet assignment。

这是 MaskCO transfer 的关键实验证据。

---

# 55. FeatureOnly 只作为消融

同步建立：

```text
JF2-Feature
```

它只使用：

- travel；
- ETA；
- TW slack；
- return slack；
- scarcity；
- regret；
- FleetState handcrafted features。

不用 MaskCO embeddings。

目的不是成为主方法，而是回答：

\[
\boxed{
\text{MaskCO representation 是否提供超越人工动态特征的增量？}
}
\]

---

# 56. JF2 主实验矩阵

| 方法 | Joint topology | MaskCO repr | Fleet state | Learned |
|---|---:|---:|---:|---:|
| NN | No | No | partial | No |
| H2-G | sequential | Yes | indirect | Yes |
| JF1-H | Yes | No | heuristic | No |
| JF-Old | Yes | old edge logits | weak | Yes |
| JF2-Feature | Yes | No | Yes | Yes |
| **JF2-MaskCO** | **Yes** | **Yes** | **Yes** | **Yes** |
| JF2-Shuffle | Yes | shuffled | Yes | control |
| OR-joint | Yes | No | full optimization | expert |

---

# 57. JF2 Attribution Gates

## Gate A：learned fleet utility

\[
JF2_{MaskCO}
<
JF1-H
\]

前提：

\[
service=100\%
\]

---

## Gate B：learned score 不是随机

\[
JF2_{Real}
<
JF2_{Shuffle}
\]

---

## Gate C：MaskCO representation 有增量

\[
JF2_{MaskCO}
<
JF2_{Feature}
\]

---

## Gate D：旧 head → 新 head 的 task alignment

\[
JF2_{MaskCO}
<
JF\text{-Old}
\]

证明：

> 不是简单“换搜索空间”，而是新的 MaskCO fleet decoder 语义更正确。

---

# 58. JF2 成功阈值

当前强 heuristic：

\[
JF1-H=24.50
\]

### Minimum

\[
<24.25
\]

且 100% service。

### Strong

\[
\le24.0
\]

### Very strong

\[
\le23.75
\]

### Expert-near

\[
23.2\sim23.5
\]

相对于 OR-joint 约 22.79 已非常有价值。

---

# 59. T4：联合 Fine-Tune MaskCO Encoder

如果 frozen encoder + new head 有效：

再解冻 encoder。

建议：

```text
head_lr    ≈ 3e-4 ~ 1e-3
encoder_lr ≈ 1e-5 ~ 1e-4
```

同时保留 route reconstruction auxiliary loss：

\[
\mathcal L
=
\mathcal L_{assign}
+
\lambda_r\mathcal L_{route}
\]

这样不会把 MaskCO backbone 微调成纯 assignment MLP。

---

# 60. T5：Fleet Preference / Regret Learning

OR assignment存在多解。

One-hot CE 会把近等价 vehicle 全部当负类。

因此下一步对 hard alternative 做 counterfactual expert query。

---

# 61. Hard Negative Query

对于 customer \(j\)：

OR：

\[
k^*
\]

模型 alternative：

\[
\tilde k
\]

强制：

\[
j\rightarrow\tilde k
\]

OR 再优化剩余 current visible assignment。

得到：

\[
J(\tilde k,j)
\]

定义：

\[
Regret(\tilde k,j)
=
J(\tilde k,j)-J(k^*,j)
\]

---

# 62. Fleet Preference Loss

选择：

\[
k^+
\]

和：

\[
k^-
\]

训练：

\[
\mathcal L_{pref}
=
-w
\log
\sigma(
(S_{k^+j}-S_{k^-j})/\tau
)
\]

权重：

\[
w
=
clip(
Regret/Scale,
0,
w_{max}
)
\]

完整：

\[
\mathcal L
=
\mathcal L_{assign}
+
\lambda_r\mathcal L_{route}
+
\lambda_p\mathcal L_{pref}
\]

---

# 63. T6：DAgger

当 JF2 自己 rollout 后，state distribution 会改变。

流程：

```text
current JF2 student
    ↓
train rollout
    ↓
student states
    ↓
OR-joint residual expert
    ↓
aggregate dataset
    ↓
fine-tune
```

直到 VAL gain 饱和。

---

# 64. JF2-M1：Best Insertion Features

M0 有效后再加。

对每个：

\[
(k,j)
\]

在 provisional route \(R_k\) 上尝试插入：

\[
p=1,\ldots,|R_k|+1
\]

计算：

\[
BestInsertionCost_{kj}
\]

\[
BestInsertionTWSlack_{kj}
\]

\[
BestInsertionReturnSlack_{kj}
\]

作为新 features。

这使 assignment scorer 从：

> anchor compatibility

升级为：

> route insertion compatibility。

---

# 65. JF3：Full Joint Fleet Masked Resource Beam

当前 JF1.5 beam 无效，不代表真正 joint fleet beam 无效。

原因是旧 surrogate：

\[
\sum_j travel(anchor_{k_j},j)
\]

近似可分。

JF3 必须让 partial assignment 真正改变 route/resource state。

---

# 66. JF3 State

\[
Z
=
\{
R_k,
loc_k,
t_k,
q_k,
status_k
\}_{k=1}^K
+
U
\]

其中 \(U\) 为尚未处理的 visible mutable customers。

---

# 67. JF3 Action

\[
a=(k,j)
\]

或者：

\[
WAIT(k)
\]

\[
CLOSE(k)
\]

扩展后立即：

- 更新 provisional route；
- 更新时间；
- 更新 load；
- 更新 TW slack；
- 更新 future insertion possibilities。

这时：

\[
(k,j_1)
\]

会改变：

\[
score/feasibility(k,j_2)
\]

于是搜索真正 non-separable。

---

# 68. JF3 的 MaskCO 作用

JF3 仍以 MaskCO 为核心：

### Fleet head

\[
S^{fleet}(k,j)
\]

决定 vehicle-customer expansion preference。

### Route head

\[
S^{route}(i,j)
\]

帮助 vehicle 内部 route insertion/reconstruction。

因此最终 beam score可以融合：

\[
Score
=
\beta_fS^{fleet}
+
\beta_rS^{route}
+
heuristic/resource\ terms
\]

但 hard feasibility永远单独执行。

---

# 69. JF3 Beam Sweep

到 JF3 才重新扫：

```text
B = 1 / 4 / 8 / 16
```

同时记录：

- quality；
- service；
- runtime；
- nodes expanded；
- pruning rate。

找到 quality-time knee。

---

# 70. Neural LNS

如果 JF3 仍存在明显 gap，可以在：

\[
\boxed{\text{mutable suffix}}
\]

上做 Neural Large Neighborhood Search。

Destroy：

- high regret customers；
- TW-critical customers；
- cross-route exchange；
- high uncertainty；
- thermal-risk customers。

禁止 destroy：

- executed；
- committed。

Repair：

\[
MaskCO FleetHead
+
MaskCO RouteHead
+
ResourceFeasibleSearch
\]

---

# 71. Stochastic Lookahead

严格在线不是不能利用未来分布。

禁止：

> 读取真实 future realization。

允许：

\[
P(Future|\mathcal F_e)
\]

生成 scenario。

先做 Value-of-Information diagnostic：

\[
\omega_1,\ldots,\omega_M
\]

对 recourse candidate \(x\)：

\[
\hat J(x)
=
\frac1M
\sum_mJ(x,\omega_m)
\]

或：

\[
CVaR_\alpha
\]

只有明显有价值后，再训练 future scenario model。

---

# 72. 冷链优化层

冷链部分必须建立在：

> causal + complete + feasible fleet routing

之后。

不能为了“冷链创新”破坏前面的正确性。

目标维度可包括：

- distance；
- quality deterioration；
- energy consumption；
- unsalable risk。

具体品质/能耗物理公式必须以当前项目最终代码与实验协议为准；在尚未冻结统一公式前，本方案不虚构新的物理模型。

---

# 73. Preference-Conditioned Cold-Chain Policy

未来可输入：

\[
w=(w_D,w_Q,w_E)
\]

学习：

\[
\pi(a|\mathcal F_e,w)
\]

其中：

- \(D\)：distance；
- \(Q\)：quality；
- \(E\)：energy。

训练时：

\[
w\sim simplex
\]

输出可控 Pareto policy。

评价：

- Hypervolume；
- IGD；
- spacing；
- controllability；
- specialist gap。

---

# 74. Capacity / Fleet Tightness 扩展

当前 canonical：

\[
K=25
\]

总 capacity 显著宽松。

因此当前主要是 TW-dominated。

后续必须加入：

- loose fleet；
- medium fleet；
- tight fleet；

验证 MaskCO fleet assignment 在 capacity-active regime 中仍然有效。

---

# 75. 最小可行 Fleet Size

不要拍脑袋设置：

```text
K=6
K=8
```

先用 OR-joint 对每个 instance估：

\[
K_i^{min}
\]

再根据分布定义：

```text
Loose
Medium
Tight
```

确保各 regime 大部分实例仍有 complete solution。

---

# 76. Scale Extension

最终规模：

\[
50
\rightarrow
100
\rightarrow
200
\rightarrow
500
\]

重点观察：

- quality；
- runtime；
- memory；
- beam expansion；
- MaskCO representation transfer；
- fleet assignment scalability。

---

# 77. Road-Network Extension

从：

```text
Euclidean distance
```

逐步扩展：

```text
asymmetric travel duration
```

再：

```text
road-network / OSRM
```

此时 edge features 可包括：

- distance；
- duration；
- direction；
- energy；
- TW compatibility。

---

# 78. Authoritative Evaluator

所有方法都必须通过同一个 evaluator。

不能相信 solver 自己输出：

```text
feasible=True
```

统一检查：

### Service

- completion；
- unserved；
- exact-once。

### Constraints

- TW；
- capacity；
- depot return；
- duplicate。

### Cost

只有 service-equivalent 后比较：

\[
pure\ travel\ distance
\]

### Resource

- vehicles used；
- runtime。

### Cold Chain

后续统一：

- quality；
- energy；
- unsalable。

---

# 79. Service-First Evaluation Contract

严格优先级：

\[
\boxed{
Completion
>
TW/Capacity/Return
>
Distance
>
ColdChainPreference
}
\]

任何：

> cost 更低但漏单

都不能说更优。

---

# 80. 必须保留的 Baselines

最终至少：

- NN / EDD heuristic；
- JF1-H heuristic-only；
- parent/static MaskCO adaptation；
- H2-G sequential DynMaskCO；
- JF-Old edge-logit assignment；
- JF2-Feature；
- JF2-MaskCO；
- JF2-Shuffle；
- OR-Tools-RH；
- OR-joint；
- dynamic HGS/ALNS；
- RRNCO rolling-horizon adaptation；
- CaDA adaptation。

---

# 81. 最关键的 MaskCO Attribution Matrix

必须通过下面一组对照证明：

> 最终提升不是“joint heuristic”冒充 MaskCO 贡献。

| 组件 | JF-H | Feature | MaskCO | Shuffle | Full |
|---|---:|---:|---:|---:|---:|
| Joint topology | ✓ | ✓ | ✓ | ✓ | ✓ |
| Min-travel prior | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fleet features | × | ✓ | ✓ | ✓ | ✓ |
| MaskCO embedding | × | × | ✓ | shuffle | ✓ |
| Fleet preference loss | × | optional | optional | shuffle | ✓ |
| Route MaskCO head | optional | ✓ | ✓ | ✓ | ✓ |
| Resource feasibility | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fleet guard | ✓ | ✓ | ✓ | ✓ | ✓ |

---

# 82. 最终要证明的四个贡献关系

## C1：MaskCO learned information

\[
Real<Shuffle
\]

---

## C2：MaskCO representation transfer

\[
MaskCO<FeatureOnly
\]

---

## C3：Fleet decision-space improvement

\[
JF-H<H2-G
\]

---

## C4：Learning adds value beyond topology

\[
JF2-MaskCO<JF-H
\]

四条同时成立，论文主线才最强。

---

# 83. Blocking Causality Tests

正式方法必须全部 PASS：

### Future Feature Perturbation

改变 future：

- coords；
- demand；
- TW；
- temp。

要求 visible logits：

\[
\Delta<10^{-6}
\]

---

### Future Target Adjacency Perturbation

改变 future target route/adjacency。

production decoder context、visible logits、visible loss：

\[
\Delta<10^{-6}
\]

---

### Prefix Invariance

任何 candidate：

\[
executed\ prefix
\]

修改次数：

\[
0
\]

---

### Commitment Invariance

任何：

\[
committed\ leg
\]

修改次数：

\[
0
\]

---

### Exact-Once

无 duplicate / omission。

---

### Fleet Concurrency

toy problem 中并发车辆时间区间真正重叠。

---

# 84. JF2 Blocking Tests

- `alpha=0` 精确复现 JF1-H；
- future customers 不进入 fleet head candidate；
- closed vehicle 不得接客户；
- committed vehicle使用 committed completion anchor；
- mutable ownership 可以跨车；
- committed ownership 不可跨车；
- same candidate set for Real/Shuffle/Feature；
- hard mask 与 learned score解耦；
- no silent customer drop；
- unresolved进入 repair；
- guard reject 时 incumbent完全恢复；
- deterministic repeat；
- checkpoint load失败 hard error。

---

# 85. JF3 Blocking Tests

- partial beam state deep-copy；
- no cross-branch contamination；
- expansion 后 resource state正确；
- B=1 deterministic；
- beam增大不降低当前 best feasible incumbent；
- committed prefix永远不被 destroy；
- customer global exact-once；
- search timeout 返回 best feasible incumbent。

---

# 86. 训练 Seed 规范

### Idea screening

```text
42
```

### Accepted module confirmation

```text
42
123
999
```

### Final freeze

```text
42
123
999
2025
2026
```

正式统计单位：

\[
\boxed{\text{training seed}}
\]

不是 128 个 validation instances。

---

# 87. Train / Val / Test

### Train

用于：

- MaskCO training；
- expert labels；
- DAgger；
- preference；
- scenario model。

### Val

用于：

- architecture；
- hyperparameters；
- gates；
- beam width；
- teacher budget。

### Test

只能：

\[
\boxed{\text{最终冻结后 one-shot}}
\]

禁止 adaptive tuning。

---

# 88. 统计规范

最终报告建议：

- mean ± std over training seeds；
- paired Wilcoxon；
- McNemar for service outcomes；
- paired/bootstrap CI；
- Holm correction；
- Cliff's delta。

Anytime：

```text
50ms
100ms
250ms
500ms
1000ms
```

后续 LNS 可以扩展到更长 budget。

---

# 89. 当前证据驱动的冻结结论

## R1.7

MaskCO learned preference：

> 有独立价值。

## Guard

> service-safe，但 cost-myopic。

## B0

> fleet allocation 是主瓶颈，sequencing不是主要 residual gap。

## JF1

> joint mutable ownership/search topology 有巨大价值。

## JF1-R

> 旧 MaskCO edge head 直接拿来做 assignment 语义错误。

## JF1.5

> separable assignment surrogate 下 beam width无效。

因此下一步绝不是：

> 放弃 MaskCO。

而是：

\[
\boxed{
\text{把 MaskCO 的 representation 和 masked reconstruction
扩展到 FleetState-conditioned assignment task}
}
\]

---

# 90. 当前执行路线：必须严格按顺序

## Phase 0 — Closure

1. `legacy_jf1h` 模式。
2. B=1精确复现 JF1-H。
3. pair feasibility false-negative audit。
4. 删除 silent drop。
5. 冻结 sound candidate semantics。

---

## Phase 1 — MaskCO Fleet Head

6. 在现有 MaskCO model 中新增 FleetAssignmentHead。
7. RouteReconstructionHead继续保留。
8. residual：
   \[
   S=S_H+\alpha r_\theta
   \]
9. `alpha=0` regression PASS。

---

## Phase 2 — Expert Dataset

10. JF1-H rollout train。
11. 收集 5k–10k recourse states。
12. OR-joint 200-state budget calibration。
13. 生成 OR-joint assignment labels。

---

## Phase 3 — JF2-M0

14. 冻结 MaskCO encoder。
15. 训练 FleetAssignmentHead。
16. seed42。
17. 4-instance smoke。
18. 16-instance sanity。
19. VAL128。

---

## Phase 4 — Attribution

20. JF-H。
21. JF-Old。
22. JF2-Feature。
23. JF2-MaskCO。
24. JF2-Shuffle。
25. OR-joint。

---

## Phase 5 — Encoder Fine-Tune

26. 只有 JF2-MaskCO 有效才解冻 encoder。
27. 低 LR。
28. 保留 MaskCO route auxiliary loss。

---

## Phase 6 — Preference

29. Hard-negative OR queries。
30. Regret labels。
31. Fleet preference loss。
32. DAgger。

---

## Phase 7 — Route-Aware Assignment

33. Best insertion features。
34. route/fleet dual-head融合。

---

## Phase 8 — JF3

35. Full Joint Fleet Resource Beam。
36. 真正 non-separable fleet state。
37. B=1/4/8/16。
38. quality-time curve。

---

## Phase 9 — Neural LNS / Value Guard

39. mutable-suffix neural LNS。
40. value-aware fleet guard（如果仍有足够机会收益）。

---

## Phase 10 — Stochastic

41. VOI diagnostic。
42. scenario lookahead。
43. learned scenario model only if useful。

---

## Phase 11 — Cold Chain

44. quality/energy统一 evaluator。
45. preference-conditioned policy。
46. Pareto experiments。

---

## Phase 12 — Generalization

47. fleet tightness。
48. capacity-active。
49. 100/200/500。
50. asymmetric/OSRM。
51. external dynamic protocol。

---

## Phase 13 — Final

52. 3-seed confirmation。
53. final architecture freeze。
54. 5-seed full。
55. statistics。
56. one-shot TEST。

---

# 91. 当前立即执行的 8 个动作

从今天开始，只做：

```text
[1] JF1.5 legacy_jf1h B=1 regression
[2] pair_feasible false-negative audit
[3] no-silent-drop repair
[4] freeze JF2 candidate semantics
[5] DynamicColdChainModel 新增 FleetAssignmentHead
[6] 保留 RouteReconstructionHead
[7] 实现 S = S_H + alpha * r_MaskCO
[8] alpha=0 → JF1-H regression
```

这 8 个 PASS 以后，才进入 OR-joint label generation 和正式 JF2 training。

---

# 92. 方法代码建议结构

建议逐步重构为：

```text
model/
├── maskco_encoder.py
├── route_reconstruction_head.py
├── fleet_assignment_head.py
└── dynamic_maskco.py

simulation/
├── strict_online_env.py
├── fleet_state.py
├── joint_fleet.py
├── resource_beam.py
└── service_first_guard.py

causal/
├── feature_gating.py
├── visible_normalization.py
├── route_projection.py
└── causal_adjacency.py

expert/
├── or_joint_solver.py
├── collect_assignment_states.py
├── build_assignment_dataset.py
└── hard_negative_query.py

training/
├── train_maskco_route.py
├── train_fleet_head.py
├── train_joint_multitask.py
└── dagger_assignment.py

evaluation/
├── authoritative_evaluator.py
├── attribution_eval.py
├── anytime_eval.py
└── statistical_tests.py
```

如果现有工程目录结构不同，不需要为了文件名机械迁移；重点是职责必须分层。

---

# 93. 最终论文方法主线

不要写成：

> “我们加了 Causal Mask、Resource Beam、Joint Assignment、Guard、OR teacher、Preference、LNS……”

应该写成三层核心问题。

---

## Contribution I：Causal Dynamic Masked Reconstruction

Static MaskCO：

\[
\rightarrow
\]

Causal MaskCO。

解决：

> future information boundary。

机制：

- feature gating；
- visible-only normalization；
- visibility attention；
- causal decoder context；
- visible route projection；
- visible-only loss。

---

## Contribution II：Fleet-State Conditioned Masked Recourse

MaskCO 的表示从：

\[
node-edge
\]

扩展到：

\[
vehicle-request
\]

新增：

\[
S(k,j|\mathcal F_e,FleetState)
\]

解决：

> dynamic multi-vehicle assignment。

核心：

\[
\boxed{
MaskCO Encoder
+
Fleet Assignment Head
}
\]

---

## Contribution III：Feasibility-Preserving Joint Fleet Search

解决：

> learned preference不能保证硬约束。

机制：

- frozen prefix；
- committed invariant；
- mutable ownership release；
- joint assignment；
- Resource Beam；
- service-first guard；
- anytime feasible incumbent。

---

# 94. 后续可选强贡献

如果实验显著：

### Contribution IV

Expert/Preference-Guided Fleet Recourse。

### Contribution V

Stochastic Lookahead。

### Contribution VI

Preference-Conditioned Cold-Chain Multiobjective。

但它们都建立在前三层之后。

---

# 95. 最终方法的一句话

\[
\boxed{
\textbf{
DynMaskCO 是一个以 MaskCO masked reconstruction 为神经组合优化核心，
通过严格因果化保留当前可见结构，
用 FleetState-conditioned MaskCO head 学习跨车辆客户分配，
并借助 Frozen Prefix、Joint Fleet Search 与 Resource Feasibility
实现动态冷链车辆路径的在线可行 recourse 框架。
}
}
\]

---

# 96. 最重要的“不能做”

后续禁止以下方向：

1. **不能为了 JF1-H 强就把 MaskCO 删掉。**
2. **不能把 FeatureOnly 变成主方法。**
3. **不能把旧 edge logits 直接当 fleet assignment 最终设计。**
4. **不能让 network score 覆盖 hard constraints。**
5. **不能把 decoder adjacency 清零做 from-scratch constructor。**
6. **不能让 future customer进入 normalization / attention / adjacency / loss。**
7. **不能修改 executed / committed route。**
8. **不能 silent drop customer。**
9. **不能只报 distance 不报 completion。**
10. **不能把 JF1-H 的 11.4% 说成 learned MaskCO gain。**
11. **不能把 5 个 shuffle seeds 说成 5 个 training seeds。**
12. **不能在 TEST 上调模块。**

---

# 97. 项目停止标准

不是：

> “一定把 OR gap 做到 <10%”。

而是：

\[
\boxed{\text{quality / service / runtime / scale / robustness 的 Pareto improvement}}
\]

当新增模块不能继续产生：

- 可复现；
- 可解释；
- 统计可靠；
- 不牺牲 service；
- 不破坏 causal correctness；

的 Pareto gain 时停止。

---

# 98. 本文档优先级

从本版本开始，后续方法开发优先遵守本文件。

旧文档中的：

- Phase A/B 顺序；
- 单车排序优先；
- assignment beam sweep；
- 旧 offline baseline；
- 已作废 cost；

如果与本文件和最新 strict-online证据冲突：

> **以本文件、最新 authoritative evaluator 和最新冻结代码为准。**

---

# 99. 当前最终路线图

```text
Parent MaskCO
    │
    ▼
Causal MaskCO
    │
    ├── visible-only feature/attention/adjacency/loss
    │
    ▼
Strict-Online FleetState
    │
    ▼
MaskCO Shared Encoder
    │
    ├── Route Reconstruction Head
    │
    └── Fleet Assignment Head
    │
    ▼
Masked Fleet Recourse
    │
    ▼
Joint Mutable Assignment
    │
    ▼
MaskCO-Guided Per-Vehicle Resource Search
    │
    ▼
Service-First Fleet Guard
    │
    ▼
Event-Driven Execution
    │
    ▼
OR-Joint Distillation / Fleet Preference / DAgger
    │
    ▼
Full Joint Fleet Resource Beam
    │
    ▼
Neural LNS / Value-Aware Recourse
    │
    ▼
Stochastic Lookahead
    │
    ▼
Cold-Chain Quality-Energy Preference Control
    │
    ▼
Large-Scale / Tight-Fleet / Real-Road Validation
```

---

# 100. 最终执行口令

当前不要跳到更后面的复杂优化。

现在的唯一正确入口是：

\[
\boxed{
\textbf{
先把 JF1-H 的搜索语义冻结成可靠 baseline，
然后在原 MaskCO 模型内部新增 Fleet Assignment Head，
以 OR-joint 为当前信息集专家，
训练真正面向 FleetState 的 MaskCO assignment preference。
}
}
\]

也就是说：

> **下一步不是“做一个新的 fleet 模型”，而是“把 MaskCO 本身动态化到 fleet-level”。**

这条原则从现在开始锁死。
