# CAL-PHYS 范围决策：coldchain-contract-v2-light

> 文档状态：**DRAFT-BLOCKED，不是已冻结预注册**
>
> 版本：0.1.0-draft
>
> 日期：2026-09-13
> 目的：在读取新 DEV/VAL/TEST 方法效果之前，冻结现实冷链模型的科学范围、可识别参数和失败边界。

## 1. 决策摘要

CAL-PHYS 推荐采用 `coldchain-contract-v2-light`，而不是直接把现实数据填入现有 `coldchain-contract-v1`。v1 可作为固定工况下的等效常数回退模型，但不足以支撑论文对动态热过程和能量一致性的主张。

v2-light 只增加当前论文结论所必需的结构：载荷相关有效热容、与开门时长和温差相关的开门热、热制冷能力—COP—电输入的一致关系，以及按商品/品质指标标定的品质动力学。它不引入空间温度场、CFD、湿度、太阳辐射、压缩机瞬态控制或时变天气。

最终选择只能由 CAL-PHYS 独立物理 holdout、参数可识别性和预注册验收规则决定，不能由 O0-CC、M0 或 M1 的方法效果决定。

## 2. 保持不变的运营合同

以下语义已经冻结，CAL-PHYS 不得静默修改：

- 业务模式：dynamic cold-chain pickup-to-depot；
- 车辆空载出发；订单在 `service_finish` 取货入舱并开始品质计时；
- 订单在 `return_arrival` 卸货并结束品质计时；
- 同构多温舱、共享总容量；
- 单次行程、禁止 reload；
- hard feasibility 与 D/Q/E/J 分开报告；
- 连续品质损失是主连续量，`num_unsalable` 是阈值派生量；
- nominal 是唯一主分析，low/high 仅为 non-gating sensitivity。

若现实车辆不符合共享容量、同构多温舱或单次行程，应另建运营合同版本；不得只调整热参数来掩盖运营语义差异。

## 3. v2-light 热学结构

### 3.1 环境与工况

每个冻结合同或场景使用一个显式、常数的环境温度 `ambient_temperature_c`。CAL-PHYS 数据应覆盖 ambient × set-point 工况，但 v2-light 不建模单次路线中的时变天气。

论文主结果使用预注册的 nominal 合同；low-stress 和 high-stress 是预先固定的完整场景合同，不是根据方法胜负逐参数调出的极端组合。

### 3.2 自然换热与有效热容

每个温区 (z) 使用：

\[
C_{z,eff}(L_z)=C_{z,empty}+c_{z,load}L_z,
\]

\[
\frac{dT_z}{dt}=\frac{UA_z}{C_{z,eff}(L_z)}(T_{amb}-T_z)
\quad\text{（制冷关闭、门关闭）}.
\]

其中：

- `ua_kw_per_c[z]`：温区等效传热系数；
- `empty_thermal_capacity_kwh_per_c[z]`：空舱、壁面和固定设备的等效热容；
- `cargo_thermal_capacity_kwh_per_c_per_load_unit[z]`：单位载荷附加热容；
- `zone_load` 使用运营合同中的同一容量单位。

不得在只有自然升温轨迹、没有独立热容信息时同时自由拟合 `UA` 和两个热容参数。若数据不足，应固定来自独立试验/来源的热容，再估计 `UA/C`，或回退到 v1 等效 `heat_transfer_per_hour`。

### 3.3 开门热

每次服务开门的热量采用：

\[
Q_{door,z}=\beta_{door,z}\,t_{open}\,\max(T_{amb}-T_z,0),
\]

其中 `door_heat_transfer_kw_per_c[z]` 是等效参数，`t_open` 必须来自实际服务/开门记录。温升由 (Q_{door,z}/C_{z,eff}) 计算。

本模型只描述环境高于舱温时的显热侵入。若存在环境低于舱温、潜热/除霜或湿度主导工况，应标为超出 v2-light 适用域，不得外推。

### 3.4 制冷能力与电耗

v2-light 不再独立使用“温降速率、输入功率、COP”三套互不约束的参数。制冷开启时：

\[
\dot Q_{cool,z}=\mathrm{clip}(a_z-b_z\Delta T_{lift},0,Q_{rated,z}),
\]

\[
COP_z=\max(COP_{min,z},COP_{0,z}-s_z\Delta T_{lift}),
\]

\[
P_{input,z}=\dot Q_{cool,z}/COP_z.
\]

温度变化使用热制冷能力除以有效热容，能耗使用输入功率积分。若未同时测量/约束热制冷能力、输入功率和热容，不允许声称分别识别了全部参数。

预冷能耗保持独立字段，由环境温度拉至设定点的独立试验积分获得，不与运行制冷或开门恢复能耗重复计入。

### 3.5 v2-light 建议字段映射

| 参数族 | 建议字段 | 单位 | 主要证据 |
|---|---|---|---|
| 环境/目标 | `ambient_temperature_c`、`target_temperature_c[z]` | °C | 实际工况与设备设定 |
| 硬边界 | `hard_temperature_bounds_c[z]` | °C | 法规、产品要求或运营安全标准；不是用轨迹拟合出的容差 |
| 被动换热 | `ua_kw_per_c[z]` | kWth/°C | cooling-off、closed-door 独立运行 |
| 空舱热容 | `empty_thermal_capacity_kwh_per_c[z]` | kWhth/°C | 空舱阶跃/拉温试验与设备资料 |
| 货物热容 | `cargo_thermal_capacity_kwh_per_c_per_load_unit[z]` | kWhth/(°C·load unit) | 多装载水平独立运行与货物物性 |
| 开门热 | `door_heat_transfer_kw_per_c[z]` | kWth/°C | 多时长、温差和装载的门事件运行 |
| 制冷能力 | `cooling_capacity_intercept_kw[z]`、`cooling_capacity_lift_slope_kw_per_c[z]`、`rated_cooling_capacity_kw[z]` | kWth | 同设备 capacity 曲线与温度轨迹 |
| COP | `cop_intercept[z]`、`cop_lift_slope_per_c[z]`、`cop_minimum[z]` | dimensionless | 同步热能力与电输入测量 |
| 预冷 | `dispatch_preconditioning_energy_kwh[z]` | kWh/dispatch | 独立预冷电表积分 |

最终代码字段名可在 Step 14b 实现前做一次无语义重命名，但字段含义、单位和方程若改变，必须递增 CAL-PHYS protocol version。

## 4. 品质动力学范围

每个温区必须先固定：代表性商品、品种/等级、包装、装载密度、初始状态、主品质指标、测量方法和不可售阈值依据。“常温/冷藏/冷冻”不是足够的商品定义。

默认候选模型为一阶剩余品质：

\[
Q(t+\Delta t)=Q(t)\exp[-k(T)\Delta t],
\]

\[
k(T)=k_{ref}\exp\left[\frac{E_a}{R}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)\right].
\]

要求：

- 每个商品至少覆盖三个位于适用温区内的等温条件；
- 独立商品批次才是生物/材料重复，单批次多个时间点只是重复测量；
- 整个批次或整条变温轨迹作为 holdout，禁止逐行随机拆分；
- 若一阶模型在物理 holdout 上不通过，应判 `MODEL_INADEQUATE` 并新建协议版本；不能只替换参数继续使用一阶方程；
- `reference_temperature_k` 推荐按温区/商品设置，以减少远距离外推和参数相关性；
- 每个批次必须保存原始品质指标到归一化 `initial_quality`/`quality_fraction` 的单调换算；不得默认所有批次真实初始品质均为 1；
- `product_value` 没有真实业务损失数据时只能作为 sensitivity weight，不得称为经济成本。

目标温度和 hard temperature bounds 来自所选商品、法规/标准和现实运营要求，不是通过最小化温度预测误差拟合出的自由参数。预测模型通过并不自动证明 hard bounds 合规；合规性必须使用实际轨迹和独立约束来源检查。

## 5. 单位与空间尺度

在冻结合同前必须同时闭合：

- 数据坐标到 km 的映射；
- 数据时间到 hour 的映射；
- 速度与距离/时间的一致性；
- capacity=50 对应的现实单位；
- 载荷单位到热容增量的映射。

必须通过端到端 trace 对账，防止 `distance_km_per_unit`、`hours_per_time_unit` 和 `speed_kmph` 重复换算。

### 5.1 CAL-PHYS 与 objective profile 分离

CAL-PHYS 可以校准物理单位、D/Q/E 的生成与累计语义，但不使用路线方法结果拟合 `distance_scale`、`quality_scale`、`energy_scale`、`lambda_quality` 或 `lambda_energy`。这些 objective profile 字段必须在 calibrated contract 冻结后，由 Step 15 的新 DEV-CAL baseline service audit 按预声明规则重新估计/冻结。

`product_value` 只有在存在独立业务价值/损失成本证据时才进入 nominal；否则保持 sensitivity-only。任何通过调 scale、lambda 或 product value 恢复 O0-CC GO 的做法都属于方法结果泄漏。

## 6. 明确排除的模型内容

本轮不建模：

- 舱内空间温度梯度和货物核心温度的三维场；
- 湿度、凝露、潜热、除霜和风机局部流场；
- 太阳辐射、风速、道路坡度和车辆动力系统耦合；
- 压缩机启停瞬态和详细控制器；
- 多次返仓补货、reload 或每温区独立容量；
- 未选定商品的通用品质结论；
- 基于路线方法表现反向校准物理参数。

这些排除项必须在论文局限性中公开。若任何排除项对 holdout 误差构成系统性主因，应升级合同版本，而不是扩大 v2-light 参数自由度。

## 7. v1 回退条件

只有同时满足以下条件，才允许回退 `coldchain-contract-v1`：

1. v2-light 的新增参数因独立实验单元不足而不可识别；
2. v1 在预注册的整运行动态 holdout 上通过验收；
3. 论文将参数明确表述为特定工况下的等效常数；
4. 不再声称热容随装载、开门热随时长或能量链条已被分别识别；
5. 回退决定在读取新 DEV/VAL/TEST 方法效果前写入 CAL-PHYS verdict。

不得在 v2-light 和 v1 之间按 O0-CC Δ 择优。

## 8. 当前阻断字段

以下现实信息未提供，因此本文件尚不能冻结：

- 车辆/箱体/制冷机具体型号及温区结构；
- 三温区代表性商品、包装和主品质指标；
- capacity/load 的现实单位；
- 可用环境舱或自然环境范围；
- 温度、电功率、门状态和品质测量设备及精度；
- 可获得的独立车辆运行数、日期 block 和商品批次数；
- ambient、set-point、load、door-duration 的安全因素水平；
- 物理验收阈值及其传感器/业务依据。

上述字段补齐、协议和运行表生成、源码基线（含提交 D 与离机备份证据）转为 `sealed` 后，才允许把本文件状态改为 `FROZEN`。

## 9. 论文可支持的结论边界

CAL-PHYS 通过后可以支持：在明确车型、工况、商品和测量误差范围内，v2-light 能以预注册精度重建温度、能耗和品质轨迹，并为路由评价提供独立校准的合同。

CAL-PHYS 不能单独支持：方法优于基线、适用于所有冷藏车/商品、具有通用经济最优性，或达到真实部署安全认证。
