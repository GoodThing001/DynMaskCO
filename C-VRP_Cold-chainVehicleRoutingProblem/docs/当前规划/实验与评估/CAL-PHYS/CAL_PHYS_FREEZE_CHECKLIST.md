# CAL-PHYS 从草案到冻结的执行检查表

> 当前状态：**DRAFT-BLOCKED**
>
> 执行人用途：按顺序完成现实信息、实验排程、数据冻结、拟合与独立验证。
> 强制原则：任何前置 Gate 未通过，不得越过；任何真实偏差必须留痕，不得覆盖原始文件或静默改表。

## 0. 当前文档包

| 文件 | 责任 | 当前状态 |
|---|---|---|
| `CAL_PHYS_SCOPE_DECISION.md` | v2-light 科学范围、方程和论文结论边界 | draft-blocked |
| `CAL_PHYS_PROTOCOL.json` | 实验单位、设计、分割、随机化、停止和防泄漏规则 | draft-blocked |
| `CAL_PHYS_SOURCE_REGISTRY.csv` | 标准、论文、厂商与实测来源登记 | metadata pending |
| `CAL_PHYS_RAW_DATA_MANIFEST.json` | 原始数据、设备、变换和 split 身份 | empty draft |
| `CAL_PHYS_RUN_SCHEDULE.csv` | 按 block 随机化的逐运行计划和偏差记录 | header only |
| `CAL_PHYS_ANALYSIS_PLAN.json` | 模型、拟合、holdout、验收和失败 verdict | draft-blocked |
| 本文件 | 人工执行和 freeze 门控 | open |

这些文件不是当前 pilot contract 的替代品。`pilot_registered` 只授权 P1 仪器/可行性 pilot，不能估计最终合同参数；只有“confirmatory 协议冻结 Gate”和“CAL-PHYS 完成 Gate”分别通过后，才能宣称正式预注册已冻结、现实校准已完成。

## 1. Step 11.9：源码字节基线 Gate

先执行并核对 `../../决策与审计/O0CC源码字节基线与行尾处置.md`。

- [x] 增加范围明确的 `.gitattributes`；活跃源文件 LF，冻结包 `-text`；
- [x] 仅规范化已识别的两个活跃 CRLF 文件；
- [x] 以原封存字节重新保存受 Git 过滤影响的 39 个归档文件；
- [x] 精确清理归档根内未跟踪 `.pyc/__pycache__`；
- [x] 当前工作树和 fresh worktree 均通过正式 v4 archive verifier；
- [x] 22 个身份文件满足 raw bytes 与 Git blob 相同；
- [x] 重跑 8 套门控测试，94/94 PASS；
- [x] 提交 D `3f2b7e3108a3d5c6486ba3b63bbbd0e197d167bc` 已完成，并已回填 sealed 机读源码基线；
- [x] 完整分支 Git bundle 已在服务器离机目的端通过 SHA、bundle、ref 和恢复 checkout 三重验证；
- [x] 离机证据已回填，`O0CC_SOURCE_BASELINE.json` 已转为 `sealed`；
- [ ] 文档提交 E 已保存 sealed 源码基线，且 E 的远端备份使用新 artifact，不覆盖 D bundle。

**失败动作：**保持 CAL-PHYS 为 `draft_blocked`。不得用 Windows 临时工作树 hash 代替 Git 可复现身份。

## 2. 现实对象定义 Gate

### 2.1 车辆、箱体与制冷设备

- [ ] 冻结车辆/箱体 manufacturer、model、尺寸、保温等级和有效容积；
- [ ] 冻结制冷机 manufacturer、model、额定工况、供能类型和控制模式；
- [ ] 说明三个温区是实物独立舱、可移动隔板还是算法代理；
- [ ] 确认共享容量与现实布局一致，否则新建运营合同版本；
- [ ] 将 capacity=50 映射为明确的 kg、箱、托盘或体积单位；
- [ ] 记录允许的 ambient、set-point、load 和 door-duration 安全范围。

### 2.2 代表性商品

每个温区独立完成：

- [ ] 商品、品种/等级、供应来源与成熟/初始状态；
- [ ] 包装形式、单件质量/体积和装载密度；
- [ ] 主品质指标及测量方法；
- [ ] 原始指标到 `initial_quality/quality_fraction` 的单调换算及批次级 (Q_0) 处理；
- [ ] 选择一阶、零阶或其他动力学的先验理由；
- [ ] 不可售阈值及法规、感官、理化或业务依据；
- [ ] 可获得的独立商品批次数；
- [ ] product value 是否有真实业务数据；无则明确 sensitivity-only。

**失败动作：**不能把温区标签本身当作商品。任一温区缺商品/指标/阈值时，该温区不能标为 calibrated。

## 3. 来源与设备 Gate

### 3.1 来源登记

补齐 `CAL_PHYS_SOURCE_REGISTRY.csv`。证据等级使用：

- A：同设备/同商品的原始试验或可复现实验论文；
- B：标准、法规或可比设备厂商规格；
- C：综述或间接可比来源，只用于界定范围；
- D：专家判断/假设，只能用于待验证先验或 sensitivity。

- [ ] 每个采用参数至少有一个 source_id；
- [ ] 保存访问日期、本地只读副本和 SHA-256；
- [ ] 记录原始单位、换算公式、适用域和不可转移条件；
- [ ] 厂商 cooling capacity 未被当成 electrical input power；
- [ ] 标准温度范围未被误写成具体车辆动态参数；
- [ ] 文献品质参数与选定商品、包装、指标和温区匹配。

### 3.2 测量系统

- [ ] 温度传感器型号、序列号、分辨率、精度和校准记录；
- [ ] 环境温度参考仪器；
- [ ] 电功率/累计电能表型号、精度和校准记录；
- [ ] 门状态或开门时长记录方式；
- [ ] 所有采集时钟的同步方法和最大偏差；
- [ ] 传感器在舱内的位置图和编号；
- [ ] 品质测量设备、方法重复性和盲化/编码方案；
- [ ] 用仪器 pilot 估计测量噪声和时间相关性。

### 3.3 P1 pilot 注册与执行

最终独立样本量和部分验收阈值依赖仪器重复性，因此允许在完整 confirmatory 协议冻结前执行一个受限 pilot：

- [ ] P1 只测试仪器、时钟、数据链路、可执行范围和重复性；
- [ ] P1 不估计或发布最终 contract 参数；
- [ ] P1 数据 role 固定为 `instrument_pilot`，永不迁入 calibration 或 dynamic_holdout；
- [ ] 单独生成受限 P1 schedule、seed、分析规则和 `CAL_PHYS_PILOT_FREEZE_SEAL.json`；
- [ ] 源码字节基线、车辆/箱体、仪器 identity 和 raw-data identity 规则已经关闭；
- [ ] 协议状态从 `draft_blocked` 变为 `pilot_registered` 后才执行 P1；
- [ ] P1 结果只用于传感器误差、时钟误差、run-level 方差、within-run correlation、可行范围和最终 n/阈值规划；
- [ ] P1 完成后保留原 seal，不把其结果回写成 confirmatory observation。

**失败动作：**修复测量系统并递增 pilot seal 版本；不得直接进入 P2–P6。

## 4. 因素水平和独立样本量 Gate

### 4.1 因素水平

- [ ] ambient 水平位于设备和实验安全域内；
- [ ] set-point 水平与三个温区现实运营一致；
- [ ] 至少包含空/低、中、高装载中的可执行层级；
- [ ] 门关闭基线与至少两个非零开门时长；
- [ ] 需要估计频次效应时，至少两个非零频次且与时长不完全混淆；
- [ ] cooling-off、cooling-on 拉温、稳态和恢复程序均有明确控制；
- [ ] 安全时加入重复 center-condition 检查曲率；
- [ ] required interactions 在设计矩阵中可估计，无完全 alias。

若资源不足以支持全组合，必须在收集前定义高分辨率 fractional/split-plot 设计及 alias structure。不得用 OFAT 结果声称已识别交互。

### 4.2 独立样本量

- [ ] thermal/door/energy 的 n 按独立运行计；
- [ ] quality 的 n 按独立商品批次计；
- [ ] 传感器、时间点和同批次样品仅作技术/重复测量；
- [ ] 根据仪器 pilot 的方差、run/lot 内相关性和预期精度计算独立 n；
- [ ] P1 分析计划在查看 pilot outcome 前已规定保守方差规则，不直接使用不稳定的 pilot 点估计；
- [ ] 按全部主响应族中所需独立 n 的最大值规划，并计入 block/cluster 结构与预先规定的非 outcome 运行损失余量；
- [ ] 单独保证 dynamic holdout 的独立运行和独立批次，不从 calibration 行中抽样；
- [ ] 将计算输入、方法、代码、输出和最终 n 保存并 hash；
- [ ] 将最终 n 写入 protocol，之后不因拟合或方法结果缩减。

如果无法达到所需独立 n，应返回 `DATA_INSUFFICIENT`，不能用增加采样频率替代。

## 5. 随机化与运行表 Gate

填写 `CAL_PHYS_RUN_SCHEDULE.csv`：

- [ ] 为每个独立运行分配唯一 `run_id` 和 `independent_unit_id`；
- [ ] vehicle/compartment、day/session、operator、sensor package、product lot 均有 block id；
- [ ] 明确 hard-to-change whole-plot 与 easy-to-change subplot 因素；
- [ ] 在可执行 block 内用固定 seed 随机化顺序；
- [ ] 不让处理条件与日期、设备、人员或传感器完全重合；
- [ ] 预先分配 calibration、internal_diagnostic、dynamic_holdout；
- [ ] holdout 是完整运行/完整批次；
- [ ] 保存生成脚本/工具版本、seed、CSV SHA-256；
- [ ] 冻结后只填写 actual time、deviation code 和 record status，不改计划因素；
- [ ] 未执行、终止或失败运行也保留原行，不复用 run_id。

## 6. 分析与验收阈值 Gate

在查看 confirmatory outcomes 前补齐 `CAL_PHYS_ANALYSIS_PLAN.json`：

- [ ] 明确拟合软件、版本、优化器、容差、初始化和参数边界；
- [ ] 明确时间对齐、插值方法和最大允许 gap；
- [ ] 明确舱内多传感器的主汇总量及空间离散诊断；
- [ ] 明确每个响应的误差尺度和权重依据；
- [ ] 明确 run/lot 层 cluster bootstrap 或模型区间方法、seed 和重复次数；
- [ ] 温度、拉温时间、恢复时间、能耗和品质阈值均已有数值；
- [ ] 阈值依据来自仪器精度、pilot 重复性或运营容忍度；
- [ ] 阈值未按任何已拟合物理模型的通过/失败结果调整；误差类主终点使用预先规定方向的置信界；
- [ ] 全部主终点采用 intersection-union 式“逐项全过”逻辑；若论文声称同时覆盖，已冻结 simultaneous interval 方法；
- [ ] 参数可识别性、覆盖率和物理不变量有数值/机读判定；
- [ ] v1 fallback 的触发和论文降级声明已经冻结；
- [ ] 不存在用 composite 平均掩盖某一响应族失败的规则；
- [ ] 不存在读取路由方法结果的模型选择字段。
- [ ] CAL-PHYS 不拟合 objective scale、lambda 或 routing-performance-selected product value；这些字段留到 Step 15 新 DEV-CAL。

## 7. Confirmatory 协议冻结 Gate（P2–P6 数据采集前）

P1 可在 `pilot_registered` 状态下按 §3.3 执行。只有以下全部通过，才将两个 JSON 的 `status` 改为 `frozen` 并授权 P2–P6：

- [ ] `CAL_PHYS_PROTOCOL.json.blocking_fields` 已清空；
- [ ] `CAL_PHYS_ANALYSIS_PLAN.json.blocking_fields` 已清空；
- [ ] raw-data manifest 中待采集 dataset identity 规则已完整；
- [ ] run schedule 无重复 id、无必填空值、split 角色合法；
- [ ] 统计单位与实验单位一致；
- [ ] 所有 JSON 可由标准解析器读取且无 NaN/Infinity；
- [ ] CSV 使用 UTF-8、固定列、无公式单元格；
- [ ] 所有引用文件路径存在并计算 SHA-256；
- [ ] 源码基线（含离机备份证据）已封口并在 Git 中保存；
- [ ] 在全部协议文件定稿后，生成独立的 `CAL_PHYS_PROTOCOL_FREEZE_SEAL.json`，记录所有文档、来源副本、运行表、分析计划和源码基线 hash；协议正文不保存自身 bundle hash；
- [ ] 在只读副本中重新计算 bundle hash 完全一致；
- [ ] 冻结时间、执行人和复核人已签名/记录。

冻结后任何实质修改必须递增 protocol version、保留旧版本并说明原因。不得覆盖原文件。

## 8. 数据采集顺序

### P1 Instrument

- [ ] 本阶段已经在独立 `CAL_PHYS_PILOT_FREEZE_SEAL.json` 和 `pilot_registered` 状态下执行；若 confirmatory 协议已经冻结，此处只做账本复核，不重复使用 P1 数据拟合最终参数；
- [ ] 传感器校准、重复性、位置差和时钟同步；
- [ ] 电表与参考负载对账；
- [ ] pilot 结果只用于精度与样本量，不估计最终合同参数。

### P2 Passive

- [ ] 制冷关闭、门关闭；
- [ ] 跨 ambient、初始温度和 load；
- [ ] 保存每个独立 run 的完整原始轨迹。

### P3 Cooling

- [ ] 独立预冷/拉温；
- [ ] 稳态制冷能力与输入功率；
- [ ] ambient × set-point 覆盖；
- [ ] 预冷能耗与运行能耗分开计量。

### P4 Door

- [ ] 门关闭 concurrent baseline；
- [ ] 多档 door duration，必要时多档 frequency；
- [ ] 记录峰值温升、恢复时间和恢复电耗；
- [ ] 门事件与其他热负荷时间戳可分辨。

### P5 Quality Isothermal

- [ ] 每商品至少三个适用域内温度；
- [ ] 独立商品批次分散到温度程序和处理批次；
- [ ] 测量者尽可能只见样品编码；
- [ ] 样品链路、温度记录和 assay 原始值完整。

### P6 Dynamic Holdout

- [ ] 使用未参与拟合的完整车辆运行；
- [ ] 使用未参与拟合的完整商品批次；
- [ ] 包含路线式行驶—服务—开门—恢复和变温品质轨迹；
- [ ] 一次性运行冻结分析，不调参、不改阈值、不回填模型。

## 9. 数据入库与偏差处理

- [ ] 原始文件立即进入只读区并计算 SHA-256；
- [ ] 更新 raw-data manifest，不修改原始文件；
- [ ] 所有清洗/换算只生成 derived 文件；
- [ ] derived 文件能从 raw + versioned transform 完整重建；
- [ ] 缺失、非有限值和同步失败按协议 fail-closed；
- [ ] 排除必须有与 outcome 无关的证据和 deviation code；
- [ ] 被排除数据仍保留在 manifest 和诊断报告；
- [ ] 不根据残差大小、拟合改善或路由效果删除运行。

## 10. CAL-PHYS 拟合与 verdict

按固定顺序：

1. protocol/data integrity；
2. measurement/unit；
3. parameter identifiability；
4. physical invariants；
5. untouched nominal dynamic holdout；
6. sensitivity contract construction。

允许的唯一 verdict：

- `GO_CAL_PHYS`：全部主 Gate 通过；
- `DATA_INSUFFICIENT`：独立数据不足或参数不可识别；
- `MODEL_INADEQUATE`：方程对 holdout 有系统性失配；
- `MEASUREMENT_FAILURE`：仪器、同步或样品链路不足；
- `PROTOCOL_ERROR`：泄漏、身份、分割、排除或冻结规则被破坏。

不得设置模糊的“基本通过”。敏感性结果不能把 nominal 的失败改成 GO。

## 11. CAL-PHYS 完成 Gate

- [ ] 输出完整参数、单位、区间、边界和 source_id；
- [ ] 输出每条件、每 block 的独立 n；
- [ ] calibration 与 untouched holdout 分开报告；
- [ ] 输出全部 residual、identifiability 和 invariant 检查；
- [ ] 输出全部排除和偏差；
- [ ] 输出 v2-light/v1-effective 决策及论文结论降级情况；
- [ ] 冻结 nominal、low-stress、high-stress 合同；
- [ ] 三合同均有 provenance、文件 SHA 和 contract hash；
- [ ] 输出 CAL-PHYS result manifest 与 evidence seal；
- [ ] 在读取新 DEV 方法效果前完成以上冻结；
- [ ] 状态文档只在证据 seal 验证后更新为 CAL-PHYS GO。

## 12. GO 后交接到 Step 14b–18

`GO_CAL_PHYS` 后严格执行：

```text
Step 14b  实现/冻结 contract vNext，重跑物理与门控回归，重算代码身份
Step 15   新 DEV-CAL baseline service audit → 新 profile/scale
Step 16   独立新 DEV-GATE 重确认 sequential GO
Step 17   一次性登记 VAL root_seed/manifest/statistical plan 并封存 source archive
Step 18   服务器 byte-preserving 搬运 → verify → tmux one-shot VAL
```

只有 calibrated VAL GO 才关闭 O0-CC 并进入 M0；只有 M0 GO 才进入论文主模型 M1。
