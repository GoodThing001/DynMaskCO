# 新正式运行的原始工件屏障与工程证据边界

本目录是 2026-10-11 的独立审查交付，未修改原183文件、旧数值结果、当前共享正文或实际现行门。未运行模型、训练或新的数字实验，也没有 SSH 或上传。

## 两套配送工程证据

`delivery/remote_actual_bundle_v2/downloaded/ACTUAL_FULLSTEP_QA.json` 描述真实原CVRP checkpoint（SHA `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`）的76,259,840参数、234叶、完整16层encoder/6层decoder的一次固定更新。源码始末、234叶的形状和参数摘要、梯度统计、保留边及C0投影保存在实际JSON。checkpoint step metadata为0不能据此称随机初始化；该报告另外核对与原配置初始化不同。该单步是重建CE工程验收，不是训练收益。

`remote_features_bundle_v1/downloaded/FEATURES_OBJECTIVE_QA.json` 将3×512投影扩成10×512，增加3,584参数，总计76,263,424，其余233叶初始化摘要保持。两个分支从同一初始化固定各一步：CE与CE+0.2×成对偏好。两个完整模型都记录234叶更新、160 encoder及60 decoder非零有限梯度；新增属性移除后的logits差约22.54。零扩展初始3D/10D映射最大logit误差约4.11e-5。能学习、属性有影响和代理loss改变不证明在线收益。

实际保存权重分别为 `b3eecbc7ed3949bbba9a6606abeec3a703c5bc83141a1477186b3dfaab0b09fb` 与 `7d1d49ab9778d57afc63652fd6a93d5e269c208bbcd5e99f24c7d00da5a3adf2`。根助手的 `INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json` 独立读取两份各305,075,460字节的服务器实际文件；这是独立完整字节身份。保存后重载logits相同来自运行器执行报告，本审查没有独立本地加载保存tensor或重新前向，不混称这两个支持等级。

根助手10D独立报告v2为330检查/0失败，含原TRAIN数据再生、完整C0路径与特征/目标复核；v1的一项tuple/list审核工具误判保留，未重复模型运行。原3D根报告528检查/0失败。此处脚本再次独立核存储形状元素数、参数统计、损失分解与资格边界，既不冒认再次独立运行模型，也不覆盖根报告。

这些QA用的是一个已观察、全部release/ready的旧教师快照。新增温区目标温度和类别初始品质是合同属性，没有实际舱温/货温测量；类别固定品质rate不是温度轨迹依赖校准。原模型配置bfloat16而实际适配运行float32。softcap/final_softcap为null，只支撑此次配置边界；当前CPU attention忽略finite softcap的问题不能被推广成所有配置核一致，也不能否定本null配置。

## 取货公共模型包

`pickup/actual_masked_fleet_engineering_v1/actual_qa_v1` 保存原完整CVRP模型、同架构随机与discard提案在一个已观察公共包的工程记录。模型通过已知订单的部分邻接进行重构；最多两车/5个mutable订单，承诺腿与保留边约束由公共证书审查。原3D模型没有时间窗、温区、当前位置或品质直接输入，物理合法性和已知尾段评分由C0/A1执行。公共包的future-mutation验收有局部证据，不等于整日因果验证。

`resident_qa_v1` 保存实际worker预热、合法提交及超时worker回收/保留原incumbent的工程验收；`wiring_qa_v1` 的完整归档日接线使用三个人工历史日，`actor_mode=null`，不等于完整模型跑过整日。legacy admission仍是同步调用，晚提交回滚不能保证10秒后计算被强制终止；clock0的证书覆盖与中途A1也有不同。新正式RUN必须先把所有对照和接单计算放入同等公共worker预算封套，不得只给模型worker硬截止。

## 新RUN必须在生成前固定

1. 唯一新目录、原PROGRAM SHA、假设、候选冻结规则、family序号、全部已看开发/QA日清单、不可重复使用旧结果与最终20260927/28。外部可信注册应有时间与权限记录；本地JSON的日期字段不能证明生成前注册。
2. 明确模型完整架构、实际attention路径/算子、dtype、原初始化与映射、训练/重建控制的相同样本顺序/优化器/步数、最后一步保存、实际weights全字节SHA。训练步数及参数统计不能代替逐步实际参数与梯度工件。
3. 同合同的release/ready、货物生命周期、已承诺腿、装卸、返仓、容量、多车归属、能耗和品质单位；原C0的固定类别rate边界与未来修订必须明确，不得用仿真温区类别声称物理校准。
4. 固定实际history池、同未来law与场景RNG，全部臂同初始incumbent、合法动作与完整评分。取货必须有cond_hist及同初始化纯重建控制；配送必须有随机同架构、非学习控制和合格强滚动。baseline名称和recipe文件存在不证明控制已合格。
5. 所有臂10秒从admission入口开始，包含公共快照、抽样、推理、搜索、认证、提交和相关I/O。仅注册且事前完成的固定合成预热/加载可排除。逐请求保存单调时钟、父子PID、截止、kill与fallback记录；独立监督实际观察这些事件。
6. 三个fresh40完整日cohort，日/臂唯一原始事件、模型调用输入/输出、进程预算和所有失败/超时行完整保留。不能把一日内状态或重复种子当独立日。
7. family j两侧alpha `.05/(j*(j+1))`，再等分取货/配送、注册比较和3cohort。固定日配对bootstrap端点与次数，在实际每尾期望MC计数≥25；2000次95%只能描述探索，通常不够确认alpha。40日是资源边界，须预先讨论可检测效果与精度，不是已有功效证据。
8. 取货主门效用差≥20且正下界，新增学习目标相对同初始化纯重建的正差/正下界；配送相对每个注册随机/非学习/合格强滚动完整成本减少≥1%且正下界。三个cohort均过、独立完整资格通过才谈稳定；正工程或一次开发优势不替代确认。

## 可运行的屏障

`new_run_gate_adapter.py` 接收冻结project、prospective registration、外部registry和逐日raw bundle。检查原PROGRAM绑定、受保护种子、三40日覆盖、familyalpha/MC端点、控制角色和阈值、外部注册13种身份角色、实际文件SHA、逐日事件/模型/进程记录入口与预算算术。JSON重复键、NaN、真假字符串、跨日重复raw或改变的工件均拒绝。输出路径必须在冻结project之外，采用exclusive新建。

```powershell
python -B new_run_gate_adapter.py --project <frozen-project> --registration <new-registration.json> --registry <external-registry.json> --bundle <raw-package.json> --output <new-output.json>
python -B -m unittest -v test_new_run_gate
```

返回0仅表示**必要设计子集和字节/覆盖/已声明预算记录算术通过**。输出始终保留以下为false：独立实际激活、独立因果、实际process-kill证明、完整轨迹独立重放、生成前注册时间、正式在线资格、MaskCO稳定核心在线增益。没有任何文件presence或storedledger能把这些字段置true。

后续应按注册原始模型调用输入重新执行完整encoder/decoder，对照实际logits/hash与mask反事实；从实际保存权重独立加载核参数再前向；监督执行预算并核递归PID；用独立语义实现逐日重放冷链、事件及效用；最后独立从合格完整日重算配对CI。这些分别出具源码与输入SHA绑定的执行报告，再交科学裁决器，不由本package adapter自授资格。

上一交付的 `followup_20261011/support_replay_bridge.py` 可执行固定实际事件validator/取货observer接口，支撑语义重放的一部分工程闭包。它复用既有实现，不能称独立物理实现或完整训练复现。此处新门明确保存这一边界。

## 单实例三项数字来源

目标是寻找23.4229距离、2213.5795能耗、2.2939品质损失的**共同原始工件绑定**。它们的成本算术约247.07475，距离份额约9.48%可以复算；这不提供来源身份或所有状态的上界。`ACTUAL_PACKAGE_SCOPE_AUDIT.json` 记录搜索目录、命中文件/哈希、未解码二进制清单和最终unknown。无文本joint候选不证明全项目不存在该组合；没有定位并语义复算之前保留unknown，不将single实例乘cohort平均距离headroom当完整效用上界。
