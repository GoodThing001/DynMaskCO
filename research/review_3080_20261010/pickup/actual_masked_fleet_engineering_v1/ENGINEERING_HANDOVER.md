# 真实 CVRP MaskCO 多车公共后缀工程交付

本目录是新的独占工程原型。旧 I3、D2、风险及 StageA 封存结果不改写，最终 seed20260927/28 未触碰。它补齐模型提案接线与物理证书，尚未建立稳定正向增益，也不证明收益空间上界。

## 已交付及实际证据

`masked_fleet.py` 实际加载原 CVRP 完整 encoder+decoder 的边 logits；从已揭示未服务订单中选至多5个可变订单，保留 committed nextstop、未掩码邻接边和其他车辆路线。提案可跨两个公开可用车辆重指派。每个完整候选由实际 simulator/C0/A1 接续当前位置、舱温、manifest、历史轨迹和全车已耗能量进行认证；仅用已知订单立即闭合的品质/燃油/拒绝损失 J 决定提交。私有整日未来 J 不进入提案或筛选。

`actual_qa_v1` 是此前已完成的三模式实际模型 QA，本轮没有重跑。234个真实参数叶已加载，预训练与随机同架构seed42参数及 logits 均不同。固定公共状态clock1.7700042724609375、committed43、可变20/17、4节点3维特征；maskseed202610101903/.3 删除17。预训练和随机各完成一次重构，都提出跨车候选已知 J=75.25672659277916，低于自身 incumbent75.48180642724037；两者均保留 incumbent。discard控制也保留 incumbent。真实单调用约0.15秒；这不是新在线收益实验。

`resident_actor.py` 新增常驻预热子进程，只接收 public packet。固定登记的图规模预热在日运行前完成，未登记图规模拒绝调用，不动态排除新 JIT。公共投影、IPC、源/权重复核、推理、候选 C0/A1 评分及最终提交使用同一单调绝对 deadline；超时 kill 并回收子进程，保留自身 incumbent。`live_hook` 可直接安装到真实 `env.snapshot_hook`，与最后同clock admission共享剩余10秒，提交前重新比较 public packet hash、认证 committed 和完整订单覆盖。默认仅一次已看状态工程干预；正式全日需显式 `max_interventions=None` 并事前登记。

`resident_qa_v1` 在新登记后只验收上述新增常驻worker功能：一次预训练公共状态合格响应，另一次.001秒真实强制超时成功kill/reap且保持原路线。不是此前三模式QA的再抽样，也未生成新训练数据、权重或在线日。`REGISTRATION.json` 与实际源原件均保存；后续严格数值参数类型/未来start拒绝修订不覆盖原件。

`fair_wiring.py` 的 `build_matched_replanner` 给 `strong_rolling`、`own_uncond_dynamic` 和 SAA 显式传入同一非空history、K10、容量、booking horizon、预算、cooling_share及版本化contract；SAA保留同hist/k并关闭部分场景投票，接单须记录完整20个accept/reject场景分支。`TraceAwareStrictEnv` 在真实车辆推进后只记录当前公开物理状态供认证使用。全部新请求按reveal_time、公开ID逐一处理，从每次入口计时，采样及认证包含在10秒提交预算中。批量reveals改为逐单调用的语义必须预声明，不能混用旧批量结果。

`wiring_qa_v1` 新登记后复用已看 seed202610101801 的完整80单/3车日，以固定独立synthetic3day历史fixture验收两条非学习控制器；fixture不取自被评估日，也不是I3正式200日历史池。两条各80次admission、0回退，C0/A1 hard=True，完整trace独立重放一致；同history hash成立。每次最大墙钟分别约.134/.196秒。J值只在完整原件中作工程回放，不据此做方法排名、统计推断或新收益结论。

## 运行接线

在新目录/新协议中先调用 `make_c0_contract_v2()`，显式确认 distance=20km/unit、speed=30km/h、hours=1h/unit，env.tw_speed=1.5。cooling_share由调用者固定登记，属于仿真参数，不伪称物理校准。

1. 以同一个实际历史池构造两条非学习对照和固定共同future sampler的SAA；所有arm记录history/data/weight/contract/source SHA。
2. 调用 `env_cls=make_trace_env_class(scripts)`；环境与replanner必须持有同一个contract对象。
3. 在日开始前构造 `ResidentActor(scripts,upstream,checkpoint,mode,registered_shapes)`；模式为pretrained/random/discard，权重只读固定。`with`/finally关闭worker。
4. 安装 `env.snapshot_hook=live_hook(actor,log,max_interventions=...)`。masked/random/discard使用相同基础admission controller、共同场景law、maskseed、一次提案日程和剩余预算；own incumbent必须独立保留。
5. 写完整日级raw trace、admission日志和public proposal/certificate，日终由独立evaluator重放 full J。公共actor不得读取日终fullJ选择下一轮。

工程命令（由主助手安排；使用新输出目录）：

```text
python -B test_actual_wiring.py --scripts <extension/scripts>
python -B run_wiring_qa.py --scripts <extension/scripts> --archive-trace <already-observed-baseline_trace.jsonl> --out <new-output>
python -B run_resident_qa.py --scripts <extension/scripts> --upstream <MASKCO_code> --checkpoint <ckpts/cvrp100.ckpt> --packet <public_packet.pkl> --out <new-output>
```

`run_wiring_qa.py` 支持 `--mode pretrained|random|discard --upstream ... --checkpoint ... --shapes ...`，实际完整日可接模型hook；本轮仅非学习整日接线及一个公共包worker工程QA，未运行此可选全日模型变体。

## 正式资格尚缺

原3维CVRP输入仅坐标/需求，没有TW、当前车辆位置、温区、品质或能耗资源编码；物理约束由真实证书执行。这是掩码生成提案工程接线，不是已经训练出的动态冷链核心策略。

本地实际上游 `modules/MultiHeadAttention.py::attention_fn` 丢弃 `softcap`，这是非null配置的潜在兼容风险。本实际同SHA checkpoint的实际模型检查文件确认softcap及final_softcap均null、qk_norm=true，故不能据此判定本次原CVRP算子改变或阻塞资格。原config dtype=bfloat16，现有只读加载provider显式使用float32适配，须明确记录；全234叶加载/完整encode+decode运行与原路径同输入数值验证仍是不同证据。`ACTUAL_CONFIGURATION_EVIDENCE.json`保存该实际检查文件SHA、checkpoint SHA和对应配置；原检查文件整体保留。

legacy rolling/SAA仍同步调用，本wrapper可阻止晚提交并回滚，但尚不能在10秒强制kill其CPU计算。masked worker已具备真实kill验收；全arm hard-compute同预算正式资格仍需将legacy admission也包在公共状态worker中。clock0使用原known-C0证书，clock>0具备完整all-fleet A1/C0；正式全日还需clock0同等A1覆盖。不能把该差异藏成全流程已经工程PASS。

当前rolling是接单+relocate/swap的非学习proxy，不是完整品质J oracle；同历史接线修正避免弱历史基线，但其优化强度仍需登记搜索空间、优化目标和预算。known immediate-close评分没有未来机会成本，选出的局部改善不保证整日正向。小两车/至多5单不能代表全车队全局上界。

尚未执行fresh四/多arm日级配对主门、独立2000次bootstrap、双开发种子复验或参数/算子完全结束核验。正式实验须用未看开发cohort与同基础SAA、公平强rolling、随机同架构、discard控制；不得复用工程日选择参数，不能删超时/失败日。

## 论文依据与取舍

[MaskCO官方ICLR2026论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)支持部分解掩码重构与迭代推理这一职责。本原型保留真实decoder/部分邻接而非只取encoder表征；论文不保证动态冷链迁移增益，也未提供本系统阈值。

[Ulmer等路线型MDP作者原文](https://justin-goodson.com/papers/RouteBasedMDPs.pdf)将路线计划和当前状态纳入决策，支持公共当前状态/计划及不可变下一腿的明确合同。本系统 manifest、温度和C0接续是应用扩展，并非论文已校准参数。

[Zhang等DVRPSR原文](https://arxiv.org/abs/2202.12983)以剩余资源估计reward-to-go并处理接单与分配，提示立即闭合J缺少机会成本。其服务数量目标不能替换本项目品质收入−燃油−拒绝损失的fullJ。共同future law、同历史公平控制和等待fresh完整日裁决是本工程据此作出的实验设计推论，而非引用论文保证收益。
