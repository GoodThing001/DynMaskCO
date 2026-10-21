# 算法v0开发检查

最新完整证据：[dev09＋dev10结果](../../../docs/static_delivery_research_20261021/dev09_dev10完整结果.md)、[封存v1](../algorithm_baselines/v1_dev09_dev10_validation/comparison.json)。四个1440步模型、456份教师与32份验证路线核验通过，原超时124及补跑0保留。当前dev11按[诊断设计](../../../configs/static_delivery/diagnostic_prereg_v1.json)执行贡献对照；[账本](development_ledger.json)已修正旧“仅1组”的预算记录，原字节快照保留于algorithm_diagnostics_20261021。以下为各早期阶段记录。

科研问题与评价规则已封存，算法仍处开发阶段。当前没有正式外层测试或方法优势结论。

- [算法说明与边界](../../../docs/static_delivery_research_20261021/算法阶段_v0.md)。
- `resources.json`：本机无GPU、可用内存约2.3GB；技能的Windows磁盘检测ABI失败后，用标准库补足磁盘信息，未改技能源码。
- `classical_run01/`：最初同目标ALNS/VNS接通记录，非正式比较。
- `neural_run01/`：最初小模型三步训练，保留原始输出及`KNOWN_ISSUES.json`；恢复核验临时第四步曾进入搜索，不用作三步检查点回放证据。
- `neural_run02/`：纠正为重新装载三步检查点后的完整小模型集成；基础/改进模型及ALNS/VNS四臂独立重算通过。
- `replay_run01/`：从`neural_run02`保存的权重重新回放，神经路线净效益与原记录相同；仍只使用第0折训练实例。
- `paired_cli_run01/`与`paired_cli_replay01/`：四变体共享教师/遮蔽的本地小模型入口检查，训练与六臂验证回放均完成，不是效果实验。
- `remote/dev01/`：完整骨干CPU三步训练完成，但ALNS漏打包导致退出1；原日志、13份SHA核验报告与远程权重身份保留。
- `remote/dev02/`：RTX5090完整骨干三步训练及四臂检查退出0；15份报告SHA核验，本地四份路线再次独立重算见`local_verification.json`。基础216402.068 U、组合216063.008 U、ALNS217363.902 U、VNS215369.684 U，仅训练r101集成记录。
- `remote/dev03/`：第0折面板在自动调优0配置下首步GPU编译失败退出1，六份报告SHA核验，未完成训练步。原始错误保留，不视为训练完成。
- `remote/dev04/`：相同38原训练例×6品质条件、四变体各240步、C2四例六臂验证已退出0；只恢复默认自动调优4重试。63份报告SHA、24份路线本地独立复算通过，8份NPZ留服务器。D比基础版均值高2798.17 U，仍低于ALNS1332.76 U；组合比基础版低546.71 U。详见[首轮结果](../../../docs/static_delivery_research_20261021/5090首轮算法结果.md)与`local_panel_verification.json`。本轮目标10s，收尾审计导致实测略超时，不称正式60s/严格10s比较。

教师标签都是在允许的原训练客户实例上由共用初解与完整目标VNS在线生成的路线，不生成新客户，不以测试路线训练。三步小模型结果没有显示改进版优于基础神经适配，不能作性能结论；完整方法效果需要按冻结协议训练/验证与正式实验。
