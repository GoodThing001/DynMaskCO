# H-GEOM-01 第一阶段执行前资格报告：BLOCKED

**2026-10-20；第一阶段准备已停止；正式评价机会消费0次；合格T2未建立。**

## 1. 授权与身份

用户在当前聊天对“T1单独冻结提交 → 隔离实现 → 纯合成回归 → 独立核验 → T2冻结，不包含正式19单元评价”明确回复“确认”。基准 `198ab92a8082338eec821abf92f7d6fcbffe8a6e` 与两个远程分支一致，原R0.1及补充合同保持冻结。该确认未授权正式数据评价。

T1先于任何实现和合成探针单独提交：`3cc964a989dc5b7fd9599868086179f10ac036af`。`registration.json` SHA256为 `ab9db91dbdb7cb872bcda0aaf1dac4bd0403409bdea2949e4db4be8435ddd2f0`。同次提交仅增加本目录的字节保全规则`.gitattributes`，未修改旧文件。

## 2. 已完成及证据边界

新增组件仅位于本research目录：冻结源码AST桥接、模型无关闭包枚举与结构恢复、固定顺序math.fsum几何、持久化账本、Windows原生资源候选、纯合成回归、标准库只读独立复核。桥接原样执行冻结定义，未导入JAX/Flax/Torch或任何模型，也未修改原route_check、动作契约或共享合法关系。

最终组件回执 `synthetic_artifacts/regression_r2/regression.json`：**29/29通过**。其中含独立实现的合成闭包交叉检查、空后缀/closed、跨车分配、全锚点/exact-once、同哈希不同坐标、强制短哈希碰撞、找到目标但未穷尽、CAP边界、完全/部分闭包缺目标区别、零分母/非有限值、1%边界、互斥数值关系、固定fsum、19/123整包门、原子文件残缺、进程异常与强杀、监督器死亡、阻塞调用超时、内存拒绝、额外进程拒绝和正式入口拒绝。第30项“source_and_result_independent_recheck”由单独只读复核回答，**不是自动计为通过**。

合成故障预算采用96 MiB及2秒等缩小值以触发边界；正式6 GiB/1800秒/120秒/5000 CAP未更改、未据合成通过而宣称正式全链路合格。没有对正式19单元创建视图、枚举、恢复或计算距离。原phases仅按字节核对T1身份；未进行源日反序列化或重跑。

## 3. 决定性阻断：原生内存峰值超过配置上限

只读复核直接读取 `synthetic_artifacts/regression_r2/memory/result.json`，没有重跑探针：

| 量 | 实测 |
|---|---:|
| Job配置上限 | 100,663,296 bytes（96 MiB） |
| Job读回的原生提交峰值 | **100,843,520 bytes** |
| 超出量 | **180,224 bytes** |
| 轮询到的进程树私有内存峰值 | 96,575,488 bytes |
| 轮询到的RSS和峰值 | 74,977,280 bytes |
| 用例终态 | MEMORY_LIMIT_EXCEEDED；worker已退出 |

资源候选已设置包含监督器及唯一worker的原生Job总提交限、最多两个活动进程、非继承kill-on-close句柄，以及双方HARDWS_MAX（各占总RSS上限一半）。96 MiB文件映射探针RSS和74,797,056 bytes、私有内存和25,165,824 bytes；两种口径已分开记录。额外第三进程被拒，监督器强杀后worker未存活，19槽位保留。

这些正向控制**不抹去**内存反例。原生Job峰值读回超过其配置天花板，轮询没有捕获该峰值。正确识别并记录一次故障，不等于证明能够防止所有越限。原生Job限值设置成功、读回一致、RSS测试通过，均不足以据此宣布“整个进程树严格满足内存合同”。本轮不推断Windows所有可能方案均不可行，也不虚构该超限的操作系统根因。

按负责人授权中的资源失败止步规则：**停止资格推进，BLOCKED**。不扩大预算、不用epsilon容忍180,224 bytes、不追加故障轮次寻求更有利峰值、不改变6 GiB口径。

## 4. 独立复核与未完成项

`independent_audit.py`只使用标准库读取源码和既有JSON，不导入被测实现、不运行新探针。回执 `independent_audit.json` 为 **12/15检查成立，资格BLOCKED**；三项未满足：

1. `strict_native_memory_ceiling`：上述原生峰值反例，严格硬限未建立。
2. `integrated_two_phase_runner_qualified`：因资源门触发止步，候选仍为组件与原生合成探针，未完成受监督的“全单元恢复 → 全局门 → 距离”流水线、两阶段累计时钟和全部运行内持久化的整体验收。
3. `new_source_bound_before_regression`：最终组件回执记录了原冻结定义身份，但没有在回归前绑定全部新增模块摘要；不得事后加摘要追认既有回归为合格T2验收。后续诊断冻结仅固定现有字节。

后两项是止步后如实保留的未完成资格，不把它们包装成新的科学失败。29/29组件检查不能替代这三项，不能在后续引用中省略BLOCKED。

## 5. 开发失败账与诊断冻结

初次原生正常路径 `synthetic_artifacts/native_normal/` 在启动worker前因当前CPython未导出`subprocess.CREATE_SUSPENDED`发生AttributeError，START/19槽位留存，无成功终态；修正为Windows原生CREATE_SUSPENDED值0x4，并扩大入口异常落账覆盖。后继`native_normal_v2`正常完成。它们均为合成准备，不消费正式机会。

第一轮整组回执`regression_r1`为22/29：七项失败来自证据路径相对/绝对混用造成报告格式异常；修正输出路径后，`regression_r2`为29/29。原记录保留，没有覆盖。该数字只证明最终组件清单，不继承独立资源资格。早期失败实现未作事前源码快照，因此不为其声称完整代码版本复现身份。

`run_identity.json`保存本次实际源码、原输入/合同、合成JSON/JSONL、环境及独立复核的诊断身份，字段明确为`DIAGNOSTIC_ONLY_NOT_QUALIFIED_T2`、`T2_qualified=false`、`formal_evaluation_authorized=false`。不补签、不追认T2，也不把此次冻结称为READY_FOR_FORMAL_EVALUATION。

## 6. 交付与停止

只交付本research目录的新增源码、T1、合成文本证据、诊断身份及本报告；不提交大体积零填充映射临时材料。现有未跟踪目录、子模块、原数据/历史结果均未清理或修改。R0.1、补充合同、权威状态文件、原引擎/认证器、MASKCO_code/保持不变。

结论：**BLOCKED，不是H-GEOM科学假设NO-GO。** 未进行模型/checkpoint/GPU/训练、正式19单元质量评价、源日重跑、L2/L3或最终测试。按合同停止本次执行准备，不自动追加运行、调整CAP/资源或扩展研究包。正式唯一一次评价仍未获授权、未消费。
