# V1.10–V1.14：真实输入通过，F 前向整轮阻断，持久化修复仅合成验收通过

2026-10-15，接续已推送 `2bba9be`。目标仍是建立可归因、可复现的稳定正向在线增益。这里区分采集、输入、评分、认证、安装和在线效用的证据等级；不把工程通过写成科研增益。

## 当前裁决和完整分母

| 门 | 当前实测 | 可支持的结论 |
|---|---|---|
| V1.9 一次 F-free 采集 | 64/64 PUBLISHED，独立1498/1498 | 共同可恢复状态来源合格，一次token已消费 |
| V1.10 真实准备后 P0/特征 | 64/64，原/隐藏扰动逐位一致；独立1619/1619 | 真实可见输入管道工程通过，不是完整C0认证 |
| V1.11 真实F接口 | **22/64完成，42/64未完成；整轮BLOCKED** | 完成22条评分finite/shape/repeat exact通过；不能外推其余42条 |
| V1.12 首版阶段文件修复压力 | **8 PASS / 1 FAIL** | 留存失败，不能认定修复通过 |
| V1.13 有界读取修复压力 | **10 PASS / 1 FAIL** | 留存失败（压力夹具自身裸读新文件）；不能改成PASS |
| V1.14 修订压力夹具，实际生产读取接口 | **11 PASS / 0 FAIL** | 仅持久化原语资格通过；修订F完整流程没有执行 |
| 独立负结果与修复证据审计 | **707 PASS / 0 FAIL** | 确认V1.11失败、原件身份和分母，没有新F调用 |

V1.10：64个状态按原序恢复两独立分支，各只调用一次真实 `prepare_decision_point` 和 JF1-H-F.plan，共128次规划，零rollout/capture/F评分/安装。原/隐藏属性扰动下的P0、deferred、B池、固定工程掩码、legal原序和全部输入类型/位模式均相等。保存诊断投影与真实准备后方案尾段64/64一致；但这不补足完整认证。B池13–23单，legal6–26项，可见节点27–30，cargo无超8笔截断。全轮4.297秒，最大单状态0.781秒。工程掩码为B池前2单，仅用于输入资格，不是未来科学掩码规格。

## V1.11 实际失败及停止

使用已登记任务checkpoint和当前骨干，正确load_mpre_trained→partition→nnx.merge，seed42。83个训练叶子与任务pickle逐位相等，其他参数分区为空；c_adapter/delta_head各6个叶子，absmax分别0.7193627953529358 / 0.2422086000442505。CPU单进程，现有mtrained_score_fn固定pad Nv30/M26，不改模型、特征精度或比较规则。

事前固定：第1状态一次编译前向，然后64个输入每个两次相同前向，最多129调用。实际仅执行 **45调用 = 1编译 + 22×2**。前22条全部finite、shape与有序legal一致、重复类型/dtype/shape/原始bytes完全相等。没有挑选输入、改掩码或额外尝试。

接着在进入day5/position3的阶段记录时，`os.replace(STAGE.json.tmp, STAGE.json)` 抛 **PermissionError / WinError5**。worker退出1，Job内进程清理后active=0，批耗时19.562秒。42条按原序保留未完成，不能记作两方法一致或F优劣。原始报错、traceback、45次调用计数和22条两次完整numpy评分均在 `v111_evidence/`。

V1.11整体裁决：`REAL_INPUT_F_INTERFACE_BLOCKED`。worker在输出运行时origin尾报告之前失败，因此 **F worker完整origin起止闭包未完成**。独立进程在任何修复前复核545份事前源码字节全稳定，并按事前路径封入 `PREFAILURE_SOURCE_BYTES_V111.zip`（SHA `6110adc3b973744ee1188eedd24d60e7957089f4df3146e74ffb8e2c94ff1bcb`）；这不冒充缺失的worker origin尾报告。

## 持久化调查、最小修复与保留的失败

独立合成反例证实：Windows上保持目标文件普通读取句柄时，覆盖式os.replace会得到相同WinError5。原F监督器轮询读取正在被替换的STAGE.json存在该竞态；不是模型数值失败。

修复只作用于未来工程资格工具：阶段进展改为**不可变且独立编号文件**，fsync临时文件后发布到全新目的名；消费者只看已发布.json。Windows重命名句柄关闭期间短暂拒读，生产读取接口最多等待50ms并计入调用者总时钟；永久拒读和损坏的最终JSON硬失败，不以旧记录冒充新阶段。父进程在读取失败时终止Job进程树并记录原因。没有放宽任何parity、科学预算或模型判断。

V1.12的1000次子进程发布全部落盘，但读侧仍有4条拒读，8/1。V1.13新增有界生产读取后，1000次发布全部落盘，压力夹具仍从旁路直接open刚发布文件，出现1条拒读，10/1；原件保留。该轮`exhausted_permission_errors=1`来自注入永久拒读反例，说明上述旁路错误不来自生产latest()的重试耗尽（据源码和计数的判断）。V1.14把64个保持打开的观察句柄放在生产开始前的稳定记录上，持续用生产latest()读取新进度，并继续验证永久拒读/最终JSON损坏两种硬失败，11/11。1000个新记录、重启不覆盖、原序、无未发布半JSON泄漏均通过。

**显式勘误**：V1.12原件`scope`字符串曾无条件写“primitive qualified”，但同一原件`n_fail=1`，该字符串不能作为PASS裁决；V1.14已改为按零失败条件判断。V1.12/V1.13原件和当时源码PREFIX_*均未覆盖。两个失败属于合成压力夹具迭代，不消耗正式槽，不是F重试。

修复后F入口CLI主动HOLD，输出目录亦改为新的未使用路径；**本轮没有执行修复后的任何F前向**。其完整端到端资格仍未建立，不能把11项原语测试当成64输入F全通过。仅在后续执行范围另行冻结并符合用户授权后处理该门，不能补跑原批并擦去失败。

## 身份、文件与重放

任务checkpoint SHA `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5`；当前骨干 SHA `03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`。标签始终 **F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED**，不得改称历史训练身份已证。数据、profile、合同及14模块V1.9采集身份保持；完整值见[V1.9审计](A1_D3_L1_COLLECTION_V19_AUDIT.md)。

新增/修改工具均在扩展scripts/tests：`d3_l1_v110_input_qualification.py`、`d3_l1_v110_readonly_audit.py`、`d3_l1_v111_real_f_forward.py`、`d3_l1_v112_phase_journal.py`、`d3_l1_v112_phase_journal_acceptance.py`、`d3_l1_v114_failure_readonly_audit.py`。本阶段无生产模拟器/合同/比较器/基线改动，未改MASKCO_code或数据。未碰CC_Compare并行工作。

原始结果入口：`v110_evidence/INPUT_QUALIFICATION_V110.json`、`v110_evidence/INDEPENDENT_READONLY_AUDIT_V110.json`、`v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json`、`v111_evidence/WORKER_FAILURE.json`、`v114_evidence/PHASE_JOURNAL_ACCEPTANCE_V114.json`、`v114_evidence/INDEPENDENT_FAILURE_AUDIT_V114.json`。v110–v114各目录ARTIFACT_SHA256清单含全部成功/失败/故意损坏夹具原件；原始F工具字节和修订字节分别封ZIP。只读重放时以新输出位置避免覆盖归档；失败F不得自动重跑。

只读重放（不重跑F、不覆盖归档），Windows PowerShell从扩展目录：

```powershell
python scripts/tests/d3_l1_v110_readonly_audit.py --out "$env:TEMP/D3_INPUT_READONLY_REPLAY.json"
python scripts/tests/d3_l1_v114_failure_readonly_audit.py --out "$env:TEMP/D3_F_FAILURE_READONLY_REPLAY.json"
```

Linux从扩展目录（未实测，不声明跨平台合格）：

```bash
python scripts/tests/d3_l1_v110_readonly_audit.py --out /tmp/D3_INPUT_READONLY_REPLAY.json
python scripts/tests/d3_l1_v114_failure_readonly_audit.py --out /tmp/D3_F_FAILURE_READONLY_REPLAY.json
```

输出路径必须尚不存在；工具独占写。审计器后来仅新增`--out`重放输出参数，判定逻辑不变；产生原始1619/707报告时的审计器原字节另存 `AUDITOR_PRE_CLI_BYTES_V115.zip`，当前审计器会写自身新SHA。跨系统原件路径/源码字节若不匹配须如实失败，不能改输入。首次执行命令均为从扩展目录`python scripts/tests/<对应工具>.py`，前向失败原入口现已CLI HOLD，不能据此重复执行。

资源技能检测先遇到GBK输出和Windows磁盘探测错误，均如实留于RESOURCES_BEFORE_F.json；回退直接测本机4物理/8逻辑核、可用RAM约2.45GiB、D盘可用118.65GiB，nvidia-smi未在PATH；显式CPU、顺序单模型，未连接5090/3080。检测故障未触及模型或科学数据。

## 剩余门与执行资格

1. 经新范围冻结、完整修订监督器与独立尾闭包后，才可能关闭真实64输入F前向门；本轮不补跑。
2. 本阶段未登记可用的具体C0预算B及状态/合同绑定认证器；route certificate只覆盖TW/capacity/return/支持温区，不能冒称C0能耗/品质硬认证。不得设B=∞或猜测预算。
3. D8仍缺真实安装回执与状态/合同闭包，E2仍缺合格学习贡献控制。后续L1归因、L2新鲜独立状态质量、L3完整配对日效用与跨条件稳定性必须逐门，预算/阈值PENDING保持。

**stable_online_gain=False；algorithmic_gain_claim=none；SCIENCE_EXECUTION=BLOCKED；SIGNATURE=HOLD。** I3正式负结果−144.680 [−191.109,−100.125]、旧D2 not_qualified和最终种子隔离不变。没有F−D-Geo比较、训练、L2/L3、40日或最终测试；正式64槽不再采集。本轮停止在未通过的F工程资格门，已落实最小持久化修复并归档其有限验收，不以修复替代实际端到端复验。
