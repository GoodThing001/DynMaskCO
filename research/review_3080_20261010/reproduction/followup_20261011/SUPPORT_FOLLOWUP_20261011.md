# 支持门跟进：数值维度、当前文档与可执行闭包（2026-10-11）

本目录只追加独立检查、窄补丁候选和工程接口。没有改当前入口、旧183包、旧handovers或实验结果，也没有训练或启动数值实验。

## 当前入口的数值维度没有差异

`ROOT_CURRENT_ENTRY_AUDIT_v2.json` 与上一份 `LATEST_SUPPORT_V3_FINAL_AUDIT.json` 的四维度计数相同：numeric 1、eligibility 5、statistical 5、scope 5；不是工具目录变化或 source manifest 想当然绑定。29项headline零失败与四维度是不同范围。

唯一部分数值资格是 pickup_impossibility_proven。0.997%与tail census工件对账通过；23.4229km、2213.5795kWh、2.2939品质组成的单实例9.48%距离占比和0.095%乘积只来自旧文档内部算术，没有原实例工件。算术能重算，不能授予原始实例对账，更不能推广为总效用上界。`NUMERIC_DIMENSION_DIFFERENCE_CHECK.json` 保存逐字段相同的具体原因。保留此非零结果，不应为追求全PASS改变维度。

本轮通过 Python subprocess 核真实当前CLI返回码2；archive verifier返回0。shell外层显示1并不改变实际Python退出2。`CURRENT_DOCUMENT_CLI_RECORD.json` 与 `ARCHIVE_CLI_RECORD.json` 保存命令和真实退出。原183包、14 JSON镜像、315恢复文件、52论文单元和9谨慎语句本轮再验通过；历史完整复現仍False。

## 当前新文档词境窄修正

当前实际入口源SHA为 `98dc3791336dc6873903f6a6a05874efdeb3b7dba88e75575001eb6fe7985d41`。传入新20261010三线修正文档与B草案后出现四个词法误报：唯一日、32唯一行、等效检验需求表、没有实现等效检验。它们不是新的正向/等价断言。

`CURRENT_CONTEXT_NARROW.patch` 仅基于上述真实源：current-doc要求“唯一”同时带穷尽路径语义；需求表另列非断言词境；否定词补“没有实现/尚未实现”。历史检测类别、非零裁决和科学False保留。候选副本 `program_support_current_context_candidate.py` 供独立运行；不自动安装。两份当前新文档经候选扫描均0当前断言、共3非断言词境保留。没有命中只是词境结果，不是科学pass；历史越界仍使CLI为2。`NARROW_CONTEXT_CANDIDATE_AUDIT.json` 保存结果。

## 工程闭包接口，不再只核presence

`support_replay_bridge.py` 要求调用者给外部可信登记、project、bundle和独占新output。bundle不能自选审计代码；只允许现有 `delivery_trace_contract.py` 或 `pickup_trace_runner.py` 两类既有注册验收器。先核protocol、runner/training source、权重本体、dataset、contract、环境文件的实际SHA；核记录的start/end ledger与相同身份；核精确day×arm原始覆盖、拒绝重复raw；然后实际执行验收器，而不是相信其报告true。验收结果与bundle逐字段复算对账，布尔/NaN/缺漏均拒绝。实际加载的项目.py须在审计器源闭包中，执行后再核源与输入SHA。

配送事件adapter把raw订单、坐标与contract逐字匹配冻结dataset/contract，检查释放/货物ownership、装货、车队状态、承诺段、TW、容量、返仓和距离；不能把raw自行修改后换hash当合法合同。它没有品质/制冷能耗认证。pickup adapter调用原observer的typed trace/evaluator replay，当前接口要求每条journal覆盖一个本地日；不是全新独立物理实现，也没有新全日actor因果及总预算证书。扩大到whole-cohort schema需要另加明确unit映射，不能用文件名或人工true强行过门。

`registered_artifact_byte_identity_pass` 与 `runtime_ledger_consistency_pass` 是工程一致性字段；存储的ledger不证明服务日志真实，weights bytes不证明激活，training source bytes不证明训练运行。`independent_semantic_implementation_established`、original_training_execution_reproduced、full_from_zero_reproduction、physical_calibration、formal_online、stable核心在线始终False。接口刻意不能补造旧原源码、s43/s44血缘与完整事件缺口；详见 `HISTORICAL_CLOSURE_GAPS.json`。

### 可运行的工程命令

```
python -B -m unittest -v test_followup_interfaces
python -B support_replay_bridge.py --project <新冻结工程project> --bundle <bundle.json> --trusted-registry <外部登记registry.json> --output <project外的新report.json>
```

`SYNTHETIC_BRIDGE_DEMO` 展示schema，训练/权重等角色是显式合成占位，不能用于真实研究资格。演示真实执行现有事件validator并重算距离，CLI0仅表示合成工程验收，stable仍False。26新测试通过：9词境+17source/真实事件重放与攻击；包括hash已更新但提前到达/修改contract/修改坐标/漏返仓，确认实际语义验证而非只看hash。

下一步对真正新运行，从开始封存实际source、训练、激活、环境、每个完整原始事件及不可改的start/end记录，使用独立已审来源的物理/因果/预算evaluator重放；然后才做PROGRAM注册的强对照与归因确认。旧数据缺失时报告unknown，不以新重放实现、合成接口测试或摘要归档冒充历史完整复現。
