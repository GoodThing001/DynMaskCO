# H-GEOM-01 第一阶段准备：BLOCKED

本目录是2026-10-20负责人在当前聊天明确“确认”后批准的第一阶段准备包。T1已单独提交；本轮未获正式19单元评价授权，正式机会消费为0。

最终资格见 [qualification_report.md](qualification_report.md)。29/29只是组件回归回执，不能替代独立资格门。`run_identity.json`明确为 **DIAGNOSTIC_ONLY_NOT_QUALIFIED_T2**，不是合格T2或正式评价许可。

## 文件

- `registration.json`：T1七要素、授权范围、原19单元/123身份元数据、30项事前验收清单；未保存或恢复正式路线。
- `source_bridge.py`：从冻结源码原样编译白名单AST定义，隔离JAX/模型/训练和源日加载；没有修改原文件。
- `recovery.py`、`geometry.py`：闭包、完整结构恢复与确定性距离的可调用组件，只在纯合成夹具上运行。
- `windows_resources.py`、`supervisor.py`：Windows原生进程组与工作集硬限的隔离候选和合成故障探针；尚不是合格的两阶段正式执行器。
- `journal.py`：持久化19槽位、校验事件、原子记录与只读故障解释。
- `synthetic_suite.py`：纯构造数据的回归；不读取正式phases/npz/checkpoint，禁止实际正式入口。
- `independent_audit.py`：只读标准库复核，不导入被测实现、不跑新探针或正式数据。
- `freeze_t2.py`：仅保存BLOCKED后的诊断字节身份，不生成合格T2。

合成记录保留于 `synthetic_artifacts/`；初次原生入口缺陷及第一轮报告路径缺陷保留，未覆盖或改写。缩小内存/时间/CAP仅用于事前规定的合成故障注入，不是正式合同改值。

内存映射探针产生的零字节填充测试文件不是研究数据，不作为公开证据提交；其生成方式、字节数及原生结果在源码和JSON中可核对。最终修正版使用自动回收的临时映射文件。

本轮因资源资格阻断停止，不自动重跑、不申请正式评价、不扩张研究包，也不修改R0.1/补充合同、状态入口、冻结证据或上游。
