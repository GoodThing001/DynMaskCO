# 三线审计修正工程组件

> **2026-10-10**：新增literature_quality.py与商品参数JSON；封闭形式、变温积分、温域/外推、货物计时和电输入单位11项测试在本地/服务器通过。历史合同保持。新组件独立使用，不声称已集成完整强动态优化。[来源和验收](../../docs/当前规划/决策与审计/三线修正_服务器复验与文献参数版本_2026-10-10.md)。

这是本地工程验收版本，不是正式科学运行入口。旧P/D协议、权重、冻结树及原负结果包不改。

- `pickup_trace_runner.run_observed`：复用当前SAA/StrictOnlineEnv与原轨迹评价器，调用方显式传入数据、采样器和合同。保存所有决定、私有状态、完整终局物理轨迹和逐日结果；观察I/O另计，软提交截止保持。数据含私有真值，仅供本地审计，不回传actor。
- `pickup_trace_runner.replay`：检查hash链/完整日及实际源码身份，按明确允许的dataclass类型重建轨迹，用原评价器逐字段复算。不会重新训练或加载历史权重。工程合成案例用空未来采样器，不是I3模型收益实验。
- `delivery_trace_contract.validate_trace`：认证动态订单装货、库存所有权、有限车辆、前缀承诺、容量与服务时窗；支持固定分舱或显式共享总容量。此新工程合同允许货物预先备好，实际装货同时满足release与ready；不追改旧D2语义。品质、能耗、仓库温控及强滚动优化器未在此实现，不据此授予动态冷链在线资格。
- `qualification.assess`：按明确证据逐项检查，缺件关闭，技术资格始终与执行授权/在线收益分开。
- `audit_io`：无损类型化JSON、逐条fsync、顺序hash链、不可覆盖新日志及截断/重复日检查。

验证命令：

```powershell
python -m pytest -q "D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/correction_engineering/test_corrections.py"
```

工程产物见[验收记录](../../results/three_line_corrections_20261009/ENGINEERING_ACCEPTANCE.json)；完整审计处置见[72项登记](../../results/three_line_corrections_20261009/ALL_CORRECTION_DISPOSITIONS.json)。每个真实研究新版本仍需模型/数据/合同/池/预算完整身份及新预声明，不能直接把该组件挂入旧正式链重跑。
