# MaskCO 静态冷链配送工作区

文档版本：[全项目文档索引](C-VRP_Cold-chainVehicleRoutingProblem/docs/文档索引与版本状态.md)。当前使用文档与历史快照的范围以索引为准。

当前主线是公开RRP论文标准下的静态冷冻配送，以及同构多车方法扩展。订单全部预先已知，每车载货出发、逐客卸货。旧动态取货和负结果已转历史。

- [当前说明和单/多车命令](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery/README.md)
- [参数、时钟和成本公式](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery/参数与公式.md)
- [本轮论文审计、服务器补查和执行记录](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_followup_20261020/README.md)
- [历史迁移设计](C-VRP_Cold-chainVehicleRoutingProblem/docs/static_delivery_migration_20261020/README.md)
- [取货/负结果保全和恢复](C-VRP_Cold-chainVehicleRoutingProblem/archive/transition_20261020/README.md)

本轮本地68测试通过；实际三步小模型训练、检查点重载、单/多车三臂和30实例×两时域的60次Split检查完成。论文内部冲突已显式裁定，默认主标准30参考可行，但全部参考成本及作者完整算法复现仍未闭合，尚无方法收益或泛化结论。

v3完整骨干已在服务器run06完成3步训练、重载和单/多车验收；run07以原n34实例完成三个方法实际2车的补充检查。两个run均正常退出，全部输出SHA通过；历史run05仅属v2。`MASKCO_code/`和`CC_Compare/`第三方原库保持只读，新代码/结果在扩展目录。凭据留本地`.vscode/`，不进入运行包或文献资料；用户private和原未跟踪文件保持原地。
