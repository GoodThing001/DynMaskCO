# 配套工具分类

工具分为现行运行/冻结入口、研究资料工具和维护工具。资料工具已从本目录根部归入evidence；现行算法、服务器和冻结入口保留原路径。

| 位置 | 用途 |
|---|---|
| [evidence/acquire](evidence/acquire/README.md) | 8个公开论文、数据与源码获取工具；不会自动重跑下载 |
| [evidence/analyze](evidence/analyze/README.md) | 3个资料转录、证据汇总及早期数值情景分析工具 |
| [evidence/verify](evidence/verify/README.md) | 4个论文来源、单位与历史结果复核工具 |
| [maintenance](maintenance/README.md) | 本次目录整理、备份和核验 |
| [static_profit_remote.py](static_profit_remote.py) | 当前净效益服务器开发任务辅助；独立输出、资源限制和角色门槛 |
| [seal_static_profit_baseline.py](seal_static_profit_baseline.py) | dev04基线留档辅助，保留现有入口 |
| [static_profit_windows.py](static_profit_windows.py) | 当前Windows运行辅助 |
| [verify_static_profit_panel.py](verify_static_profit_panel.py) | 当前开发面板本地复核 |
| [verify_static_profit_training_study.py](verify_static_profit_training_study.py) | 当前训练开发对照复核 |
| [preflight_publication_model_v1.py](preflight_publication_model_v1.py) | 已冻结协议预检工具，保持原路径和字节 |
| [audit_and_seal_publication_protocol_v1.py](audit_and_seal_publication_protocol_v1.py) | 已冻结审查/封存工具，保持原路径和字节 |
| [static_rrp_remote.py](static_rrp_remote.py) | 原RRP服务器辅助，也供现行连接辅助使用，保留稳定入口 |
| [static_rrp_cleanup.py](static_rrp_cleanup.py) | 旧迁移恢复工具；仅按其历史范围使用，不对当前开发重新执行旧清理 |

例如，资料核验命令现在为：

```powershell
python -B tools/evidence/verify/verify_publication_model_evidence_20261021.py
```

资料工具重新执行可能写回对应阶段的报告；本次未重跑历史实验，最终原报告字节保持原样；导入核验的恢复事项见整理记录。原路径、新路径与原件备份见[整理记录](../results/workspace_organization_20261021/README.md)。
