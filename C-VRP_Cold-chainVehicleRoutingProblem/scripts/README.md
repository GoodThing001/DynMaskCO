# 运行代码分类

| 位置 | 用途 |
|---|---|
| [static_profit](static_profit/README.md) | 当前静态多车冷链净效益：任务/特征、神经骨干、搜索、教师、训练与验证 |
| [static_delivery/run_rrp.py](static_delivery/run_rrp.py) | 旧RRP v3任务，保留原33托盘、目标和检查点身份 |
| [tests](tests) | 对应任务的语义、搜索、恢复和工程检查 |
| [project_paths.py](project_paths.py) | 统一工作区、扩展和只读上游的路径与导入契约 |
| simulation及tests中非static历史文件 | 原位保护的历史材料，不是当前算法模块 |

使用python -B；不新增扩展__init__.py。当前运行请先看[训练策略](static_profit/README.md)，科研冻结与测试门槛见[版本索引](../docs/文档索引与版本状态.md)。
