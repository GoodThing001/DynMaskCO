# 研究资料工具

| 子目录 | 工具职责 |
|---|---|
| [acquire](acquire/README.md) | 公开论文、数据和官方源码获取 |
| [analyze](analyze/README.md) | 表格转录、文献归纳、早期情景汇总 |
| [verify](verify/README.md) | 原件SHA、单位、历史路由和文档链接核验 |

工具按用途归类，文件名中的日期保留研究阶段身份。它们输出仍在原docs/data/results路径，以保持原数据和科研记录可追溯。各工具通过scripts/project_paths.py获得工程根目录，同目录获取工具之间保留直接导入。

从扩展根目录使用python -B运行新路径。历史运行记录里的旧命令不改写；路径对照见[迁移清单](../../results/workspace_organization_20261021/migration_manifest.json)。[返回全部工具](../README.md)。
