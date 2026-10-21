# 工作区维护

[organize_workspace_20261021.py](organize_workspace_20261021.py)是本次一次性整理的准备、迁移和核验工具。prepare与apply拒绝覆盖已有计划或重复应用。

在扩展根目录复核：

```powershell
python -B tools/maintenance/organize_workspace_20261021.py verify
```

它检查原件备份、17份迁移结果、72项科研封存、两个历史恢复ZIP、19项原位保护记录、未触碰的算法文件以及导航链接。不会运行训练或历史报告生成，也不会提交或推送。

[整理报告](../../results/workspace_organization_20261021/README.md) · [原件恢复说明](../../archive/workspace_organization_20261021/README.md)


[reorganize_cc_compare_20261021.py](reorganize_cc_compare_20261021.py)记录7项对比方法分类与完整旧树归档。核验命令为`python -B tools/maintenance/reorganize_cc_compare_20261021.py verify --full-archive`；prepare/apply是一次性操作，不覆盖旧输出。[整理报告](../../results/cc_compare_organization_20261021/README.md)。


[workspace_release_20261021.py](workspace_release_20261021.py)是用户后续授权的双服务器同步清单工具；[workspace_remote_apply_20261021.py](workspace_remote_apply_20261021.py)在服务器执行受限归档与安装；[workspace_git_release_20261021.py](workspace_git_release_20261021.py)按显式增删清单发布。连接只读取原SFTP配置，配置内容不传输、不输出。回执见[本轮同步报告](../../results/workspace_sync_20261021/README.md)。

早前整理核验中的“算法未改动”仅对应当时快照。后续算法开发可改变未冻结的算法文件；原整理清单不改写，本轮改用当前源码SHA和测试重新验收，科研72件冻结约束继续保持。
