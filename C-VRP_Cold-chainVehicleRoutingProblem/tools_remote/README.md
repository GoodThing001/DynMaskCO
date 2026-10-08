# 服务器辅助脚本

这里集中存放原先散落在扩展目录根部的 `tools_remote/tools_remote_*.sh` 和配套的 `tools_remote/tools_remote_rr3_run_frozen.py`。脚本仍以服务器上的扩展目录为工作目录，内部调用路径已按此位置更新。

从扩展目录根部运行，例如：

```bash
bash tools_remote/tools_remote_rr3_readiness.sh
bash tools_remote/tools_remote_run_rr3_frozen.sh
```

`tools_remote/tools_remote_rerun_rr3.sh` 是旧版、从可变源码目录运行的诊断脚本；正式 rr3 请遵守当前进度文档中的冻结与验收要求。这里只做文件归档和路径迁移，没有启动服务器实验，也没有改变实验结果。
