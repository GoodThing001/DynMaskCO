# Workspace Assets

本目录存放不参与源码执行的工作区外围文件。

```text
workspace_assets/
├── downloads/   # 原始下载压缩包；仅归档，不参与训练
└── tools/       # 本机开发辅助工具；不属于 MaskCO/DynMaskCO 算法代码
```

- `downloads/` 中的压缩包尚未删除，也未判定为冗余备份。
- `tools/cleanup_cursor.bat` 会操作用户的 Cursor 缓存，只应在明确需要且关闭 Cursor 后手动运行。

