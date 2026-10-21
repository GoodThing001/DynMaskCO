# CC_Compare旧树与恢复说明

用户于2026-10-21确认保留7项方法并要求按职责分类。[legacy/CC_Compare](legacy/CC_Compare)是旧目录整体重命名后的原树，共29,506件、约7.75GiB，原字节不改。

它保留原方法、动态适配、失败/NO-GO及限定结果、原文档和MAPT_code完整内嵌Git仓库（含未跟踪工作文件）；不依赖仅包含已跟踪文件的Git归档。navigation_before保存原导航；navigation_concurrent_before保留整理期间研究进展文档的并行更新，未覆盖其最新内容。

[本地迁移与SHA清单](../../results/cc_compare_organization_20261021/README.md)记录原路径、现路径和内嵌Git状态。恢复时使用新的独立工作区，按原相对路径CC_Compare放置完整旧树，避免覆盖当前目录。历史运行可能仍依赖当时其他工作区或服务器资产，不据此声称完全复现。

根.git、private、凭据和其他保护范围未移动；MAPT内嵌仓库随被移出的旧方法整体保全。恢复树仅本地保存，不自动提交或推送。[当前CC_Compare](../../../CC_Compare/README.md)。
