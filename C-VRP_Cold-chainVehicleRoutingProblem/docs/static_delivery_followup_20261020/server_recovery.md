# RRP服务器缺项查找证据（2026-10-20）

> 文档身份（2026-10-20同步）：只读服务器搜寻快照，不是当前执行状态。没有找到独立补全原件的结论保留；后来允许的D1-n9全零派生基于完整保存钟的全部等式，不是凭全零外观猜值，原TXT仍只有8值。run06/run07为后续同源代码功能验收，不能算新找回的资料来源；当前见[总索引](../文档索引与版本状态.md)。

在两台已授权服务器的用户项目范围内，未找到独立的历史RRP实现，也未找到可补全`sol-D1-n9-np32.txt`缺项的独立参考副本。此结论仅覆盖下列搜索范围；不表示其他位置绝无材料。未修改服务器文件、未启动训练、未删除文件。

## 查找范围与结果

| 服务器 | 根目录 | 遍历条目/目录 | RRP相关命中 |
|---|---|---|---|
| Linux | `/home/hzeng/project/MASKCO-Main` | 22614/2571 | 仅本次`research/static_rrp_20261020`目录 |
| Windows 3080 | `C:\Users\zh\project\MASKCO-Main` | 4547/572 | 0 |

先用SFTP列出配置所指项目根，再作只读、有界文件名遍历。每台设深度12、18万条目、22秒上限；两台均自然完成，未触及时间/条目上限、深度剪枝均为0、遍历错误均为0。匹配文件/目录名中的`rrp`、`refrigerated`、`energies.?13.?6214`、`sol-D1-n9-np32`，并收集相关目录内的JSON/TXT/C++/MiniZinc/Python/Markdown/ZIP条目。没有扫描其他用户目录，没有读出凭据。

排除`.git`、缓存、依赖目录、checkpoints、private及名称含private/credential/secret的路径；Linux不跟随符号链接，Windows拒绝reparse/junction。本次Linux上传的run01–run05从发现遍历排除。未展开压缩包，且无相关命名线索的实现文件可能不被发现。

## 已知上传副本的SHA对照

额外仅对run01–run05的命名D1-n9文件只读计算SHA，五份均为498字节，且均与本地原TXT一致：

`d9d5892773d4901f25b0862393e427b6f3d459042072bfb356d5a770bbc47bb6`

服务器完整路径逐份保存在[机器记录](server_recovery.json)的`known_uploaded_copy_checks`。这些是本次从本地上传的同源副本，不能作为独立恢复来源；没有把其数量当作正确性证据。未下载任何恢复文件。

## 缺项维持原状

本地文件为`data/public_sources_20261020/refrigerated_routing_rrp/Solutions/D1/sol-D1-n9-np32.txt`，作者固定commit为`f5c5317a5ecd8fe64df779e79bd0397904bc0f9e`，当前SHA与原下载清单匹配。原TXT的Route包含9个值，Waiting times仅8个值。服务器搜索未恢复第9个等待值，不能据全零外观推定缺项也是0；保持原字节与不完整标记，不据此声称全部参考成本已复现。

原始查找证据：[Linux](server_evidence/linux_discovery.json)、[Windows](server_evidence/windows_discovery.json)。机器报告保存搜索参数、逐份副本SHA、局限及证据文件SHA；报告仅记录证据所需项目路径，不输出连接主机、认证字段或配置内容。
