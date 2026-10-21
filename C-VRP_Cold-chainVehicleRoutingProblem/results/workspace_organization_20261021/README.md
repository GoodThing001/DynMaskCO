# 本地项目目录整理记录

日期：2026-10-21。范围是本地目录、工具路径和导航；不属于算法实验，不提交或推送。

本次迁移：

| 原位置 | 新位置 | 数量 |
|---|---|---:|
| tools/acquire_*.py | tools/evidence/acquire/ | 8 |
| tools/build_*.py、summarize_net_benefit_scenario_20261020.py | tools/evidence/analyze/ | 3 |
| tools中的4份资料核验脚本 | tools/evidence/verify/ | 4 |
| results/static_delivery中的2份品质中间JSON | results/static_delivery/history/quality_evidence_20261020/ | 2 |

可读导航更新为当前静态多车净效益研究入口，并按代码、工具、配置、数据、文档、结果和归档分类。移除一个经检查为空的旧ckpts目录；没有删除数据文件。

[逐文件迁移与SHA清单](migration_manifest.json)记录每份原件与新文件身份；[整理前备份说明](../../archive/workspace_organization_20261021/README.md)提供30份原件的可恢复副本。

最终核验：

- 17份迁移结果逐件核对；2份品质中间JSON原字节不变。
- 30份整理前原件已备份，ZIP成员及整体SHA通过核验；包括迁移文件、导航文档与忽略规则。
- 72项科研封存、两个旧恢复ZIP的整体SHA、原RRP 61件与原Solomon 56件通过核验；作者56件原实例在科研封存内核验。
- 19项原位保护记录的元数据、18份现行/旧RRP算法文件SHA保持一致。
- 15份迁移脚本语法通过；13份安全导入通过，2份带顶层下载工作流的旧脚本仅做静态检查。[安全检查回执](safe_tool_verification.json)。
- 当前算法入口的CLI帮助加载通过；导航链接检查数以[机器清单](migration_manifest.json)中的最终verification字段为准。
- 仅为新导航、迁移后的既有小型结果和整理清单增加忽略规则例外；ZIP备份、权重、private和凭据仍不进入版本化范围。

核验过程保留一次已恢复的问题：首次导入检查触发两个旧下载脚本的顶层流程，写回一份获取记录并新增一份记录。原获取记录已按整理前无修改的Git版本恢复为原16,284字节；新增记录移出资料目录。错误回执及被替换内容保存在本目录的import_probe_side_effects中，仅作维护追溯，不能当作原文献获取证据。最终检查改为禁止网络、子进程及文件写入，并跳过这两份顶层工作流脚本的导入。


现行算法、科研冻结工具、原始数据、已有运行目录、历史manifest、上游源码、private、凭据及用户原文件保持原位置。根目录research/results/tools的保留项有明确保护原因，不能为清空目录而移动。

后续复核命令（扩展根目录）：

```powershell
python -B tools/maintenance/organize_workspace_20261021.py verify
```
