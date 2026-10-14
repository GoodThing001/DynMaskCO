# B 论文静态工件发布候选（2026-10-09，本地，未公开）

此候选支持封存文件身份与已有 JSON 算术核查。它不支持模型推理、正式在线重放、从零训练或历史完整闭包复现。当前论文正文仍是[唯一可编辑工作稿](../../文档/论文与投稿/论文工作稿_收口修订_2026-10-09.md)，此处不复制第二份论文正文。

## 已生成与验收

- [candidate_manifest.json](candidate_manifest.json)：25 份小型原封存快照，共 289,470 字节（5 个入口、6 个 pin 模块、14 份主表/C1 JSON），原包 183 文件基线与 16 项原资产只读引用。没有复制大文件。
- [dependency_inventory.json](dependency_inventory.json)：33 文件 AST 读取，6 个 pin、5 个与 s42 训练身份 SHA 匹配的依赖、5 个封存入口快照、17 个仅当前可获得而无匹配历史 SHA 证据的源码。没有导入项目模块。
- [validation_record.json](validation_record.json)：构建时只读 SHA/AST、四组闭环逐实例均值、原 C1 判定核查全通过；原包 183 文件全部未改动。服务数与主表单元格另见[本轮完整核查](../../核查工件/论文封存与收口检查_2026-10-09.json)。归档 CI 被读取核对，未再次生成 bootstrap。
- [output_guard_validation.json](output_guard_validation.json)：真实 CLI 尝试写原包、写上游、覆盖已有候选均被拒绝，三个退出均为 1；候选 manifest 未变。构建工具另有七个路径/覆盖反例检查全部拒绝。

## 安全的只读校验命令

```powershell
python -X utf8 "D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/03_负结果线/发布候选/2026-10-09/verify_static_artifacts.py"
```

构建时该入口已以子进程执行并退出 0；它只读候选与本地原资产，不写结果文件，不联网、不加载模型、不调用仿真。原[构建工具](../../发布工具/build_static_candidate_20261009.py)已经生成本目录，再次构建必然拒绝，不能覆盖已有候选。

## 未支持的历史复现范围

固定状态六模块 pin 与闭环两物理模块 pin 的完整其他依赖身份仍未齐备。AST 清单中的当前源码只供发现依赖，不复制为旧历史实现，不以 syntax/工件 PASS 宣称旧模型评估可复现。`mpre_trained.py` 当前 SHA 不匹配原 s42 训练身份（原版本已丢失）；包中没有原 `repro_check.txt`、旧回退计数日志或原闭环逐事件 JSON。动态 sys.path、上游包导入和外部依赖版本须另行校验，静态列表不能取代实际来源核验。

这里的封存入口快照仅作身份/语法材料，不应直接执行。没有收录或调用旧 `run_repro.sh`，没有在当前运行树替换模块，没有重跑科学结果。公共上传、DOI、模型冒烟和投稿均不在本候选执行范围。
