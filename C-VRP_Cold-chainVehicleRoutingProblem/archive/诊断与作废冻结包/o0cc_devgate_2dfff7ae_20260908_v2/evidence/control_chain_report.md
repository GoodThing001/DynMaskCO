# 控制链验收报告（第 4 步，不跑 oracle rollout）

> 生成时间：2026-09-08 · 结果：**控制链 PASS（13/13）**

## 结论

完整控制链「冻结计划 → 续跑复用 → 九份 canonical 汇总 → 正式分组聚合」端到端跑通，13 项负例在 runner 启动前 / 正式聚合前被拒绝。

## 身份

| 项 | hash |
|---|---|
| compute（15 文件） | `2dfff7ae8948ce6eea2bb7b2b60fdae6924ca294652fffff66a7dee0dca7b888` |
| runner self-hash | `b74ea7cb5c5b2b7b5497aa16a752cf688a41219ab2edce3703643d7c3181b322` |
| control | `49019c4adf0cd71c80a81c81897fd7479c2c18e8bdf767904de90865073b848e` |
| analysis | `cd266de12c267aa44ef1bae9a72d5a1b4dfda3a13b44e326ae02a9e5c339ca30` |
| 测试脚本 `test_control_chain.py` | `92bb573b07bd5bac96307131d7e5cef626edc28748435b586a08531ff469f355` |

> compute/runner/analysis 未变；control 相对上一轮有变化（本版移除了 `--runner` 旁路 + 加了 9-cell 精确集合/profile 自洽前置校验）。

## 验证项（13/13 PASS）

| 测试 | 预期/结果 |
|---|---|
| 主路径 | exit=0；runner 未调用；9 份 canonical marker 均 COMPLETE 且绑定 compute/run_id；聚合 total=1152、n_seed_groups=128、bootstrap={10000,42} |
| 纯续跑 | 同目录二次执行 exit=0，不调 runner |
| 非空无 manifest | 拒绝 |
| config（baseline）漂移 | 拒绝 |
| null 实例 | 拒绝 |
| compute hash 漂移 | 拒绝 |
| NPZ 内容漂移 | 拒绝 |
| profile 内容漂移 | 拒绝 |
| cell/seed 漂移 | 拒绝 |
| bootstrap n_boot 漂移 | 拒绝 |
| 多余实例（inst_128） | canonical 精确实例集拒绝 |
| marker 产物 hash 篡改 | 聚合器拒绝 |
| marker run_id 替换 | 跨批混入拒绝 |

## 冻结的正式聚合口径

正式模式额外校验 `instance_set==[0..127]`、`config`（capacity/num_vehicles/slack/objective/local/baseline/n_boot/boot_seed）、CLI 参数与冻结口径一致，以及每 cell canonical marker 绑定 `compute_sha256`+`run_id`（九 cell 同 run_id）。

## 下一停点

**控制链 PASS 达成；仍未启动正式 DEV-GATE。** 后续：独立源码归档（已完成）→ 归档路径执行验证 → 服务器实读核验 9 NPZ/profile/运行路径 → 独立回归目录小规模接线 → 最后才批准空目录正式跑 9×128。
