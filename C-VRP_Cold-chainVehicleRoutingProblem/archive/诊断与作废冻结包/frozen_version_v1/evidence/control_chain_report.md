# 控制链验收报告（第 4 步，不跑 oracle rollout）

> 生成时间：2026-09-08 · 结果：**控制链 PASS**

## 结论

完整控制链「冻结计划 → 续跑复用 → 九份 canonical 汇总 → 正式分组聚合」端到端跑通，且关键负例在 runner 启动前 / 正式聚合前被拒绝。

## 身份

| 项 | hash |
|---|---|
| compute（15 文件） | `2dfff7ae8948ce6eea2bb7b2b60fdae6924ca294652fffff66a7dee0dca7b888` |
| runner self-hash | `b74ea7cb5c5b2b7b…`（compute_files['run_action_oracle.py']） |
| control | `a2c911d1a15fa7b8e73f48991d6ea14ef512a94a4a2671ce1dc4b59ac53d2a18` |
| analysis | `cd266de12c267aa44ef1bae9a72d5a1b4dfda3a13b44e326ae02a9e5c339ca30` |
| 测试脚本 `test_control_chain.py` | `0b597ac54c9a2c49a74b11e468e9246dcd2c1ad1ce745e2c9940f7f6523df0d2` |

> 说明：compute 未变（本轮只改 control/analysis 层）；control/analysis 相对上轮审计有变化（本轮修了正式聚合 config/bootstrap 校验、canonical compute_sha256/run_id 绑定、driver docstring）。

## 验证项（5/5 PASS）

| 测试 | 结果 | 关键断言 |
|---|---|---|
| 主路径 | PASS | driver exit=0；runner 未调用；9 份 canonical marker 均 COMPLETE 且绑定 compute/run_id；正式聚合 total_instances=1152、n_seed_groups=128、bootstrap={n_boot:10000, seed:42} |
| 纯续跑 | PASS | 同目录二次执行 exit=0，结果一致（不调 runner） |
| 非空无 manifest | PASS | 拒绝（exit=1，防接管孤立旧实例） |
| config 漂移 | PASS | 改 pre_run_manifest config.baseline → resume 拒绝（exit=1） |
| null 实例 | PASS | 某实例写成 `null` → 扫描阶段拒绝（exit=1，不覆盖） |

## 冻结的正式聚合口径（本轮新增）

正式模式（`--mode dev_gate`）现在额外校验：
- `frozen.instance_set == [0..127]`
- `frozen.config`：capacity=50 / num_vehicles=25 / slack_vehicles=1 / objective=coldchain / local=true / baseline=JF1-H-F / n_boot=10000 / boot_seed=42
- CLI 参数与冻结口径一致（`--objective` / `--local` / `--n-boot` / `--seed`）
- 每 cell canonical marker 绑定 `compute_sha256` + `run_id`（九 cell 必须同 run_id == 冻结 run_id），防跨 run 混入

## 下一步停点

**控制链 PASS 达成；仍不启动正式 DEV-GATE。** 后续：独立源码归档（保持目录结构、相对路径键）→ 归档路径执行验证 → 服务器实读核验 9 NPZ/profile/运行路径 → 独立回归目录小规模接线检查 → 最后才批准空目录正式跑 9×128。
