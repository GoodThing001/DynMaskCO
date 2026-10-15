# DynMaskCO-CC D3-L1 P0-C V1.8 执行审计

2026-10-15；代码与证据以本文件所在提交为准。仅申请 F-free 采集工程签署，不申请科学比较、训练或在线增益授权。

## 实际修改与影响

| 文件（相对扩展目录） | 实际变化 |
|---|---|
| `scripts/simulation/recourse_snapshot.py` | 新增默认关闭的标量原类型/原字节捕获扩展及恢复；旧默认字段语义保持；新 SHA，不冒用旧身份 |
| `scripts/simulation/d3_l1_state_collector.py` | 传递显式标量保真选项，默认仍关闭；保持真实 reveal 的 WAL→capture 顺序 |
| `scripts/simulation/d3_l1_collection_runner.py` | 启用保真、完整 parity/recapture、原件持久化、逐槽 manifest、durable 完成标记和真实失败分账；正式模式硬阻断；入口/新身份登记 |
| `scripts/simulation/d3_l1_p0c_v18.py`（新增） | 强制身份闭包、独立监督、单调计时、Windows Job/POSIX process group、失败记账与不可重采、并发锁、写盘故障闭环 |
| `scripts/tests/d3_l1_v18_{parity_probe,engineering_acceptance,kill_acceptance,synthetic_worker,readonly_audit,regression,seal_evidence}.py`（新增） | 固定工程矩阵、不同实现的独立精确复核及封存工具 |

另仅追加 V1.8 勘误/本审计/原始证据及两份权威状态顶部导航。未混入 `CC_Compare/MAPT_code` 的原有脏子模块；未修改 upstream、冻结数据、合同、冷链物理、比较器、正常事件循环、权重或旧冻结树。

`v18_evidence/SOURCE_CLOSURE_V18.zip` 保存实际验收的源码原字节（14 个运行模块及 V1.8 工具），SHA256：

`f83e862f0cb0431c9287c41e070005b14285a79dbbacbefb0225b5649171d352`

它是新版本快照，不修改旧冻结树。Git 的自动换行可能使 fresh checkout 源码文件 SHA 改变；复放时应先在**新建隔离 checkout**核验并恢复 ZIP 中的原字节，不能规范化 SHA 来通过。证据目录 `.gitattributes` 只对 V1.8 原件禁用换行转换，不影响历史工件。

## 身份原件

运行环境：本地 Windows、Python 3.12.7（Anaconda）、NumPy 1.26.4、JAX/JAXlib 0.5.0、Flax 0.10.4。未使用 5090/3080 运行。来源绝对路径、包版本及 frozen checks 原件见 `l1_gates/A1_D3_L1_P0C_RUNTIME_IDENTITY_V18.json`。

| 对象 | SHA256 |
|---|---|
| dev_check dataset | `66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89` |
| M0 teacher manifest | `b84e171676f8afc02e5ac416538965dda224671d719863a1412f659464bcaf04` |
| profile 文件 | `63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8` |
| profile 内部 hash | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |
| base contract | `32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0` |
| effective contract | `edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833` |
| 历史冻结身份文件 | `f44e3456db5fac67c85829e5f28648174b4c50b81d1af85a288dc78a7795172c` |
| action_contract | `9d64273aa3df98cb931c0c8c4b0d0112c134c13e080d3b4f1615b8ae06d2b53b` |
| coldchain_contract | `0fb9e6e8fcff11c538e5ee41d6b0f4fb5c7db64fb5d528ceac5b46a130a268d2` |
| coldchain_state | `a2ee98dcb7290f527e5f93ce4f21d094a1a706bc662ca1a897d234708e56e43e` |
| cvrptw_utils | `1fd6aa427bd242227963972c9a439dd94c5b6ca3cb5dbcef37bda3f2f315a73f` |
| d3_l1_bitwise_parity | `eafb83ab602640d8ce9cc17ebbda26d9aadc1f0e9aeb829eba5e225081ee1228` |
| d3_l1_causal_perturbation | `d11e3a08f7d494595dfef97aa8ef134bfacf23aafd0f1ef09f10ff6caf606d5a` |
| d3_l1_collection_runner | `b09a0b7dd976e5eef6af29097c4e8b741bbc467769a21c593721cc35dde79ce9` |
| d3_l1_p0c_v18 | `7ce307af786879466e965f283e57cbd7bb34dffa4c96bcab01cfdf7c8f89e85b` |
| d3_l1_state_collector | `fff9158855ca75c1469025a99432ca9c6eb6369147f5ec42aeb5bf1cf96230c4` |
| dynmaskco_cc_context | `4eebb2a51a024a9e881ce7100bbe568219e1afa5ac8c6d33c707924a36ff1685` |
| jf1h_repair | `233f8439536a2af676fe389b012b74c4fa8cf458fde7aef226b084342ca089ee` |
| joint_fleet | `7dafcd6d82bc7cc602d5353b8fd3afdd790c1fec6b8aefd449988652be707935` |
| recourse_snapshot | `a2c13f75566e383312b369fd8cf9453a41a22c7eeeb8f4cd38021c8baa965537` |
| strict_online_env | `fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6` |

旧 recourse SHA 为 `0af3535acaf6a84956104c0f1c8788c04e02499a087b6faa3d264239a19f27f8`；旧 collector SHA 为 `849667e4d572e8250041913eee8d59ce9d376b8119f777179ab5a99f9b71e86f`。旧清单仍作为历史记录，其全表不被改写；只对新 V1.8 使用上述新身份。

## 可复核结论与证据入口

- 首差异与原始完整对象：`v18_evidence/before/{PARITY.json,TYPED_OBJECTS.pkl}`；修后对应 `after/`。
- 发布版完整资格：`l1_gates/A1_D3_L1_P0C_ENGINEERING_QUALIFICATION_V18.json`，8 个门全 True。
- 真 worker 矩阵 44/44：`v18_evidence/acceptance_release/ACCEPTANCE.json`。负例第一槽分别失败，另外三槽不因首槽失败而筛除；正控四槽 PUBLISHED。独立原件审计 8 份、675/675。
- 强杀 8 场景、72/72：`v18_evidence/kill_release/`；每个 `WORKER_DAY00.json` 含 PID、exit code、真实 Job active count、kill/reap 时间；每场景新 run_id 无新 FIXTURE_STARTED。
- 跨版本共享回归 38/38：`REGRESSION.json`。没有调用包含 VAL 数据的旧 snapshot 测试；只执行其四个合成函数。
- 旧 accept 13/13 与闭包 14/14：`legacy_accept_release/`、`runtime_closure_release/`。旧 N11 是检测单元测试，真实强杀结论由新的独立场景提供。
- 类型、dtype、正负零、NaN payload、车辆归属、恢复异常、write failure 的 JSONL 均按实际失败保留；`staged/` 和没有发布完成回执的对象不是有效快照。
- 全部原始文件逐字节 SHA/大小索引：`v18_evidence/ARTIFACT_SHA256_V18.json`。包含此前失败及 superseded candidate，不仅收录发布版 PASS。

独立进程使用与生产比较器不同的严格类型/字节编码；它证明保存工件与登记身份一致、完整恢复轨迹精确一致。它没有在别的机器上重跑仿真，不能宣称服务器或科学复现通过。

## Windows PowerShell 复核（从扩展目录执行）

以下只读复核不执行 rollout；输出必须是不存在的新文件。

```powershell
cd D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
$prereg = '../research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
python scripts/tests/d3_l1_v18_readonly_audit.py --root "$prereg/v18_evidence/acceptance_release/worker_none" --runtime-identity "$prereg/l1_gates/A1_D3_L1_P0C_RUNTIME_IDENTITY_V18.json" --out "$env:TEMP/D3L1_V18_readonly_new.json"
```

独立工程重放入口（新 scratch，不覆盖封存目录；仅开发日 0 的固定矩阵，绝非正式采集）：

```powershell
python scripts/tests/d3_l1_v18_engineering_acceptance.py --outdir "$env:TEMP/D3L1_V18_engineering_new"
python scripts/tests/d3_l1_v18_kill_acceptance.py --outdir "$env:TEMP/D3L1_V18_kill_new"
python scripts/tests/d3_l1_v18_regression.py --out "$env:TEMP/D3L1_V18_regression_new.json"
python scripts/simulation/d3_l1_collection_runner.py --role accept --outdir "$env:TEMP/D3L1_V18_legacy_new"
python scripts/simulation/d3_l1_collection_runner.py --role closure --runtime-closure --outdir "$env:TEMP/D3L1_V18_closure_new"
```

## Linux 复核命令（资格尚未实测）

在新建隔离 checkout 中按已验证 ZIP 恢复源码原字节；已冻结的 NPZ/profile 应只读提供，不能生成替代数据或修改原件。仅只读复核不需要加载数据。工程矩阵需要同 SHA 的 dev_check、manifest、profile；它会为该平台产生**新的工程身份**，不能把旧 Windows identity 的平台/包字段改写成匹配。

```bash
cd /path/to/new-isolated-checkout/C-VRP_Cold-chainVehicleRoutingProblem
prereg=../research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1
python scripts/tests/d3_l1_v18_readonly_audit.py --root "$prereg/v18_evidence/acceptance_release/worker_none" --runtime-identity "$prereg/l1_gates/A1_D3_L1_P0C_RUNTIME_IDENTITY_V18.json" --out /tmp/D3L1_V18_readonly_new.json
python scripts/tests/d3_l1_v18_engineering_acceptance.py --outdir /tmp/D3L1_V18_engineering_new
python scripts/tests/d3_l1_v18_kill_acceptance.py --outdir /tmp/D3L1_V18_kill_new
```

运行路径 `argv`、run_id、源 SHA、PID 和输出 JSON 原件均已封存；命令中的 `new` 路径若存在应换一个新目录，不能删除旧证据后重放。

## 剩余边界与签署申请

本地工程阻断已关闭，**可以申请用户签署一次固定 16 日×4 槽 F-free 采集**。尚无用户签署，正式模式仍拒绝执行。`SIGNATURE=PENDING_USER`，`FORMAL_COLLECTION_AUTHORIZATION=NOT_AUTHORIZED`。若用户选择 Linux 服务器执行，必须先完成该环境身份和强杀资格，当前不申请服务器执行放行。

本轮仅正常提交/推送 `codex/o0cc-cal-phys`；`main` 保持基准，不擅自同步其他分支。提交前再次 fetch/检查远端，如存在并行推进则先保留并处理，不 force push。最终具体提交及远端 HEAD 以交付消息和 Git 历史为准。

隔日 18:00（Asia/Shanghai）GitHub 同步已登记为当前聊天定时任务 `dynmaskco-github`：只同步本聊天可归属且验证完成的改动，不启动科学实验，不提交其他写者内容；没有变化保持安静。
