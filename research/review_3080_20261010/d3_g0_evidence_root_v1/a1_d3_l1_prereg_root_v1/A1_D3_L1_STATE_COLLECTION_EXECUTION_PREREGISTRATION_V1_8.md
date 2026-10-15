# V1.8：P0-C 精确恢复、强制身份与真实强杀工程勘误

日期：2026-10-15。基准：`4ca034dbe9827f750be57ddd22348ecc5e7ec08f`；实施前 fetch/ls-remote 确认 `main`、`codex/o0cc-cal-phys` 均在该提交。本文是追加工程勘误及资格记录；V1–V1.7、旧 FAIL、`V17_PATCH.py` 均保留，不追改历史结论。每轮资格矩阵在相应测试源码中固定后执行；不将事后文字冒称新的科学预登记。

**P0-C 工程资格：在封存的本地 Windows/Python 环境通过，可申请用户签署一次正式 64 槽采集。`SIGNATURE = PENDING_USER`，`FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED`，正式采集仍为 `0/64`。** Linux/5090/3080 的进程组和运行环境尚未实测，不具有服务器执行资格。跨平台支持代码不等于跨平台验收。

## 1. WP1：先实测首差异，再修复恢复编码

`v18_evidence/before/PARITY.json` 与 `TYPED_OBJECTS.pkl` 保存开发日 0 的四个固定揭示槽、完整对象及原始比较器输出。四槽一致：

| 阶段 | 首路径 | 原始 → 恢复 | 数值及 binary64 大端位模式 |
|---|---|---|---|
| capture → restore | `root[0][0].committed_arrive` | `numpy.float64` → `builtins.float` | 两侧 `0.21461221575737`；`3fcb7869c0000000` |
| 完整原始 → resumed 轨迹 | `root[0].services[0].arrival_time` | `numpy.float64` → `builtins.float` | 同上；NumPy `dtype=<f8`、shape `[]`、原始小端字节 `000000c06978cb3f` |

捕获未修改运行态；旧 snapshot 的重新捕获哈希相同仍不能证明精确类型恢复。`recourse_snapshot._t` 和多个显式 `float(...)` 丢掉原类型；`strict_online_env._advance_fleet` 将恢复后的 `committed_arrive` 原样写入 `ServiceRecord.arrival_time`。因此至少一部分终局类型差异来自恢复字段，而非新的数值计算误差。结论范围是本次四槽；不宣称任意未测状态的唯一根因均已穷尽。

最小共享补丁：`capture_recourse_snapshot(..., preserve_scalar_types=False)` 保持旧默认数组投影不变。V1.8 worker 显式启用 `lossless_scalar_v18`，在捕获时保存车辆、轨迹及服务记录的标量原类型、dtype 与原字节；恢复时按编码重建，之后继续**未修改的** `StrictOnlineEnv.run_resumed/_run_loop`。None、NaN payload、正负零和混合 NumPy/Python 类型均有回归。没有转换或规范化原始/恢复轨迹，没有修改 `bitwise_equal`；没有修改 C0 物理或正常 rollout。

这是新的快照实现身份：保留 `recourse_snapshot_v3_cc` 数组 schema，增加显式版本化扩展。共享 `recourse_snapshot.py` 与采集器 SHA 已改变，**旧身份不继续冒用**；新旧源码对照见审计。所有历史冻结树不变。旧 snapshot 无扩展时继续走原恢复行为，并不追认旧类型 FAIL 为 PASS。

`after/PARITY.json`：四槽 capture→restore、完整 traces、served mask 全部严格 `EQUAL`；四次同点重捕获哈希相同。普通 rollout 与修复前保存对象严格相等；剥离扩展后旧数组投影严格相等。

## 2. WP2：身份、WAL、持久化和一次 pass

- 运行前显式注册独立 V1.8 身份文件，supervisor 不会在启动时自冻结。启动前、每个 worker 前、worker 起止及 supervisor 结束强制核对数据、teacher manifest、profile 文件及内部 hash、base/effective contract、runner、参数、Python/包版本和真实模块闭包。旧 cal 字段不替代 dev_check SHA。
- 闭包 **14 模块**（旧 13 加 `d3_l1_p0c_v18`）：AST 覆盖 lazy imports，加实际加载的本地模块；解析 `__file__`、`__spec__.origin` 与绝对来源，逐文件 SHA 必须与登记相符。起止原件均保存，汇总逐模块起止 SHA。缺失、来源错、任一身份不等立即 `HARD_STOP_IDENTITY`，不启动后续 worker。
- reveal 真正到达时先写 `ATTEMPT_STARTED` 并 fsync，再 capture。辅助进度事件不覆盖生命周期状态。进行中终止为 `FAILED(terminated_mid_slot)`；未到达为 `NOT_ATTEMPTED_DUE_TO_TERMINATION`；强杀当日、已经启动过的日、已失败槽及损坏账本均不能新 run_id 重采。批次终止封闭后续日，分母始终 64。
- 快照先 `staged/`，完整验证后写 `snapshots/`，manifest 记 `VALIDATED_CANDIDATE`，账本 `PUBLISHED` 才有发布意图。`SLOT_FINISHED` 表明发布 fsync 已返回；缺此回执的发布行不会计为成功，杀死后追加失败保留原行。暂存及孤立文件一律不被独立审计接受。失败原件不删除。
- 每个 run 的监督身份、结果和每日结果同时保存在 `runs/<run_id>/`，根目录同名文件只是最新视图。OS 文件锁防止两个 supervisor 并发使用同一 pass。异常后不会无条件 `BATCH_COMPLETE`。磁盘故障返回非零，尽可能保存 `SUPERVISOR_PERSISTENCE_FAILED`，stderr 同时保留。
- 每个发布槽保留完整 snapshot、原始/恢复轨迹、同点重捕获、两次因果投影、SHA、state_hash 和逐阶段计时。原始与扰动投影各调用 JF1-H-F 一次，总数严格为 2；显式 restore、完整 resume 及重捕获次数分别列账，不隐瞒 resume 内部恢复。
- 独立只读进程另写严格类型/字节编码，不调用生产比较器来复核其结论；不运行新的 rollout。核对 manifest、快照字节、完整轨迹、合同、来源 SHA、durable 回执与投影证据。

## 3. WP3：真实进程树强杀及冻结计时

正式值不变：**rollout 120 秒**（START→FINISH，不约束结束后的验证存活）；**槽跨度 60 秒**（capture 边界至验证完成，含等待，监督器保守覆盖 WAL/持久化）；**全批 3600 秒**（监督开始至最终写盘，身份和初始化亦纳入）。计时用单调时钟，UTC 时间戳仅作审计。`slot_active_compute_s` 单列，不能代替跨度。

`d3_l1_v18_synthetic_worker.py` 不读 NPZ、不运行仿真或模型。生产 supervisor 的同一轮询/终止路径接受固定的合成进程命令。Windows 用 Job Object，并设置 kill-on-close；子进程等待父级完成 containment 的 GO 回执后才执行/派生。终止后真实查询 active process count 为 0、父进程已 reap。Linux 使用独立进程组和 `killpg`，但本轮只测 Windows，不外推。

八个固定分支：capture 前、capture 中、parity 中、write/publish 中、普通 worker_error（带活子进程）、batch 截止、延后 capture、PUBLISHED 可见但 fsync 回执未完成。测试参数为 0.4 秒/槽、0.6 秒/rollout、3 或 15 秒/批；正式默认值未改变。另测 rollout 已结束后 0.8 秒验证不会误触 0.6 秒日窗，以及最终写盘越过批预算会降级。stdout/stderr 使用文件，100 KB 双流不会阻塞等待。

## 4. 实测矩阵与保留的失败

| 资格项 | 发布版结果 | 原件 |
|---|---|---|
| capture/restore/recapture/full resume | 4/4 严格相等 | `v18_evidence/after/` |
| 真 worker 正反例、身份与状态机、时间作用域、写盘故障 | **44/44** | `acceptance_release/ACCEPTANCE.json` |
| 真 worker 正控 | 4 槽实际 PUBLISHED，逐槽两次投影、精确 parity 和序列化通过 | `acceptance_release/worker_none/` |
| 七类真 worker 负控 | 第一槽 FAILED，其余固定槽照常验证；不挑选失败槽、不改判据 | `worker_{type,dtype,signed_zero,nan_payload,vehicle_boundary,restore_error,write_error}/` |
| 独立只读工件复核 | **675/675**，8 份报告 | `acceptance_release/INDEPENDENT_WORKER_*.json` |
| 真强杀 | **8 场景 / 72 断言全通过** | `kill_release/KILL_E2E.json` + 各场景原始账本/进程结果 |
| C0 20 项 + v2 合成 4 项 + scalar 10 项 + 旧投影 4 项 | **38/38** | `REGRESSION.json` |
| 正确扩展目录执行旧 accept 入口 | **13/13**，属于开发真实 rollout + 反例，非全合成 | `legacy_accept_release/P0C_ACCEPTANCE.json` |
| 运行时闭包 | **14/14** 来源与逐文件 SHA 一致 | `runtime_closure_release/P0C_IDENTITY_CLOSURE.json` |

首轮 `kill_first/KILL_E2E.json` 的 **publish 分支 FAIL** 原样保留：辅助 WRITE 事件遮蔽 ATTEMPT，导致漏记 FAILED、下一日误杀。修复为按生命周期事件读取槽状态及按当日过滤 watchdog 后，新版本矩阵通过。另保留首次身份注册被 `cvrptw_utils` 缺导入路径拒绝的 `IDENTITY_REGISTRATION_FAILED.json`；修复只是添加既有 models 导入目录。`identity_candidate`、`worker_candidate`、`acceptance_first/final`、`kill_revised/final/durable` 均为前序工程源码身份，不冒充发布版。每次复验是具体源码修改后的固定全矩阵，不覆盖原 run，不用于科学择优。

V1.6 的 `_V16` 文件误引及 V1.7 accept 因路径错误未产出之事实仍按交接保留；本轮不追认那两次验收。当前新增带 `_V18` 的 gate 是实际 JSON 原件的逐字节副本，SHA 索引可交叉核对。

## 5. 执行资格与停止条件

只具备**申请签署**封存本地环境中一次 F-free 64 槽采集的工程条件。正式模式仍被代码硬阻断，需用户独立签署及后续显式授权接线，不把本文件、定时 Git 同步或本次资格运行当作授权。迁往 5090/3080 前须保留同一数据/合同/源码字节，在该环境重新登记版本并验收进程组，不能直接复用 Windows PASS。

D3 G0 未通过、D8 阻断、E2 缺失、旧 D2 `not_qualified`、I3 `ΔJ=-144.680`、95% CI `[-191.109,-100.125]`、`stable_online_gain=False` 与 `algorithmic_gain_claim=none` 全部不变。本轮没有正式 64 槽、F−D-Geo、训练、L2/L3、40 日批次或最终测试种子。

详见 [执行审计与复放命令](A1_D3_L1_P0C_V18_AUDIT.md)、[机器可读资格门](l1_gates/A1_D3_L1_P0C_ENGINEERING_QUALIFICATION_V18.json) 与 [逐文件证据 SHA](v18_evidence/ARTIFACT_SHA256_V18.json)。
