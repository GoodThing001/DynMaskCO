# 交付线工程诊断交接（run_r3）：CPU-only 8+2 秒诊断

- **日期**：2026-10-14
- **授权**：用户在「新独占登记中先做一次 CPU-only、无发布的完整 8+2 秒工程诊断，无需再问我」；本次三跑（run_r1/run_r2/run_r3）全部属于这一次授权。
- **诊断结论**：**run_r3 `gate_pass = true`（10/10 门）**，`publication_performed = false`。
- **诊断不构成**：可发布请求的资格、掩码-重构-冷链可行性、任何在线增益证据。
- **`stable_online_gain` 仍为 `False`**。

## 1. 三次调用的原始记录（全部保留，未改写）

| 运行 | 执行器 | 到达阶段 | 结果 | 超时 | 退出码 | 回收 | 墙钟 |
|---|---|---|---|---|---|---|---|
| run_r1 | `server_diagnostic.py` | `captured_P0` | FAIL | 否 | 2 | 是 | 36.999 s |
| run_r2 | `server_diagnostic.py`（**误用**：监督器把子进程名写死成 r1 执行器） | `captured_P0` | FAIL（无效运行） | 否 | 2 | 是 | 30.060 s |
| run_r3 | `server_diagnostic_r2.py` | `complete` | **gate_pass=true** | 否 | 0 | 是 | 115.672 s |

两次 FAIL 的原因都是**我自己的工程缺陷**，不是资源、科学或协议条件：

1. **r1**：闭包候选 r20 的 `_real_file(module.__cached__)` 假设 `__file__/__cached__` 一定是路径；真实 maskco 加载栈里有模块通过模块级 `__getattr__` 伪造属性，于是 `Path()` 抛
   `TypeError: expected str, bytes or os.PathLike object, not _OpNamespace`。→ 候选 r21：类型收窄；并把**每个**已加载模块名都记进映射（文件或 `None`），因此 r20 完全看不见的「无文件模块」现在可见。
2. **r2**：`supervise_diagnostic.py` 把子进程名写死 → 修正后的执行器根本没被启动。→ 监督器 r2：`DIAG_CHILD` 必填无默认、`DIAG_EXPECT_MANIFEST` 必填，**启动前**逐条校验 sha256，不符即以退出码 3 拒绝启动并落盘记录。r3 启动前 9/9 清单条目全部匹配。

同一缺陷输入已在本地复现并验证：`LOCAL_UNIFIED_PROTOCOL_v2_RESULT.json` **9 pass / 0 fail**（含 r1 崩溃输入：`hasattr` 为真而捕获不崩；无文件模块在计时区内出现 → `module_added` 命中、`new_files=0`，正是 r20 会漏掉的情形）。

## 2. run_r3 实测（原样抄录自 `run_r3/DIAG_STATE.json`）

| 项 | 值 |
|---|---|
| CPU 环境锁 | 亲和性 `[15]`、`JAX_PLATFORMS=cpu`、`CUDA_VISIBLE_DEVICES=''`、四个 `*_NUM_THREADS=1`，均在导入 JAX 之前设置 → `enforced=true` |
| 可用 RAM | **180.178 GiB**（门槛 8 GiB） |
| 本人遗留进程 | **0 个**，`owned_leftovers_checked=true`；排除 3 个自身祖先进程（pid 543895/543896/543898，逐条记录） |
| 目标核负载 | core 15 = **0.0%**（门槛 <25%） |
| JAX 设备 | `["TFRT_CPU_0"]`，`backend=cpu`；CUDA 插件在发现阶段报 1 条非致命 ERROR（`CUDA_ERROR_NO_DEVICE`），全程未使用 GPU |
| 模型加载 | `CVRPModel` 16.028 s，logits 形状 `[1,51,51]`、全部有限 |
| 声明工作单元（一次 encode+decode） | 首次 **0.806 s**，稳态中位数 **0.667 s**，最小 0.541 s |
| 8 秒计时段 | 8.729 s 内完成 **13** 个单元 → 单推理仅占预算 **8.3%** |
| 2 秒回收预留 | 实测 **0.307 s**；RSS 前后 1274.52 MiB（释放 0.0 MiB，属观测，不是门槛） |
| 闭包 P1→P2 | **qualified=true**：0 变更 / 0 新增 / 0 缺失 / 模块映射未变 / native 映射未变 |
| 闭包 P2→P3 | **qualified=true**：同上 |
| P0→P1（信息性，不设门） | 3585 个新增文件（热身阶段本就为此存在） |

## 3. 本次诊断新发现的**阻塞性**工程事实

**启动级闭包捕获无法塞进 8 秒请求预算。**

| 捕获点 | 耗时 | 覆盖文件数 | 模块数 |
|---|---|---|---|
| P0（热身之前） | 1.217 s | 168 | 112 |
| P1（热身之后） | **25.627 s** | 3753 | 2007 |
| P2（计时段之后） | **22.766 s** | 3753 | 2007 |
| P3（发布之前） | **26.740 s** | 3753 | 2007 |

- 已登记的请求式闭包协议要求 P1/P2/P3 三点捕获；三点合计 **75.133 s**，单点最贵 26.74 s = **请求预算的 3.34 倍**。
- 对比：单次模型推理 0.667 s；整棵登记*源码树*扫描测得 5.618649 s / 8903 文件。
- 因此**绑定约束不是推理，而是闭包验证本身**：捕获 3753 个文件的内容 sha256（含 jaxlib/numpy/scipy 等被映射的原生二进制，数百 MB）是主要开销。

结论：`delivery/actual_loaded_closure_request_root_v1` 里「在请求内完成三点捕获」的设计**不可行**，该登记**仍不放行**——但阻塞理由已从「两点协议未统一」变成这个可量化的预算事实。

（可行方向，尚未执行、尚未登记：把请求内闭包面缩小为「显式 shard 内容哈希 + 实际映射的原生路径集合 + 全模块名→路径映射 + 其余文件 `(路径, 大小, mtime_ns)` 身份」，把内容哈希留给请求外的发布前全量检查。这需要一条**新的登记**，并且必须先用等时测量证明它能进 8 秒。）

## 4. 3080 源码 shard 身份

- 登记 shard 集共 **46** 个工件，全部位于 `research/` 之下；**在 3080 上 46/46 缺失**（`run_r3/SHARD_PROBE.json`）。它是**宿主机本地的研究工件集**，所以 3080 侧请求无法对着它做身份校验，除非把 shard 集上传。
- 请求实际加载的**计算源**在 3080 上全部存在，且与本地逐字节一致（8/8，含 305,061,150 字节的 `cvrp100.ckpt`）：
  `mpre.py`、`cvrptw_utils.py`、`project_paths.py`、`coldchain_contract.py`、`coldchain_state.py`、`coldchain_evaluator_a1.py`、`rolling_optimizer.py`、`ckpts/cvrp100.ckpt`。
- 即：**本地↔服务器计算源身份已闭合**；**登记工件 shard 身份对 3080 不适用**，这一点按事实记录，不再含糊。

## 5. 门状态（本轮后）

| # | 线 | 门 | 状态 |
|---|---|---|---|
| 1 | 取货 | 登记规则＝实际动作（1000 条决策日志独立复算） | **通过**（用户已独立复核） |
| 2 | 取货 | 当前源码冻结为候选身份 + 在线管线部署资格（五日规则/同期对照精确复现） | **通过（工程，仅限已被消耗的五日）**；对新动态日的漂移仍不可推断 |
| 3 | 取货 | 独立强动态接/拒对照 | **未通过（尚未尝试）** |
| 4 | 取货 | 学习归因（三种对照部署动作确有差异；等时 mask–重构–冷链可行性修复） | **未通过（尚未尝试）** |
| 5 | 交付 | 统一 P1→P2 准入规则 | **通过**（本地 9/9 + 服务器实测 qualified） |
| 6 | 交付 | `owned_leftovers_checked` 得到已验证取值 | **通过**（3080 实测 0 个，附 3 条排除记录） |
| 7 | 交付 | 3080 登记 shard 身份 | **未通过**：46/46 缺失（计算子集已逐字节匹配） |
| 8 | 交付 | 请求内实际加载闭包验收 | **未通过**：单点捕获 22.8–26.7 s，无法进 8 秒预算 |
| 9 | 交付 | 可发布完整请求登记 | **未通过**（按用户要求暂不放行；且需先解决第 8 项） |
| 10 | 两线 | 稳定核心在线增益 | **`False`** |

## 6. 待用户决定

1. 交付线：是否按第 3 节的「缩小请求内闭包面 + 请求外全量校验」方向**新开一条登记**（先做等时可行性测量，再谈可发布请求）？
2. 取货线：是否现在启动第 3/4 项（独立强动态接/拒对照 + 学习归因、等时迭代）——这是「两线工程放行」里唯一还没动过的部分。
3. 是否需要把本轮证据推到 GitHub（当前按用户要求**未推送**）。

## 7. 工件索引

- 原始逐请求记录：`run_r1/`、`run_r2/`、`run_r3/`（`DIAG_STATE.json`、`DIAG_FAILURE.json`、`DIAG_FAILURE.traceback.txt`、`SUPERVISOR_RECORD.json`、`SUPERVISOR_STDOUT.txt`、`SHARD_PROBE.json`）
- 登记：`REGISTRATION.json`（r1）、`REGISTRATION_r2.json`（r2）、`REGISTRATION_r3.json`（r3）
- 身份清单：`EXPECTED_IDENTITY_r3.json`（r3 启动前 9/9 匹配）
- 协议：`unified_closure_protocol.py`（r1）、`unified_closure_protocol_r2.py`（r3 使用）
- 本地测试：`LOCAL_UNIFIED_PROTOCOL_RESULT.json`（6/0）、`LOCAL_UNIFIED_PROTOCOL_v2_RESULT.json`（9/0）
- 交接汇编：`DIAGNOSTIC_HAND_BACK.json`（含全部原件的 sha256 清单）
- 复用登记引用：`REGISTERED_SHARDS_REFERENCE.json`（46 条登记 shard 的副本，仅作对照）
