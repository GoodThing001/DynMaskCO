# D3-G0 Evidence B3-AT2H — 结果报告（只读写入点审计）

- 预登记：`B3AT2H_COUNTEREXAMPLE_SPEC.json`；设计：`B3AT2H_OWNERSHIP_DESIGN.md`
- 机械清单：`B3AT2H_REAL_ENV_WRITE_INVENTORY.json`；原始输出：`raw/B3AT2H_stdout.txt`
- 命令：`python -B scripts/tests/test_d3_b3at2h_write_inventory.py`
- 本批**未修改任何源文件**（`source_modified: false`），未接入容器/守卫，未改历史基线或 `MASKCO_code/`

## 1. 机械清单结果

| 指标 | 实测 |
|---|---|
| `strict_online_env.py` 写入点总数 | **115** |
| `DECISION_WRITE`（需认证） | **28** |
| `EXECUTION_ADVANCE`（执行推进） | **54** |
| `UNCLASSIFIED` | **33**（已逐组裁定，见设计 §2.3） |
| `failure_semantics = UNKNOWN` | **33**（= 上述构造期/局部量，均为非运行时状态写入） |
| 裁定要求的"必须出现在真实路径上"的目标字段缺失 | **无**（`mutable_suffix`/`committed_next`/`status`/`ready_time`/`current_node`/`current_load`/`served_route`/`coldchain_state`/`coldchain_updated_time` 全部找到） |
| 函数×目标 组合待裁定数 | 219（"该函数合法地不写该字段"必须逐条裁定，不得静默省略） |
| 扫描的 D3 模块 | 5 |

分类键是**所属函数**（预登记 R1），因为同一个 `status` 既被 `_commit`（决策）也被 `_advance_fleet`（执行）写入；仅按字段分类必然出错。

## 2. 关键发现（本批的实际产出）

1. **两套权限确实必须分开**：`DECISION_WRITE 28` 与 `EXECUTION_ADVANCE 54` 在**同一批字段**上重叠（`status`/`ready_time`/`committed_next` 等）。若不区分，要么把合法的"沿已承诺路线继续行驶 / 服务完成 / 时钟推进"全部锁死，要么让真正的提交绕过认证。
2. **基类 `Replanner.plan` / `on_reveal` 不写状态**（抽象方法，写入点计数为 0）：真正的决策写入来自**子类**——`GreedyReplanner.plan` 在本文件，另 7 个 replanner 站点在 `evaluation/` 与 `research_20261010/`（B3-AC 已清点，全部为 `INDIRECT_WRITER_CAN_WRITE_D3_STATE`）。**接口移交必须覆盖子类**，否则出现第二套状态。
3. **reveal 清尾不在 `on_reveal` 里，而在 `_run_loop`**：`v.mutable_suffix = []` 属执行推进（新信息使旧尾部失效），但其后果等价于一次决策撤销。设计已给出裁定建议（归执行推进 + 加强不变量 1 + 登记 reveal 事件），并明确该清空**不得**计入"未经认证的新客户写入"。
4. **`_force_return_all` 无直接写入**（委派给 `_commit` 与 `_advance_fleet`）：其权限类由被委派函数决定，必须记入清单而不是忽略。
5. **33 个 UNCLASSIFIED 全部为构造期/局部量**：27 个在 `__init__`（配置与只读引用数据、钩子、计数器初值——**不写车辆状态**，且发生在任何执行实例存在之前）、2 个在 `get_reserved_customers`（新建局部集合）、3 个在 `_greedy_suffix`（局部 `route`/`unvisited`）、1 个模块级 `sys.path`。**无一属于需要认证的第三类**。

## 3. 三个问题的回答（详见设计）

- **Q1 权威状态所有者**：每个 D3 执行实例**只有一份**权威车辆状态，由 `D3StateContainer` 持有；env 用**执行推进接口**、协调器用**决策提交接口**操作同一份状态；**禁止** env 一份 + container 一份长期并行。接入方式为可关闭的 `state_authority` 形参，缺省 `None` 时行为与现行完全一致。
- **Q2 哪些写入需要认证**：见上表与设计 §2.1/§2.2，含每行的**写入对象 / 触发条件 / 权限类别 / 是否需重新认证 / 失败语义**，以及执行推进必须满足的**五条不变量**（不改未执行后缀客户集合、服务恰好一次成对、时钟单调、执行推进不得引入未认证新客户、冷链只经 `transition_segment`）。
- **Q3 协议一致性**：预声明回归规格 = 相同初始状态/揭示序列/合法决策输入 + 六项比较（服务顺序与时刻、承诺冻结、恰好一次、C0 能耗口径、冷链与品质记录、拒绝/WAIT/超时）；并**分设两类比较标签**：`COMPATIBILITY`（差异=回归）与 `SAFETY_FIX`（差异=预期，不得记为回归失败）。

## 4. AT2B 门槛与拆分建议

门槛表见设计 §4：**env 所有权移交设计**、**决策/执行权限分类** 两项由本批建立；**真实执行路径无旁路的调用图证明**未建立；**D8 仍阻断**；**正式预算安装资格未建立**。

因此 AT2B 必须拆为 **AT2B-1 可关闭默认不启用的生产接线（含调用图证明）→ AT2B-2 独立隔离验收（用 §3 规格）→ AT2B-3 才考虑启用真实写回**，每步单独预声明、单独裁决。

## 5. 非声称

- 不声称 P2 / G-L / G-C；不声称唯一写入者在生产路径成立；不声称 D8 已解除；不声称正式预算下的安装资格。
- 威胁模型**不**覆盖任意反射与恶意内存修改（见设计 §0）。
- **本批只是设计**：完成设计**不**自动启动 AT2B（用户停止条件）。
- **研究方向**：本批及 B3-Q→AT2G 的安全工程**不构成** MaskCO 算法创新；所有产物记录 `algorithmic_gain_claim: none`。P2/G-L/G-C 未建立、D8 阻断、D3 G0 未通过、G1/G2 未启动、MaskCO 可归因 F−D 增益未建立、I3 与旧 D2 `not_qualified`、`stable_online_gain=False` 全部不变。

## 6. 停止条件核对

用户停止条件："形成完整、可机械核验的真实 env 写入清单、权威状态所有权方案、权限分类和预声明回归规格。不得因为设计完成而自动启动 AT2B。"

| 要求 | 状态 |
|---|---|
| 完整、可机械核验的真实 env 写入清单 | ✅ 115 写入点，AST 派生，每条带 file:line + 源码锚点（脚本每次运行重读，代码漂移即失效） |
| 权威状态所有权方案 | ✅ 唯一权威副本 + 两个授权接口 + 禁止并行 live 状态 + 可关闭接入形参 |
| 权限分类 | ✅ `DECISION_WRITE 28` / `EXECUTION_ADVANCE 54` / 33 个 UNCLASSIFIED 逐组裁定为构造期或局部量 |
| 预声明回归规格 | ✅ 六项比较 + 两类比较标签分设（不得合并统计） |
| 不自动启动 AT2B | ✅ 本批未接线、未修改源文件；`does_not_authorise_AT2B: true` |
