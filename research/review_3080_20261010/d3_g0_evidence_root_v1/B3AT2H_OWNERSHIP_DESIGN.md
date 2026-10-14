# D3-G0 Evidence B3-AT2H — 真实 env 状态所有权移交设计 + 写入点审计（只读）

- 阶段：`D3_G0_Evidence_B3AT2H`（只读设计与审计；`B3-AT2B` 未授权）
- 预登记（先于实现冻结）：`B3AT2H_COUNTEREXAMPLE_SPEC.json`，sha256 见 `B3AT2H_BYTE_FREEZE.json`
- 基础提交：`74b58583a72fb5dcad969431dc1b3b22fa4debfb`
- 机械清单：`scripts/tests/test_d3_b3at2h_write_inventory.py` → `B3AT2H_REAL_ENV_WRITE_INVENTORY.json`
- 本批**只读**：未修改 `strict_online_env.py`、未接入容器/守卫、未改历史基线或 `MASKCO_code/`

## 0. 威胁模型（用户要求明确）

| 覆盖 | 不声称覆盖 |
|---|---|
| 项目正常执行路径 | 任意反射代码（`getattr`/`setattr`/直接访问单下划线属性） |
| 意外持有的可变引用（旧引用/复制后仍被使用） | 恶意内存修改（ctypes / gc 遍历） |
| 历史 replanner 直接写入 | |
| 已登记的动态调用（属性赋值、列表原地修改、元素替换） | |

**理由（照录用户裁定）**：`_vehicles`、`_authoritative()` 这类单下划线名称在 Python 中只是访问约定，不是语言级私有权限边界；因此**不得**仅凭"权威状态放在私有属性里"就宣称任意代码都无法越权修改。AT2G 的结论因此限定为"**在受控接口与已登记调用范围内**的所有权成立"。

## 1. Q1：谁拥有权威车辆状态

**方案（唯一权威副本）**：每个 D3 执行实例只存在**一份**权威车辆状态，由 `D3StateContainer` 持有；`StrictOnlineEnv` 与 `D3CommitCoordinator` 都不再各自持有长期并行的 live 副本，而是通过**两个不同的授权接口**操作同一份权威状态：

| 接口 | 持有者 | 可做的事 | 不可做的事 |
|---|---|---|---|
| **执行推进接口** `env.advance(*, authority)` | `StrictOnlineEnv` | 推进已冻结前缀、服务完成、状态计时、返仓推进 | 修改未执行后缀、提交新的派车头部 |
| **决策提交接口** `coordinator.submit(...)` → `container.publish(...)` | `D3CommitCoordinator` | 在认证通过后写入未执行后缀与新派车头部 | 直接推进物理时间/服务完成 |

**禁止**：env 一份、container 一份长期并行的 live 状态（用户明令）。两个接口都必须向容器**证明身份**（执行推进提交 `authority`，决策提交提交 `capability` + 临界区 + epoch），容器是唯一的字段写者。

**接入方式（AT2B 前段，默认不启用）**：`StrictOnlineEnv` 增加一个可关闭的 `state_authority` 形参（缺省 `None` = 现行行为完全不变）；非 `None` 时 `run()` 把 `vehicles/traces/served_mask` 的所有权移交给容器，自身只保留只读视图与推进接口。**默认关闭**、与历史代码零行为差异，才允许进入独立隔离验收。

## 2. Q2：哪些状态修改需要认证（机械清单 + 权限分类）

`strict_online_env.py` 共 **115 个写入点**：`DECISION_WRITE 28` / `EXECUTION_ADVANCE 54` / `UNCLASSIFIED 33`。分类键是**所属函数**（不是目标字段）——因为同一个 `status` 既被 `_commit`（决策）也被 `_advance_fleet`（执行）写入。

### 2.1 决策写入（必须经过认证事务）

| 函数 | 目标字段 | 触发 | 需重新认证 | 失败语义 |
|---|---|---|---|---|
| `_commit` | `status`、`committed_next`、`committed_arrive`、`committed_finish`、`ready_time`、`committed_coldchain_before`、`return_finish`、`dispatch_time` | 决策点提交头部 | **是** | `partial_possible` |
| `_plan_and_commit` | `mutable_suffix`（reveal 清尾 / 计划写回）、`needs_replan` | 决策点 | **是** | `partial_possible` |
| `prepare_decision_point` | `current_node`、`current_load`、`ready_time` | 决策点准备（锚点一致化） | **是** | `refuse_before_write` |
| `GreedyReplanner.plan`（及 B3-AC 清点的 7 个 replanner 站点） | `mutable_suffix` | replanner 返回计划 | **是** | `refuse_before_write` |
| `_force_return_all` | **无直接写入**（委派 `_commit` + `_advance_fleet`） | 收尾 | 由被委派函数决定 | `partial_possible` |

### 2.2 执行推进（由 env 执行原语处理，需要不变量而不是认证）

`_advance_fleet`（33 个写入点：`status`、`current_node`、`current_load`、`ready_time`、`committed_next`=None 清理、`coldchain_*`、`traces[*].services`、`traces[*].depot_unload_records`、`served_route`、`served_mask`）、`_advance_coldchain_to`、`_dispatch_coldchain`、`_run_loop`（时钟推进、reveal 清尾、booking_end）、`run`、`run_resumed`。

**必须满足的不变量（预声明，供 AT2B 前段逐条验收）**：

1. 执行推进**不得**改变未执行后缀的客户集合（只能消费头部、清理已完成项）。
2. 服务**恰好一次**：`served_mask` 置位与 `traces[*].services` 追加在同一次推进内成对发生。
3. 时钟单调：任何 `clock` 不得回退（`_advance_coldchain_to` 已有 `end < start` 抛错）。
4. 执行推进不得引入**未经认证的新客户**（不得把新客户写进任何车辆后缀）。
5. 冷链与品质只经由 `transition_segment` 推进，不得直接改 `coldchain_state` 内部字段。

### 2.3 第三类：构造期与局部量（33 个 UNCLASSIFIED 的逐组裁定）

| 函数 | 数量 | 裁定 | 理由 |
|---|---|---|---|
| `__init__` | 27 | `CONSTRUCTION_ONLY`（既非决策也非执行） | 仅写入构造期配置与只读引用数据（`dataset`/`coords`/`tw_*`/`reveal_time`/`dist_mat`/钩子/计数器初值），**不写车辆状态**；发生在任何执行实例存在之前 |
| `get_reserved_customers` | 2 | `LOCAL_ONLY` | 只累加一个**新建的局部集合**并返回；不触及权威状态 |
| `_greedy_suffix` | 3 | `LOCAL_ONLY` | 只构造**局部** `route`/`unvisited` |
| 模块级 | 1 | `MODULE_CONSTANT` | `sys.path` 常量 |

**结论**：33 个 UNCLASSIFIED 全部裁定为**非运行时状态写入**，无一属于"需要认证的第三类"。该裁定必须显式记录（用户要求"不得默认归类"），并随清单冻结。

### 2.4 两个必须记录的架构发现

1. **基类 `Replanner.plan` / `on_reveal` 不写任何状态**（抽象方法）：真正的决策写入来自**子类**（`GreedyReplanner.plan` 在此文件；另外 7 个 replanner 站点在 `evaluation/` 与 `research_20261010/`，已由 B3-AC 分类器清点，全部为 `INDIRECT_WRITER_CAN_WRITE_D3_STATE`）。**接口移交必须同时覆盖子类**，否则出现第二套状态。
2. **reveal 清尾不发生在 `on_reveal` 里，而在 `_run_loop`（`v.mutable_suffix = []`）**：这属于**执行推进**（新信息使旧尾部失效），但它的后果等价于一次决策撤销。AT2B 前段必须裁定它到底归哪一类——本设计建议：**归执行推进，但必须满足不变量 1 的加强版（清空尾部时同时清空 `needs_replan` 语义并登记 reveal 事件）**，且该清空**不得**被计入"未经认证的新客户写入"。

## 3. Q3：如何证明接入前后协议一致（预声明回归规格）

**输入必须相同**：相同初始状态、相同订单揭示序列、相同合法决策输入。

**比较项**：车辆服务顺序与时刻 / 承诺冻结 / 订单恰好服务一次 / `certify_plan` 的 C0 能耗口径 / 冷链状态与品质记录 / 拒绝·WAIT·超时等协议行为。

**两类比较必须分开（用户要求，不得合并统计）**：

| 类别 | 判据标签 | 结论字段 | 预期 |
|---|---|---|---|
| **兼容性比较** | `COMPATIBILITY` | `regression: true/false` | 合法行为的结果应与历史 env **一致**；差异 = 回归 |
| **安全修复比较** | `SAFETY_FIX` | `expected_divergence: true` | 非法行为原先可能被写入，新路径应在**写入前**拒绝；轨迹差异是**预期结果**，**不得**误记为回归失败 |

## 4. AT2B 授权门槛（本批后的状态）

| 门槛 | 状态 |
|---|---|
| AT2G 隔离容器机制 | 已有验收证据 |
| **env 所有权移交完整设计** | **本批建立**（§1） |
| **决策写入与执行推进权限分类** | **本批建立**（§2） |
| 真实执行路径无旁路的调用图证明 | **未建立**（AT2B 前段） |
| 正式品质认证 D8 | **仍阻断** |
| 正式预算下的安装资格 | **未建立** |

因此 **AT2B 不得作为"完整生产接入 + 真实安装 + P2 通过"一次执行**，必须拆为：

1. **AT2B-1**：生产代码中的**可关闭、默认不启用**的 D3 专用接线（+ 调用图证明）。
2. **AT2B-2**：独立隔离环境验收（用本批 §3 的回归规格）。
3. **AT2B-3**：仅在 1、2 通过后，考虑启用生产真实写回。

每一步单独预声明、单独裁决。

## 5. 研究方向提醒（照录并强制记录）

从 B3-Q 到 AT2G 的认证与提交安全工程**不构成 MaskCO 的算法创新**。即使最终完成 P2，也**不得**把"拥有了安全提交机制"写成"生成模型提升了在线决策质量"；两条证据链必须继续分开。本批所有产物记录 `algorithmic_gain_claim: none`。

## 6. 非声称

- 不声称 P2；不声称唯一写入者在生产路径成立；不声称 D8 已解除；不声称正式预算下的安装资格。
- 不声称抵御反射或恶意内存修改（威胁模型见 §0）。
- 本批**只是设计**：完成设计**不**自动启动 AT2B（用户停止条件）。
