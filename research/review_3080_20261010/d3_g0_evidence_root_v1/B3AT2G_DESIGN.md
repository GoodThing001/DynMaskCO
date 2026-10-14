# D3-G0 Evidence B3-AT2G — 状态所有权与直接属性写入旁路封闭（隔离实现）

- 阶段：`D3_G0_Evidence_B3AT2G`（受限；`B3-AT2B` 未授权）
- 预登记（先于实现冻结）：`B3AT2G_COUNTEREXAMPLE_SPEC.json`，sha256 见 `B3AT2G_BYTE_FREEZE.json`
- 基础提交：`6653d3fa546d32dc5a339dec4bda375e50c3b751`
- 产物：`scripts/simulation/d3_state_container.py`、`scripts/tests/test_d3_b3at2g_container.py` → `B3AT2G_CONTAINER_MATRIX.json`

## 0. 只针对 AT2A 已复现的那个结构缺陷

AT2A 的结论是 `AT2-F10 = REFUSED_BUT_ONE_WRITE_PATH_ESCAPED`，诊断是"守卫保护的是部分函数入口，而不是状态对象本身"。本批不做重复验证，只针对这一点提出并检验解决方案：

**由 D3 专用状态容器独占真实 `VehicleState`，外部模块只接收只读快照或脱离副本。**

## 1. 强制机制（结构性，而不是包裹更多方法）

| 机制 | 做法 | 关闭的路径 |
|---|---|---|
| **摄入即深拷贝** | `D3StateContainer.__init__` 对 `vehicles/traces/served_mask` 全部 `deepcopy` | 容器建立**之前**泄露出去的引用（`G-F10-STALE`） |
| **私有权威对象** | 权威对象只存在 `container._vehicles`，`_authoritative()` 不对外暴露 | 任何"持有 live 引用再改"的路径 |
| **只读投影** | `read_only_view()` → `ReadOnlyStateView`（不可变序列）+ `FrozenVehicle`（`mutable_suffix` 是 **tuple**；`__setattr__`/`__delattr__` 抛 `FrozenStateError`） | 属性赋值、原地 list 修改、嵌套修改 |
| **元素替换/增长被拒** | `__setitem__` / `__delitem__` / `append` / `extend` / `insert` / `pop` / `sort` 全部抛 | `vehicles[i] = other` 之类 |
| **唯一变更入口** | `container.publish(...)` 要求**令牌存活 + 位于协调器临界区 + epoch 匹配**，且**授权前置检查发生在任何 mutation 之前** | 未授权 mutation 残留 |
| **发布后独立重读** | `container.post_read(...)` 读的是**容器自己的权威对象** | "重读"来自副本或缓存 |

留在原处的第二层：AT2A 的实例守卫仍包住 `_commit` / `_plan_and_commit`。

## 2. 三件必须分别证明的事（所有权 ≠ 原子性）

| 子证明 | 行 | 结论 |
|---|---|---|
| **旧引用失效** | `G-F10-STALE` / `G-F10-ATTR` / `G-F10-NESTED` / `G-F10-REPLACE` | **成立** |
| **发布排他**（认证/消费/发布/重读在同一受控版本边界） | `G-F8-RACE` / `G-ONCE` | **成立** |
| **失败处理**（部分发布 / 验证失败 / 结果未知分别记录） | `G-PARTIAL` / `G-POST` | **成立** |

三项在矩阵中各自独立成字段（`sub_proofs`），一项通过不得推断另外两项。

## 3. 明确的非声称

- **`max_concurrent_publish_windows = 1` 只证明受控窗口内没有重叠**，**不**证明外部观察者永远看不到中间状态；因此**不得**把多车辆发布称为整体原子操作。
- 不声称原子性 / 崩溃恢复 / 跨进程排除。
- 本轮通过**不自动授权 B3-AT2B**。
- 唯一写入者正确 **≠** 可提交计划满足鱼可售硬约束：**D8 仍是独立阻断**。
- 机制控制与 `ISOLATION_ONLY` 预算**不能**用于证明正式 `B=300` 预算下的安装资格；`energy_budget_exceeded` 保留为独立拒绝原因。

## 4. 机制正向控制为何必须存在

`G-POS` 用合法非空计划经机制专用门走完整容器发布并验证。没有这一行，"所有越权写入都失败"完全可能只是**状态对象根本不可写**，而不是权限控制正确。该行同时标注 `counts_as_install_qualification=false`、`not_a_production_path=true`、`does_not_establish_P2=true`。

## 5. 本批自查发现并修掉的两个实现缺陷（都会制造假结论）

1. **未授权 mutation 残留**：第一版把授权检查交给守卫在 `_commit` 处隐式完成，于是容器**先**把 `mutable_suffix` 写进自己的权威对象、**才**被拒——既留下了未授权变更，又因此破坏了下一次合法发布的 epoch，表现为"连合法发布都被拒"。现改为**授权前置检查在任何 mutation 之前**（`not_authorised_before_any_mutation` + `nothing_written=true`）。
2. **把授权拒绝误报成 `PARTIAL_COMMIT`**：异常的宽 `except` 会把 `UnauthorizedWrite` / `FrozenStateError` 记成"部分发布"。这违反词汇表定义（`PARTIAL_COMMIT = 已确认部分状态改变`），且会把授权失败藏进状态变更报告里。现按异常类型分流，并用**前后字段比较**确认是否真有部分变更；若异常但字段未变 → `OUTCOME_UNKNOWN`。

## 6. 只读与边界

只针对 D3 专用执行实例；守卫与容器都不注册到任何生产 env；不改 `_commit` 源码语义（实例级包裹 + `restore()`）；**不修改历史七条基线**，**不动 `MASKCO_code/`**；`strict_online_env.py` 保持 `fd17fe77…`。
