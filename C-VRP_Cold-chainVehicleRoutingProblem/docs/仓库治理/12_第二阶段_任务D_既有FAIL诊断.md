# 12 · P0 第二阶段 · 任务 D：既有 FAIL 诊断

`scripts/tests/test_rolling_optimizer.py::test_in_transit_no_double_count`

> 只读诊断。**未修改测试、未修改业务源码、未修改任何历史裁决。**
> 诊断方式：另写只读脚本在 `%TEMP%` 下复现同一场景（不写入仓库），用 `inspect.getsource` /
> `__mro__` 取**运行期真正绑定**的实现，再用参数扫描定位边界。

## 1. 结论（先说答案）

| 问题 | 答案 |
|---|---|
| 是不是"在途重复计数"缺陷复发了？ | **不是。** 该修复**在位且生效**（§4 有正反证据）。 |
| 那为什么 FAIL？ | 该用例把载重**正好压在容量边界上**（25+25 = 50 = capacity），而当前合同规定接单侧容量判据是 **`l > capacity − 1e−4`**（保守余量），**边界恰好 50 被判不可行**。 |
| 断言与当前合同一致吗？ | **不一致。** 断言的语义是"恰好满容量可行"（等价 `+1e−6` 口径）；当前合同是"必须严格小于 `capacity − 1e−4`"。 |
| 是代码退化还是测试陈旧？ | **测试陈旧**（且从未绿过，§5）。代码的收紧是 2026-09-28 **有意为之并写在源码注释里**。 |
| 影响范围 | 生产行为方向是**更保守**（只会少接单，不会多接）；真正的问题是**这一类缺陷目前没有绿的守卫**，以及**跨代方法口径不可比**（§6）。 |

## 2. 失败调用链（运行期实测，非静态推断）

```
test_in_transit_no_double_count()                     test_rolling_optimizer.py:71
└─ StrictOnlineEnv(...).run(0)                        strict_online_env.py
   └─ RollingOptimizeReplanner.on_reveal(clock=1.0)    rolling_optimizer.py:238
      ├─ _uncond_lookahead_lambda(...) -> lam = 0.0    （无预拒绝：REV[1]=20 > e·0 + 1e−9）
      └─ _optimize(new_set={2})                        rolling_optimizer.py:181
         └─ evaluate(acc={2})                          rolling_optimizer.py:189
            ├─ _start_states(...) -> {0: {cur:1, cur_time:1.0, load:25.0, route:[]}}
            └─ _greedy_insert_into(o=2, st)            rolling_optimizer.py:68
               └─ _route_feasible(...)  ← 运行期解析到
                  C0ReserveReplanner._route_feasible   run_exp_energy_c0.py:245
                     l = 25.0 + 25.0 = 50.0
                     if l > self.capacity - 1e-4:      → 50.0 > 49.9999  → True
                        return False                   ← 插入失败
            evaluate() 返回 (None, None)               → 该候选被丢弃
         └─ 唯一可行候选 = 全拒 acc=∅ , util=9.8        → best = (∅, {2})
      └─ rp._accepted |= ∅ ; rp._rejected |= {2}
└─ assert accepted == {1, 2}                           test_rolling_optimizer.py:91
   AssertionError: ({1}, {2})
```

**关键点：类继承关系**（`type(rp).__mro__` 实测）

```
RollingOptimizeReplanner → C0ReserveReplanner → Replanner → object
bound from : C0ReserveReplanner._route_feasible
source file: scripts/evaluation/run_exp_energy_c0.py
```

即：`RollingOptimizeReplanner` 继承的是 **`C0ReserveReplanner`（`run_exp_energy_c0.py`）**，
**不是** `run_exp_reserve.py` 里那个同名类。全仓有 **8 个** `_route_feasible` 实现，容量判据并不一致（§6.2）。

## 3. 判据原文（源码，逐字）

`C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_exp_energy_c0.py:245-258`

```python
def _route_feasible(self, env, inst_idx, cur, cur_time, load, route):
    t, c, l = cur_time, cur, load
    for o in route:
        d = env.dist_mat[inst_idx, c, o]
        arr = max(t + d / env.tw_speed, env.tw_start[inst_idx, o])
        if arr > env.tw_end[inst_idx, o] + 1e-6:
            return False
        l += env.demands[inst_idx, o]
        if l > self.capacity - 1e-4:   # 2026-09-28：保守容量余量 1e-4——接单侧检查必须严格于
            return False               # 物理 transition（_EPS=1e-9），杜绝 float32 边界载重执行期崩溃
        t = arr + env.service_time[inst_idx, o]
        c = o
    ret = t + env.dist_mat[inst_idx, c, 0] / env.tw_speed
    return ret <= env.tw_end[inst_idx, 0] + 1e-6
```

注释写明这是**故意的保守余量**：接单侧检查必须**严格于**物理 transition 的 `_EPS=1e-9`，
以避免 float32 边界载重在执行期崩溃。**这是一条安全约束，不是 bug。**

## 4. 反证：在途重复计数修复**在位**

`_start_states`（`rolling_optimizer.py:39-51`）对 committed/已服务车的处理：

```python
committed = v.committed_next if v.status == 'committed' else None
route = [x for x in plan.get(v.vehicle_id, []) if not served_mask[x] and x != committed]
```

诊断实测的两帧状态：

| clock | 车辆状态 | `cur` | `cur_time` | **`load`** | **`route`** |
|---|---|---|---:|---:|---|
| 0.5（单 1 揭示） | `idle` | 0 | 0.5 | 0.0 | `[]` |
| 1.0（单 2 揭示） | `ready` | **1** | 1.0 | **25.0** | **`[]`** |

- `load = 25.0` 已经包含**在途/已取到手的单 1 货物**；`route = []`（单 1 不再出现在剩余路线里）。
- 插入单 2 时从 `l = 25.0` 起累加 → **25 + 25 = 50，只算一次**。
  **若重复计数仍存在，`l` 会是 75，任何低于容量的组合都会失败** —— 但实测 49.98 通过（§5），
  说明 **`committed` 未被执行侧重复计入**。

结论：**这不是 `in_transit_no_double_count` 想守的那个缺陷。**

## 5. 边界扫描（同一场景，只改两单需求）

| 载荷合计 | 距容量余量 | accepted | rejected | `hard_feasible` |
|---:|---:|---|---|---|
| 40.00 | 10.0000 | `[1, 2]` | `[]` | True |
| 48.00 | 2.0000 | `[1, 2]` | `[]` | True |
| 49.00 | 1.0000 | `[1, 2]` | `[]` | True |
| 49.80 | 0.2000 | `[1, 2]` | `[]` | True |
| **49.98** | **0.0200** | **`[1, 2]`** | `[]` | True |
| **50.00（= 测试用例）** | **0.0000** | **`[1]`** | **`[2]`** | True |
| 52.00（真超载） | −2.0000 | `[1]` | `[2]` | True |

- **余量 ≥ 0.02 全部通过；余量恰为 0 被拒。** 边界行为与 `capacity − 1e−4` 逐字吻合。
- 真超载（52）被拒是**正确的**。
- 所有情形 `evaluate_trace_a1` 都判 `hard_feasible=True` —— 说明**评价器认可这个结果**，
  失败只发生在"测试期望"这一侧。

## 6. 影响范围与风险定级

### 6.1 对生产行为：**无错误，方向更保守**

`l > capacity − 1e−4` 只会**少接**边界订单，不会多接。它不产生不可行计划，也不破坏协议。

### 6.2 真实影响一：**跨代方法口径在"恰好满容量"处不可比**

全仓 8 个 `_route_feasible`，容量判据分两派：

| 实现 | 位置 | 容量判据 |
|---|---|---|
| `C0ReserveReplanner` | `run_exp_energy_c0.py:253`（**A-v1 全家共用**） | **`capacity − 1e−4`** |
| （同文件另一处 helper） | `run_exp_energy_c0.py:574` | **`capacity − 1e−4`** |
| （同文件第三处） | `run_exp_energy_c0.py:428` | `env.capacity + 1e−6` ⚠️ 同文件内不一致 |
| `ReserveReplanner` | `run_exp_reserve.py:165` | `capacity + 1e−6` |
| `AcceptRejectReplanner` | `run_exp_energy.py:319` | `capacity + 1e−6` |
| `AcceptRejectReplanner` | `run_exp_capacity.py:309` | `capacity + 1e−6` |
| `AcceptRejectReplanner` | `run_causal_accept_gate.py:218` | `capacity + 1e−6` |

继承关系实测：A-v1 的四个臂（`RollingOptimizeReplanner`、`UncondDynamicReplanner`、
`FixedThresholdReplanner`、`ClassPriorityReplanner`）**全部**派生自 `C0ReserveReplanner`
⇒ **A-v1 内部可比**；但与 `reserve / energy / capacity / causal` 这几代（`+1e−6`）**在恰好满容量处不可比**。
真实数据需求是整数、容量 50 ⇒ "恰好 50"（如 25+25、20+30、10+40）**是常见事件**，不是渐近小量。
**具体影响了哪些已登记数字，本轮未量化**（需要重跑才能给出，属未授权范围）。

### 6.3 真实影响二：**该缺陷类目前没有绿的守卫（最该修的一项）**

`test_in_transit_no_double_count` 是**唯一**守"在途重复计数"的用例，而它现在是红的。
后果：**未来若真把重复计数改回来，测试仍然是红的，不会报警。**
这是本轮发现里**风险最高的一条**（因为它让一类真实缺陷失去探测能力），
风险等级 **中高**，且**与本次整理无关、属既有状态**。

### 6.4 时间线：这个用例**从未在任何已提交状态下通过过**

| 事实 | 证据 |
|---|---|
| 测试文件、`run_exp_energy_c0.py`、`- 1e-4` 判据 **在同一个 commit 里一次引入** | `git log --diff-filter=A` → 三者均为 `f9a02c5`（2026-10-08 09:46:15） |
| 该 commit 里 `rolling_optimizer.py` 的 import **已经**指向 `run_exp_energy_c0` | `git show f9a02c5:…/rolling_optimizer.py` → `from run_exp_energy_c0 import C0ReserveReplanner, certify_plan`；与今天一致 |
| `- 1e-4` 行自创建起未被改过 | `git blame` → `f9a02c57 (2026-10-08)` |
| 测试文件自创建起未被改过 | `git diff HEAD -- <test>` → 空 |

⇒ **不是"以前绿、后来变红"，而是从落地那一刻起就与它所测试的实现不一致。**
（也解释了为什么没有任何文档登记这个失败：它是被一次性大批量落盘时带进来的。）

### 6.5 不影响的东西

- 不影响 `test_accept_profitable`（载荷合计 4 ≪ 50）与 `test_certify_rejects_over_budget`（B=10 预算门）——两者 **PASS**。
- 不影响任何冻结裁决、不影响 `stable_online_gain=False`、不影响 Q3/D3 任何结论。
- 不影响 `results/` 里任何已登记的正式数字（那些数字本来就带着这个 1e−4 口径产生）。

## 7. 可选处置（**本轮一个都没做，均需你批准**）

| 选项 | 动作 | 代价 | 我的评价 |
|---|---|---|---|
| **D-a（推荐）** | 改**测试夹具**：把两单需求从 25/25 改成 24/24（或 25/24），使余量 ≥1，断言继续表达"修复后两单都该接受"的原意；并在注释里写明 `capacity−1e−4` 口径 | 改 1 个测试文件、2 个数字 | **最小、最忠实于测试意图**。但仍是改测试，需你批 |
| D-b | 保留 50/50 夹具，把断言改成"边界被保守拒绝"（`accepted == {1}`），另加一个 24/24 用例守重复计数 | 改断言语义 | 承认了保守口径，但把原意图（两单都接受）丢掉了 |
| D-c | 把 `_route_feasible` 的 `−1e−4` 改回 `+1e−6` | **改生产源码** | ❌ **不建议**：源码注释说明该余量是为了避免 float32 边界载重执行期崩溃，撤掉会重新引入崩溃风险 |
| D-d | 只在文档登记，不动代码 | 零 | 缺陷类继续无守卫（风险持续） |

**无论选哪个，都必须：** ① 不改既有正式数字的历史裁决；② 若改夹具，需在结果表里注明该用例的
`capacity − 1e−4` 口径；③ 若日后量化 6.2 的跨代差异，须按新协议重新登记，不得混用旧结果。

## 8. 复现命令（只读，不写仓库）

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem
python scripts/tests/test_rolling_optimizer.py ; echo "exit=$?"     # 期望 exit=1
```

前两个用例打印 `[PASS]`，第三个抛 `AssertionError: ({1}, {2})`。
本文件 §2/§4/§5 的所有中间量均可用同一个只读脚本复算（本轮脚本位于 `%TEMP%`，未入库）。
