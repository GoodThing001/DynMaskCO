# D3 动作字段与 tie-break schema（D-Geo / F 共用部分）

**日期**：2026-10-14 · **状态**：`D3-Engineering schema`（`eps_d` 仍为暂定）
**依据**：`action_contract.py` 实际字段类型 + 用户 2026-10-14 技术修正 3。

---

## 1. 动作对象的真实字段类型（取自源码，不是转述）

```python
@dataclass(frozen=True)
class ActionSlot:
    kind: str      # 'anchored' | 'new_route'
    anchor: int    # >0: 该车 anchor_node；<0: depot-route 车，vid = -anchor-1；new_route 恒 -1

@dataclass(frozen=True)
class FleetAction:
    customer: int
    slot: ActionSlot
    position: int        # 0..len(suffix)
    predecessor: int     # 插入后前驱（anchor_node 或 suffix 元素，0=depot）
    successor: int       # 插入后后继（suffix 元素或 0=return depot）
    incumbent: bool = False
```

**slot.anchor 编码**（`action_contract.py:42-45`）

| 取值 | 含义 |
|---|---|
| `> 0` | `ready@customer` 或 `committed` 车的 `anchor_node` |
| `< 0` | `idle@depot` 但已有非空 suffix 的 depot-route 车，`vid = -anchor - 1` |
| `-1`（配 `kind='new_route'`） | 匿名，物理匹配后置 |

## 2. 规范动作键（**唯一排序依据**）

```python
canonical_action_key(customer_id, action, vehicle_id) -> tuple
    (customer_id,
     vehicle_id,                     # 由确定性物理匹配解析（见 §3）
     SLOT_KIND_RANK[action.slot.kind],
     action.slot.anchor,
     action.position,
     action.predecessor,
     action.successor)
```

**字符串排序规则（显式声明）**：`slot.kind` **不**用字典序隐式比较，而用**声明式秩映射**

```python
SLOT_KIND_RANK = {'anchored': 0, 'new_route': 1}
```

⇒ 顺序固定为 `anchored` 先于 `new_route`；未知 `kind` **抛错**（不静默排到某处）。
**`slot.anchor` 为 int** ⇒ 直接整数比较，无字符串歧义。
**`new_route` 的 `anchor` 恒为 `-1`**，与任何 anchored 的 anchor 都不可能碰撞 ⇒ 键不会退化。

## 3. `vehicle_id` 的确定性解析

```python
resolve_vehicle_target(plans, action, allowed_vehicle_ids):
    kind == 'new_route' -> action_contract._match_new_route(plans, allowed_vehicle_ids)
    否则                -> action_contract._match_anchored(plans, action.slot.anchor)
```

* `_match_new_route`：在所有 `anchor_node == 0 and suffix == []` 的 idle 车中，按
  `allowed_vehicle_ids` 约束确定性选取（同一 `plans` 必然同一结果）。
* `_match_anchored`：`anchor > 0` 找 `anchor_node == anchor`；`anchor < 0` 取 `vid = -anchor-1`。

**跨进程一致性**：`vehicle_id` 只依赖 `plans` 与 `allowed_vehicle_ids`（整数集合），
不含字典迭代顺序依赖 ⇒ 匿名 `new_route` 与普通 anchored 动作在**不同进程**中形成**同一键**。
T-G1 断言解析结果 `[1, 3, 2]`；T-G4 断言**两个独立进程**产出逐字节相同的 steps。

## 4. 选择规则：两条路径，**不得混用**

| 臂 | 规则 | 实现 |
|---|---|---|
| **D-Geo** | 统一最小参照 + 规范键序 | `unified_minimum_tie_set` → `select_geo` |
| **F** | **首个最大值** `np.argmax`（历史规则，**未改动**） | `select_historical_argmax` |

```python
d_min = min_i Δd_i
eps   = max(eps_abs, eps_rel * |d_min|)        # eps 暂定：1e-12 / 1e-9
T     = { i : Δd_i <= d_min + eps }
idx   = argmin_{i in T} canonical_action_key(i)
```

⚠️ **禁止**两两近似相等（不具传递性，会让结果依赖比较顺序）。
⚠️ **距离容差不得套用到 F 的学习 logits**：那会变成新的决策策略，须单独声明 + 等价测试。

## 5. 顺序纪律（硬性）

1. 传入适配器的 `mask_set` **必须是有序序列**；裸 `set`/`frozenset` **抛 `TypeError`**。
   理由：`stepwise_sample` 内 `remaining = list(mask_set)`，而 `argmax` 取**首个最大值**
   ⇒ 顺序未定则平局结果未定。
2. `enumerate_legal_actions` 的顺序由调用方传入的 `remaining` 顺序决定，**须落盘记录**。
3. 每步的 `selection`（`d_min` / `eps_used` / `tie_set` / 选中项）与所选动作的
   `incremental_distance` 都进 `records`，供事后审计。

## 6. 预算计数（五项**分别**记录，不可互相折算）

```python
Budget.KEYS = ('n_actions_enumerated', 'n_actions_scored',
               'n_candidate_plans_certified', 'n_solver_calls', 'wall_seconds')
```

理由（用户裁定）：这五项**不是天然等成本的工作单位** ⇒
工程阶段的最大值**只能作容量规划**；D3-Science 必须在读取新鲜状态结果前**独立**固定预算，
至少同时约束**墙钟**与**同定义的候选工作量**，并记录实际消耗与超时回退。

## 7. 失败语义（全部入账，分母不缩）

```python
FAILURE_REASONS = (None, 'no_legal', 'nan', 'timeout', 'rollback', 'cert_fail')
```

* `no_legal` 无可行动作；`nan` 评分非有限；`timeout` 触及墙钟上限；
  `cert_fail` 完整认证（计划对账 / 分区 / C0）不过 ⇒ **不采纳**并回退；
  `rollback` 回退原计划。
* 认证**不得**只凭插入动作的局部 `feasible=True` 认定整条计划合格。

## 8. 已知接口陷阱（本轮实测）

| 陷阱 | 现象 | 正确做法 |
|---|---|---|
| `apply_action` 第三位参数是 `src_vid` | 按位置传 `mutable_ids` ⇒ `TypeError: unhashable type: 'set'` | **必须**用关键字 `allowed_vehicle_ids=` |
| 共享采样器不校验 `len(scores) == len(legal)` | 长度不符 ⇒ `IndexError` 崩溃（非 fail-closed） | 评分器须自行保证长度；若要改共享路径须另立新策略 + 等价测试 |
