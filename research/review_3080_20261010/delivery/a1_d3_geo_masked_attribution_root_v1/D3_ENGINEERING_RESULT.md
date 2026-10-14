# D3-Engineering 验收结果（D-Geo 适配器 + T-G1..T-G9）

**日期**：2026-10-14 · **状态**：`D3-Engineering 完成`；**科学执行保持关闭**
**授权**：用户 2026-10-14「批准开始 D-Geo 实现及 T-G1～T-G9 测试」——**仅** D3-Engineering 的代码开发与工程验证。
**未做**：未做 F−D 数值比较；未训练；未前瞻采集；未跑正式 32 状态比较；未进入 D3-Science。

---

## 1. 交付物

| 交付 | 路径 |
|---|---|
| D-Geo 适配器（新增） | `C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py` |
| T-G1..T-G9 + X1/X2 测试（新增） | `C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_geo_adapter.py` |
| 测试原件输出 | `test_run_output.txt` |
| 源码身份清单 | `D3_SOURCE_IDENTITY.json`（+ `build_d3_identity.py`） |
| 动作字段与 tie-break schema | `D3_ACTION_TIEBREAK_SCHEMA.md` |

**结果：32 通过 / 0 失败**（解释器 `D:\AA_Py\A_Anaconda\python.exe`，jax 0.5.0 / numpy 1.26.4）。
**历史文件零修改**：`git status` 在扩展目录内**只有两个新增未跟踪文件**，既有冻结树与历史结果**字节未动**。

## 2. 三处用户技术修正的落库

### 2.1 32 条状态：**仅作状态索引，恢复性尚未证明**（用户修正 1）

* 本轮**没有**声称完成 32 条真实固定状态的工程验证。
* `sgbs_fixed_cal/per_state.json` 只含**标量评价结果**（`inst/event/J0/各臂 J`），**不是**完整
  `RecourseSnapshotV2`。因此按用户裁定：这些 `(inst,event)` **可以**用于**静态 schema 检查**，
  **不能**作为"已完成 32 条真实固定状态工程验证"的依据。
* **状态可恢复性验证尚未执行**；若恢复失败，**不得事后删除**该状态（须原样记录）。
* 处置：列入 §5「待办/待冻结」，未授权前不启动。

### 2.2 距离平局：**统一最小参照**，且 `eps_d` **未冻结**（用户修正 2）

实现为 `unified_minimum_tie_set`：

```
d_min = min_i Δd_i
eps   = max(eps_abs, eps_rel · |d_min|)
T     = { i : Δd_i ≤ d_min + eps }
在 T 内按规范动作键取最小
```

* **不使用两两近似相等**（近似相等**不具传递性**）。T-G5 用
  `[1.0, 1.0+0.9e-12, 1.0+1.8e-12]`、`eps=1e-12` 构造出"0~1 且 1~2 但 0 不~2"的链；
  统一最小参照给出 `T = {0,1}`、**排除 2** ⇒ 不依赖比较顺序。
* `eps_d` 仍为**工程暂定值**（相对 `1e-9` / 绝对 `1e-12`），产物中**逐处标记 `eps_is_provisional`**；
  `distance_stability_report()` 提供冻结前的数值稳定性证据钩子，并明确写 **NOT frozen**。
  ⚠️ **不再**引用别处的 ~1e-5 模型噪声作为距离误差依据（用户指出二者不同源）。
* **容差只作用于 D-Geo 的距离评分**。F 的学习 logits **不套用**该容差；X2 断言 F 仍是
  **首个最大值 `argmax`**，且 1e-13 的间距**不**被当成平局（`argmax` 返回 index 1）。

### 2.3 tie-break 使用**真实字段类型**（用户修正 3）

见 `D3_ACTION_TIEBREAK_SCHEMA.md`：`slot.kind ∈ {'anchored','new_route'}` 用**声明式秩映射**
`{'anchored':0,'new_route':1}`；`slot.anchor` 为 int（`new_route` 恒为 `-1`，与 anchored 不可能碰撞）；
`vehicle_id` 由**确定性物理匹配**解析（`_match_anchored` / `_match_new_route`）。
T-G1 断言三个真实候选的目标车解析为 `[1, 3, 2]`（anchored / depot-route 负编码 / new_route）。

## 3. 测试结果（T-G1..T-G9 + X1/X2）

| 编号 | 断言 | 结果 |
|---|---|---|
| T-G1 | 索引一一对应；逐项 **bit-equal** 于 `legal[i].cand.incremental_distance`；目标车确定性解析 | 4/4 PASS |
| T-G2 | 冷链/shadow/接拒字段**不影响** `geo_score`；缺键**抛错**（无静默回退评分器） | 2/2 PASS |
| T-G3 | `pool_B` **采样前**构造并剔除 deferred；保序；**空池原样记录且不丢弃** | 3/3 PASS |
| T-G4 | 裸 `set` **被拒**；有序输入原样保留；20 次进程内 + **两次独立进程** steps 完全相同 | 4/4 PASS |
| T-G5 | 统一最小参照（非传递性链被切断）；平局取最小规范键；**对候选顺序置换不变**；规则标记为暂定 | 5/5 PASS |
| T-G6 | 完整认证不过 ⇒ `cert_fail`（不采纳）；认证明细保留 | 2/2 PASS |
| T-G7 | `no_legal`/`nan`/`timeout`/`cert_fail` **全部**记录；预算**五项分别报告** | 2/2 PASS |
| T-G8 | 两臂首步可不同（真实分歧）；**分歧不产生失败理由** | 2/2 PASS |
| T-G9 | 四模块逐文件 SHA 记录；历史 `stepwise_sample` **未被替换**；适配器**从不赋值**历史采样器 | 3/3 PASS |
| **X1** | 历史模式**精确复现**原 `stepwise_sample`（同一 legal 列表，monkeypatch 隔离枚举器）；**增量键恰好多一个**（已固定） | 3/3 PASS |
| **X2** | F 规则**未变**（首个最大值）；**无距离容差泄漏** | 2/2 PASS |

## 4. 测试过程中发现的两项实现性事实（如实记录，未擅自"修绿"）

### 4.1 `apply_action` 第三位参数是 `src_vid`，不是 `allowed_vehicle_ids`

`action_contract.apply_action(plans, action, src_vid=None, allowed_vehicle_ids=None)`。
我最初按位置传 `mutable_ids` ⇒ `TypeError: unhashable type: 'set'`。
**已修**：适配器改用关键字 `allowed_vehicle_ids=`（与 `mpre_policy` 自身调用一致）。
⇒ 这是一条**接口陷阱**，后续任何新臂都必须用关键字调用。

### 4.2 共享采样器**不校验** `len(scores) == len(legal)`

历史 `stepwise_sample` 直接 `legal[idx]`。若某评分器返回长度不符的向量，
会得到 **`IndexError`（崩溃）**，而不是 **fail-closed 拒绝**。
D-Geo 侧不受影响（其长度由共同 `extract_fn` 保证），但这对 **F 臂**是潜在风险。
**未修**：在共享路径内加长度校验会**改变 F 的行为**，属新决策策略，须单独声明 + 等价测试。
⇒ 列入 §5 待办。

### 4.3 X1 记录的一处**受控增量**

适配器的 steps 比历史多**恰好一个**字段 `incremental_distance`（保留为所选动作的
环境距离证据）。已用断言固定该增量，**不是**让比较变宽松。

## 5. 待冻结 / 待办（**均未授权启动**）

| # | 项 | 现状 | 启动条件 |
|---|---|---|---|
| 1 | `eps_d` 冻结 | **暂定**，逐处标记 provisional | 先做**距离运算数值稳定性**验证（`distance_stability_report` 已有钩子） |
| 2 | 32 条状态**可恢复性**证明 | **未执行**；当前只能称"状态索引可用于静态 schema 检查" | 另行授权；恢复失败**不得事后剔除** |
| 3 | 共享采样器长度不匹配的 fail-closed 处置 | 发现未修（会改变 F 行为） | 若修，须声明为新决策策略 + 等价测试 |
| 4 | 评估预算 | 工程阶段**五项分别计数**已实现；**上限未冻结** | Science 前**独立**固定墙钟 + 同定义候选工作量；工程最大值**仅作容量规划** |
| 5 | T-G9 **正式冻结身份验收** | 已记录当前 SHA，但 **`frozen: false`** | 源码真正冻结后重做（新冻结树 + 逐文件 SHA + 环境 + 独立只读验收） |
| 6 | D3-Science | **未授权** | E2 训练资格 + 前瞻采集资格 + 完整 8+2 秒发布门 + 统计预声明 |

## 6. 源码身份（`D3_SOURCE_IDENTITY.json`）

| 文件 | 当前 SHA-256（前 16） | vs F 训练身份 |
|---|---|---|
| `scripts/simulation/action_contract.py` | `9d64273aa3df98cb…` | **一致** |
| `scripts/simulation/mpre_policy.py` | `5b03e4cae812fa20…` | **一致** |
| `scripts/simulation/dynmaskco_cc_context.py` | `4eebb2a51a024a9e…` | 不一致（故 B 门在新树内显式实现） |
| `scripts/simulation/cc_lns_replanner.py` | `ae634f5323595aaf…` | 不一致（D-Geo 不用它作主对照） |
| `scripts/simulation/d3_geo_adapter.py`（新增） | `41e9dfb8baed7b1b…` | — |
| `scripts/tests/test_d3_geo_adapter.py`（新增） | `f51ba071d21c0a74…` | — |

⇒ **正面结论**：D-Geo 必须与 F 共享的**动作枚举层没有漂移**（`action_contract` / `mpre_policy` 逐字节一致）；
漂移只在**池的合法性门**与 **D 臂搜索**上——这正是采用 B 门并显式实现的原因。

## 7. 本轮的**不**主张

* 不主张 MaskCO 有效或无效；不主张 D-Geo 与 F 有任何效果差异；
* **不把"工程实现完成"升级为"主模型有效性获得证明"**；
* 不追认旧 D2（其 `not_qualified` 保持）；
* 不因本轮通过而启动 F−D、E2 训练、正式 32 状态比较或 D3-Science。

## 8. 研究判断（维持）

D1 已建立资源价格的可识别机制；旧 D2 仍为 `not_qualified`；**D3 处于工程准备**；
**MaskCO 稳定核心在线增益仍未建立（`False`）**。
