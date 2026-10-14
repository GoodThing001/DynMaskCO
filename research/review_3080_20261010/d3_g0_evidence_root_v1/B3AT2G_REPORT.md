# D3-G0 Evidence B3-AT2G — 结果报告（状态所有权封闭，隔离实现）

- 预登记：`B3AT2G_COUNTEREXAMPLE_SPEC.json`；设计：`B3AT2G_DESIGN.md`；矩阵：`B3AT2G_CONTAINER_MATRIX.json`
- 原始输出：`raw/B3AT2G_stdout.txt`、`raw/B3AT2G_repeat{1,2,3}.txt`、`raw/B3AT2G_instance{0..5}.txt`、`raw/B3AT2G_CONTAINER_MATRIX_instance{0..5}.json`
- 命令：`python -B scripts/tests/test_d3_b3at2g_container.py --instances 0 --heavy-cap 8`

## 0. 结论

**AT2A 中被准确记录的那条旁路（`mutable_suffix` 直接赋值）已被结构性关闭**：权威对象私有 + 只读投影 + 摄入深拷贝 + 唯一变更入口。三个子证明各自成立，机制正向控制成立，守卫与容器均未接入正式运行。

## 1. 逐行判定（13 行全部到达预登记判定；实例 0–5 完全一致；实例 0 三连跑一致）

| id | 判定 | 关键证据 |
|---|---|---|
| `PRE` | `SETUP_OK` | 8 个捕获决策点只读；谓词与每个候选的通过/失败全部记录 |
| `G-F10-ATTR` | `BLOCKED_AND_STATE_UNCHANGED` | `view[0].mutable_suffix = [999]` → `FrozenStateError`；容器字段与 epoch 前后不变 |
| `G-F10-NESTED` | `BLOCKED_AND_STATE_UNCHANGED` | 原地 `append`、嵌套 `coldchain_state` 赋值、对已冻结对象再赋值 —— 全部抛错；状态不变 |
| `G-F10-REPLACE` | `BLOCKED_AND_STATE_UNCHANGED` | `view.__setitem__/append/__delitem__` 全部抛错；view 长度不变 |
| `G-F10-STALE` | `STALE_REFERENCE_INEFFECTIVE` | 容器建立**前**泄露的引用改 `mutable_suffix=[777]`、`committed_next=888`、调用方列表 append、逐车 `suffix=[555]` → 权威状态与 epoch **均不变**；脱离副本可自由修改但同样不影响容器 |
| `G-F8-RACE` | `RACE_PREVENTED_OR_SERIALIZED` | 真实线程在发布期间：view 赋值被 `FrozenStateError` 阻止、`_commit` 被实例守卫 `UnauthorizedWrite` 拒绝、**脱离副本赋值按设计允许且对容器无效**；容器发布 `COMMITTED`；临界区外发布 `REFUSED_NO_MUTATION` |
| `G-ONCE` | `AT_MOST_ONE_COMMITTED` | 两真实线程同一票据：至多一次有效提交；注册表仅进程内 |
| `G-PARTIAL` | `PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS` | 第 2 辆车异常 → `PARTIAL_COMMIT`、`n_published_before_failure=1`、`rollback_claimed=false`、真实字段落库；未报成功 |
| `G-POST` | `COMMITTED_VERIFIED_FROM_CONTAINER` | 重读来自容器权威对象且与授权结果一致；临界区外发布被拒且 **`nothing_written`** |
| `G-D8` | `REFUSED_BEFORE_PUBLISH` | 正式认证不完整 → 拒绝，`publish_state=null`，容器发布次数 0；未放宽 D8、未放宽预算 |
| `G-POS` | `COMMITTED_VERIFIED_FROM_CONTAINER` | 合法非空计划（25 车有客户）经机制门完成隔离发布并验证；`counts_as_install_qualification=false` |
| `ISOLATION` | `OK` | 容器/协调器零非测试导入者；守卫未装到生产实例；历史七条基线修改 0；`MASKCO_code/` 未触碰；排除三个授权写入者后 D3 内写入原语调用 0 |
| `PREREG` / `AUDIT` | `OK` | 10 条反例全覆盖；无 hook 新 env 复现同一生产轨迹哈希 |

`sub_proofs = {1_stale_reference_invalidation: true, 2_publish_exclusivity: true, 3_failure_handling: true}`

## 2. 与 AT2A 的对照（本批只解决那一个已复现缺陷）

| | AT2A | AT2G |
|---|---|---|
| `mutable_suffix` 直接赋值 | `REFUSED_BUT_ONE_WRITE_PATH_ESCAPED`（检测不阻止） | **被阻止**（投影不可写 + 权威对象私有） |
| 写入能力排他（A 项） | 未达成（部分） | **本批范围内达成**：三条写入路径均被阻止或结构上无效 |
| 票据一次性（B 项） | 进程内达成 | 保持（`G-ONCE`） |
| 失败一致性（C 项） | 在指定注入下达成 | 保持并细化：授权拒绝与部分发布**分开记录**（见设计 §5.2） |

## 3. 明确非声称（照录用户裁定）

- **`max_concurrent_publish_windows = 1` 只证明受控窗口无重叠**，不证明外部观察者永远看不到中间状态 ⇒ **不得**称为整体原子操作。
- 不声称原子性 / 崩溃恢复 / 跨进程排除；票据注册表仅进程内。
- **本批通过不自动授权 B3-AT2B**。
- **D8 仍是独立阻断**：唯一写入者正确 ≠ 可提交计划满足鱼可售硬约束。
- 机制控制与 `ISOLATION_ONLY` 预算不能证明正式 `B=300` 下的安装资格；`energy_budget_exceeded` 保留为独立拒绝原因。
- 即使未来 P2 成立，也只说明 D3 目标路径的提交全部经过认证入口，不自动证明 G-C 的跨事件持续安全，更不能追认 MaskCO 的算法增益。

## 4. 只读与禁用项

| 禁项 | 状态 | 证据 |
|---|---|---|
| B3-AT2B / 生产接入 | 未做 | `container_registered_on_env=false`、`guard_on_a_production_instance=false`、`ISOLATION=OK` |
| 修改历史七条基线 | 未做 | `historical_baselines_changed=false`、AST 派生 |
| 修改 `MASKCO_code/` | 未做 | `maskco_code_untouched=true` |
| 解除 D8 | 未做 | `G-D8` 阻断；`d8_relaxed=false` |
| 真实写回 | 未做 | `production_state_written=false`；`AUDIT=OK` |
| 改 `_commit` 语义 | 未做 | 实例级守卫 + `restore()`；`strict_online_env.py` = `fd17fe77…` |
| G1/G2 或 F−D 实验 | 未做 | 无任何效用数值 |

## 5. 本批未得出

- 不声称唯一写入者在**生产路径**上成立（P2 未建立）；本批只证明隔离容器内状态所有权成立。
- 不声称原子性；不声称 D8 已闭合；不声称 G-L/G-C。
- I3 正式负结果、旧 D2 `not_qualified`、`stable_online_gain=False`、G1/G2 未启动、MaskCO 可归因 F−D 增益未建立 —— 全部不变。

## 6. 停止条件核对

用户设定的停止条件："上述反例获得独立可复核结论，守卫未接入正式运行，生产数据与历史基线均未改变。不以测试通过自动授权 AT2B。"

- 13 行确定判定、6/6 实例一致、3 次连跑一致 ✓
- 三个子证明**分别**给出结论（不是一项推另两项）✓
- 守卫/容器未接入正式运行、生产数据与历史基线未改、`MASKCO_code/` 未触碰 ✓
- 机制正向控制存在且不计作安装资格 ✓
- 非声称已逐条写入矩阵与本文档 ✓
