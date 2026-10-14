# 旧 D2 收束记录（同掩码 MaskCO 归因筛查）

**收束日期**：2026-10-14 · **基准密封提交**：`023b945`（用户指定；此后 remote `main` 已前进到 Role C 资源门 CLI 提交，
但**本 D2 的一切科学判断只以 `023b945` 对应的封存产物为准**，不混入后来提交的工作）
**裁决**：`not_qualified` —— **维持，不追加比较、不补造历史数据、不改写原失败原件**。

---

## 1. 收束时的事实清单

| # | 事实 | 证据 |
|---|---|---|
| 1 | **F 历史训练权重已找回**，SHA 与历史记录一致 | `results/m0_scale/mpre_reinforce_s42_clean/model.ckpt`，89,592,616 B，`f5acf83f…a7f5`；另有 `training_identity.json` 与冻结 `SOURCE_START/END.json` 独立记录；`B_REPRO_CLEAN/scripts/simulation/` 与 `negative_result_paper/release/v1_modules/` 亦存同一 SHA |
| 2 | **D 实现存在，但算法语义未锁定** | `scripts/simulation/cc_lns_replanner.py`（regret-2，`ae634f53…`）；文档所述 `cc_lns_swap.py` 与 `run_cc_lns_swap_fixed_state.py` **两树皆无** |
| 3 | **E2 缺失** | 搜索范围内零 E2 产物；`train_mpre_reinforce.py` 载入预训练 backbone，`--seed` 不改 backbone |
| 4 | **两个必需执行字段不可追溯** | 按真实 schema：`per_candidate_executable_actions` / `true_committed_differences` 在所有已定位封存资产中都不存在 |
| 5 | **训练期与当前实现存在源码 SHA 差异，且已查明其中两处是真实语义变化**（不再是"仅 SHA 不同"） | 见 §2 |
| 6 | **训练器本体的身份未漂移** | `scripts/training/train_mpre_reinforce.py` 当前 SHA `33c617a3…` **等于** F 训练身份记录值 |

## 2. 只读行为等价审查结论（本轮唯一被授权的动作）

审查器：`run_d2b_equivalence.py` · 结果：`D2B_EQUIVALENCE.json`
方法：**构造性函数体比较**（空白归一化后逐字节）+ **有序输出的动态对拍**（含 seeded selection，
以覆盖"集合相同但数组顺序不同导致同 seed 采样不同"的风险）。

### 2.1 `dynmaskco_cc_context.py`（掩码池模块）— 8/9 函数逐字节相同

| 函数 | 训练期 vs 当前 |
|---|---|
| `decision_pool_from_vehicles` | **相同** |
| `mask_candidate_pool` | **相同** |
| `mutable_scope` / `compute_incumbent_plans` / `_decision_context` / `cc_state_dict` / `restore_vehicles_from_json` / `validate_full_partition` | 相同 |
| **`validate_mask_scope`** | **不同** |

**`validate_mask_scope` 的实际 diff（不是注释）**：当前版本删除了整整一条断言

```python
-    mutable_plan_customers = set()
-    for vid, p in plans.items():
-        if vid in mutable_ids:
-            for x in p.suffix:
-                if int(x) > 0:
-                    mutable_plan_customers.add(int(x))
 ...
-        assert c in mutable_plan_customers, f"mask {c} not from mutable_ids vehicle"
```

⇒ **当前版本严格更宽松**：凡在 `decision_pool` 内、但不在任何 `mutable_ids` 车辆 plan suffix 中的
候选（典型即 **deferred 客户**），训练期会 `raise`，当前会**放行**。当前 docstring 亦自述这一放宽
（"含 deferred 客户：不在任何车 suffix 但可分配"）。

### 2.2 动态对拍（10 条事前声明的状态，覆盖 committed / non-replan 冻结 tail / deferred / protected tail / depot / 空池 / 乱序 visible）

| 比较项 | 结果 |
|---|---|
| `decision_pool_from_vehicles` **有序**输出 | **每一状态完全一致** |
| `mask_candidate_pool` **有序**输出（含/不含 protected） | **每一状态完全一致** |
| **同 seed 下的采样选择**（对有序池） | **每一状态完全一致** ⇒ **顺序也没有变** |
| `validate_mask_scope` | **不同**：训练期对"池内但不在任何 mutable plan suffix 的客户"报 `AssertionError`，当前放行 |

**精确边界**：差异**只**落在 mask 合法性的**执行强度**上；**掩码池的生成与顺序可证一致**。
因此"F 与 D 在同一评价阶段共用同一掩码池"这一必要条件**成立**，漂移本身**不推翻**同掩码比较。

⚠️ **未证明**（不得外推）：
* 状态套件是**合成且计划稀疏**的，它证明"该子句被删除"，**不度量**真实状态下这类客户出现的频率；
* 函数体相同只证明该函数的输入输出等价，不等于整模块等价（模块级常量与导入另行比较）；
* 训练期身份与实际部署身份**必须继续显式记录**，不得因行为等价而写成身份一致。

### 2.3 `cc_lns_replanner.py`（D 臂）— 7/8 函数逐字节相同

`_reconstruct` / `_select_one_mask` / `_detour` / `_plan_suffixes` / `_partition_ok` / `CCLNSReplanner` **相同**；
**`cc_lns_search` 不同** —— 当前版本新增了时间预算中断：

```python
+                if time.perf_counter() - t0 > budget_s:
+                    stats['stop_reason'] = 'budget'
+                    break
```

⇒ D 臂的**搜索轨迹**不能由训练期文件直接复现（少了这个中断），这一点须写进新规格。

## 3. 收束后仍然阻断的三件事（不在旧 D2 内解决）

1. **E2 需要新的训练证据**（相同架构/数据/优化器/更新日程/模型选择，仅改 backbone 这一个因素；
   且新增可训练层的初始状态也要可比，否则引入第二个混杂因素）。
2. **`P_after_install` 类字段需要新的前瞻性采集**（且必须由真实安装/事务回执证明，不得由 `P_certified` 推算）。
3. **D 语义须在前瞻规格里锁定**（本轮由用户裁定：D-Geo 为主对照，regret-2 独立保留，不采用缺失的 swap 变体）。

## 4. 本收束**不**做的事

未做 F−D / F−E 比较；未启动训练；未采集新数据；未访问最终测试种子；
未改写 `D2_QUALIFICATION_AUDIT.json` 与原 `not_qualified` 裁决；未修改任何冻结目录 / gate / checkpoint / 封存结果；
**未因找回 F 权重而修改任何历史负裁决**（含 `negative_result_paper/` 路线 C 的既有负证据）。

## 5. 关联产物

* 本轮解阻清查：`../a1_d2_asset_unblock_root_v1/D2_ASSET_INVENTORY.json`、`D2_FIELD_COVERAGE.json`、
  `D2_MASK_SCOPE_AUDIT.json`、`D2_UNBLOCK_VERDICT.json`、`D2_UNBLOCK_SUMMARY.md`、`raw/`
* 本轮等价审查：`D2B_EQUIVALENCE.json`、`run_d2b_equivalence.py`
* 原 D2 裁决（**未改动**）：`../a1_d2_same_mask_attribution_root_v1/D2_QUALIFICATION_AUDIT.json`
* 下阶段前瞻规格：`../../../delivery/a1_d3_geo_masked_attribution_root_v1/D3_PROSPECTIVE_PREREGISTRATION.md`
