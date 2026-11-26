# P5-V2 源码审计报告（SOURCE_AUDIT）

> 起始身份：commit `736414664773d83ef55007a541b8bfb6ca3aaae0`（分支 codex/o0cc-cal-phys，
> 与 origin 同 SHA；origin/main 落后一个提交 8c12570b）。
> 审计对象：冷链锁版 `maskco_perishable_external_20261119/reserved_validation/solver/`
> 与 P5-V1 `P5_standard_vrp/standard_vrp/`。
> 结论先行：**P5-V1 未完整保留锁版 ORDER_SPLIT 的搜索机制**（核实 6 项机制缺失、5 项缺陷、
> 3 项语义简化），因此 P5-V1 的标准任务结果不能用于判定冻结冷链神经网络的跨目标迁移能力。

## 1. 锁版机制清单（以源码为准）

`reserved_validation/solver/pool_search.py`（arm=ORDER_SPLIT → base=HYBRID）：

1. **掩码池**：`group = mask_rng.choice(arange(1,101), 40)`，每 epoch 重采；
2. **跨车 bundle（每 3 epoch 一次）**：`bundle(parent, dist, index)` 选 2–4 条相近路线
   （总客户 ≤24）；被选路线的全部客户并入掩码 `group`（hidden + 随机补齐到 40）；
3. **每 epoch 一次神经前向**：`provider.heatmap(task, state, group, "maskco_base", epochs)`，
   同 epoch 内所有重构与跨车共用该热图（`forwards` 计数）；
4. **距离条件残差校准**（CALIB/HYBRID）：`residual_edges(dist, heat)` →
   `heat = spatial + alpha * residual`，`spatial=-d/scale`（scale=各行第 8 小距离的中位数），
   `residual = clip(log(rank_D/rank_H), -3, 3)`，rank 从 1 开始，对角线 -inf，alpha 取自
   `calibrated.json`；
5. **六配置短时试选**（HYBRID）：3 种排序（距离 / 热图 / 校准热图）× 2 种算子（贪心 / regret），
   每配置 `min(allowance, .04s)`，按真实目标比较并采用最优配置，`ranking_uses` 计数；
6. **冲突池**（FOCUS/HYBRID 且 mode=1）：`conflict_pool(routes, group, heat)` = 现有邻接在热图
   中排名靠后的 12 个被遮客户 + 每人的最佳替代 1 个 + 组内前 8 个；
7. **自适应双模式**：`mode = argmin(uses)`（epochs<2 或 %4==0）否则
   `argmax(rewards + .1/sqrt(1+uses))`；`rewards[mode] = .65*r + .35*min(3, log1p(gain))`；
8. **跨车访问顺序搜索**：`proposals(parts, heat, seed, limit)`（随机路线排列 + 25% 反转 +
   1–4 次热图引导的段反转/客户搬移），逐个用 C++ Split DP 评价，`stop=min(deadline-.003,
   epoch_start+1.6, now+.30)`；
9. **工作解 / 历史最好解**：C++ `repair_batch` 内 SA 接受
   `new_u >= u-eps || (tolerance>0 && new_u > best_u-6*tolerance && rand < exp((new_u-u)/tolerance))`，
   tolerance=`2.5*max(.10, 1-elapsed/seconds)`；`record` 只接受严格更好；
10. **周期重置**：`epochs % 4 == 0 → state = best`；
11. **重插内核** `masked_repair.cpp::choose_slot`：
    - 候选 = 全部路线全部可行插位（容量） + **新建单客户路线**（始终在列）；
    - 贪心：距离序前 6 + 热图序再 6，共 ≤12 个候选逐一精确评价；不足则按距离序补扫；
    - Regret：热图序 top-8 + **最短可行插位保护**（按绕行增量升序扫描全部插位，取第一个
      可行者；若不在 top-8 则替换最后一个成员）；
    - 组内破坏规模：贪心 `k = 6 + rng()%12`，regret `k = 10`；
    - related 模式：以 anchor 按热图亲和 `heat[a,i]+heat[i,a]` 做 `r²` 采样。
12. **统计字段**：iterations / improvements / native_route_requests / forwards / forward_s /
    ranking_counts / regret_epochs / mask_mode_counts / split_blocks / split_orders /
    split_offers / timing。

## 2. P5-V1 差异与缺陷（已逐条核实源码行号）

| # | 位置 | 锁版语义 | V1 实现 | 判定 |
|---|---|---|---|---|
| D1 | `standard_pool_search.py` L90 | 客户顺序由 ranking 决定 | `remaining = list(removed)`，ranking 参数未消费 | **缺陷**（排序失效） |
| D2 | 同文件 L112 | VRPTW 也按 operator 执行 | `for cust in (list(removed))`，greedy/regret 共用贪心 | **缺陷**（算子失效） |
| D3 | L73–L87 | 每次插入后重算全部 pending 的后悔值 | `order_key` 在插入前一次性排序 | **缺陷**（静态） |
| D4 | L201–L206 | calib 用残差校准热图 | heatmap 与 calib 传入同一 `heat` | **缺陷**（校准缺失） |
| D5 | L58–L66 | 最短**可行**插位保护 | 追加几何绕行最小位置，不验证可行性 | **缺陷**（语义错误） |
| M1 | 全文件 | 冲突池 / 自适应双模式 / 奖励更新 | 无 | 机制缺失 |
| M2 | `split_search` | bundle 2–4 条相近路线且并入掩码（共享热图） | 随机 2 条路线、不并入掩码、独立预测 | 机制缺失 |
| M3 | `propose_order` | 随机排列+反转+1–4 次热图引导扰动 | 单一最近邻链 | 机制缺失 |
| M4 | epoch 循环 | SA 工作解 + 历史最优 + 每 4 epoch 重置 | 严格改善才更新、无重置 | 机制缺失 |
| M5 | 六配置 | 每配置 ≤0.04s 短试选后优选持续 | 6 组全跑（每轮全部执行），无时间片、无优选 | 机制简化 |
| M6 | 组破坏规模 | 贪心 6–17 / regret 10（组内） | 每轮掩码 40 全量重插 | 机制简化 |
| M7 | choose_slot | 贪心=6 距离+6 热图候选；regret=top8+保护 | 全位置精确扫描（更慢、无 top-k 语义） | 语义差异（效率与行为） |
| S1 | 内核 | C++ `masked_repair` / `split_order` | Python 全位置扫描 | 性能差异（60s 内重构次数差 1–2 个数量级） |
| S2 | 输入 | 16 通道训练语义 | 中性编码 + 需求/20000（容量 200 时仅 0.01） | 待第四阶段核查 |

## 3. 缺陷复现（audit/v1_defect_reproduction.json）

- **D1**：对同一 removed 集合使用升序与降序 ranking，`masked_repair` 输出路线完全相同 →
  排序未生效（`ranking_asc_equals_desc: true`）。
- **D2**：C101 上 greedy 与 regret2 输出完全相同 → VRPTW 算子未生效。
- **D3**：`_insert_delta` 调用数 regret2=1627、greedy=781。k=12、约 65 插位/客户时，
  静态实现 ≈ 2×12×65≈1560，动态实现应 ≈ 12×13/2×65≈5070 → 实测为静态量级。
- **D4**：源码 L201-206 两种 ranking 名共用同一 heat；`same_heat_same_routes: true`。
- **D5**：源码 L58-66 只取几何 detour 最小位置，未做容量/时窗可行性检查。

## 4. 对既有结论的影响

P5-V1 的"神经热图在标准任务无正贡献"结论**不能**归因于冻结网络本身：V1 中
(a) 排序与算子未生效使候选生成退化为单一贪心全扫描；(b) 校准缺失使 calib 与 heatmap 等价；
(c) 无 SA/重置/冲突池/自适应使搜索机制整体退化；(d) Python 内核使 60s 内重构次数远低于
锁版水平。上述实现缺陷与迁移能力混杂，必须按 P5-V2 五版本受控实验分离。

## 5. 审计产出

- `SOURCE_AUDIT.md`（本文件）
- `audit/V2_原始机制与标准适配差异矩阵.csv`
- `audit/v1_defect_reproduction.json`（缺陷复现原始记录）
- `manifests/frozen_identity.json`（起始 commit 与锁版/V1 源码 SHA256）
