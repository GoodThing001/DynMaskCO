# OR-Tools-RH-D — strict-online rolling horizon 距离优化基线（已冻结）

> 定义：在 strict-online rolling horizon 下以 travel distance 为求解目标的
> 传统强基线（OR-Tools 求解器），最终由统一 C0 evaluator 重放 D/Q/E/J_CC。
> 非冷链感知；drop 只是 DEFER 不是 service success。
> **状态：已冻结（protocol revision 1 + identity manifest revision 1）。**
> OR6 行为 Gate PASS、OR6.1/OR6.2 证据控制链闭合；OR7 预算扫描（solution_limit=
> {10,30,100}）三档全通过协议 Gate，预注册选择器选定 **solution_limit=30**
> （九 cell 等权 pure distance 17.3956）。冻结三件套：`SOURCE_MANIFEST.json` ←
> `FROZEN_CONFIG.json` ← `FREEZE_SEAL.json`（核验 `verify_freeze.py`）。

## 文件

| 文件 | 作用 |
|------|------|
| `identity.py` | 环境身份（ortools==9.11.4210 + site-packages 路径 + wheel sha256） |
| `problem_builder.py` | DecisionView → RoutingModel（virtual start / 保守整数化 / drop penalty / dummy） |
| `route_mapper.py` | Solution → PlanProposal（IndexToNode 链遍历 + dummy 过滤 + 浮点 certificate + WAIT/CLOSE） |
| `ortools_adapter.py` | `ORToolsRHDAdapter`（空 pool 快速路径 / solve / 状态映射 / 失败记录） |
| `run_dcc.py` | 评估入口 |
| `tests/` | 5 组测试（builder / unit / hardening / integration / devproto_control 27 项） |

## 环境（OR0，候选状态）

- **ortools==9.11.4210**（独立 `cc_ortools` env，py3.12；不装进 MASKCO_env / cc_pyvrp）
- python 路径：`D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe`
- wheel sha256：见 `identity.WHEEL_SHA256`（`assets/ortools-9.11.4210-cp312-cp312-win_amd64.whl`）
- Linux wheel 可用性待服务器阶段确认；无 wheel 则从官方 sdist 构建

## 建模要点（OR1-OR4）

- **节点布局**：0=真实 depot（end）；1..V=virtual start（每 replan 车一个，位置=anchor、
  时间钉 ceil(ready_time)）；V+1..V+C=客户；V+C+1..V+C+V=**每车一个 dummy 终端**
  （depot 坐标/0 需求/0 服务/全时段 TW——保证「所有客户均不可服务」时 first-solution
  仍非空，并钉死空路线 CLOSE/WAIT 的目标语义，映射时过滤）
- **车辆数严格 = len(replan_ids)**，不补 parked 车；starts/ends 经
  RoutingIndexManager(n, V, starts, ends)
- **容量**：fix_start_cumul_to_zero=False，start cumul 为自由变量，再用
  `capacity_dim.CumulVar(Start(v)).SetValue(ceil(current_load))` 钉死
  「start capacity cumul = current_load」；virtual start demand=0，后续 pickup
  在其上累加，绝不可能在 depot 前释放当前货物
- **时间**：transit = ceil(travel) + ceil(service_from)；TW ceil/floor；返仓截止由
  dimension horizon=floor(depot_tw_end) 强制
- **drop penalty** = distance_upper_bound + 1（service-first：少 drop 一个优先于任意
  距离改善）；int64 溢出检查；planned ∪ dropped 精确分区
- **确定性**：固定 solution_limit + PATH_CHEAPEST_ARC + 固定策略 + 单线程；
  wall-clock time_limit 仅安全上限。9.11.4210 无 random_seed 字段——确定性由
  test_ortools_integration 3 次重复实测
- **失败语义**：NO_SOLUTION/TIME_LIMIT → 安全 WAIT/CLOSE + `fallback_triggered=True`
  + `solver_status` 记录（不伪装空解）；正式 Gate 要求 fallback=0

## 本版本已实测的 OR-Tools 9.11.4210（win wheel）坑（全部绕开并有测试覆盖）

1. **状态码 7 = ROUTING_OPTIMAL（成功）**，不是失败（pyi 枚举确认）；
2. **end depot 节点 cumul 访问（SetRange/Min/Max）原生 segfault**——end 到达由
   dimension horizon 强制，绝不访问 end cumul；
3. **RegisterUnaryTransitCallback + AddDimensionWithVehicleCapacity** 返回错误状态
   ——容量维度必须用普通 (i,j) transit callback；
4. **NodeToIndex(end depot) = -1**（end 哨兵）——所有回调必须经 IndexToNode 把
   manager 索引转节点号再取矩阵；end 索引映射回 depot 节点，返仓弧自动正确；
5. **所有客户均不可服务时 first-solution 构造失败**——dummy 节点兜底。

## 测试

```bash
cd dcc_vrp/tests
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" test_problem_builder.py     # virtual start/容量/非连续 ID
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" test_ortools_unit.py        # mapper/边界/可选客户/空 pool/失败记录
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" test_ortools_hardening.py   # 时间极限/结构化失败/环境身份
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" test_ortools_integration.py # complete/trace/prefix parity/确定性
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" test_devproto_control.py    # 27 项控制链（P0-1..P0-4 + P1 + 负例）
```

## 下一步

OR7 预算扫描：在 DEV-PROTO 9×2 上做 solution_limit 敏感性（只按 hard/service 稳定、
距离、可复现性、时间预算选择，不用 J_CC）→ 9×32 协议稳定性 → protocol/identity/
源码包正式冻结。
