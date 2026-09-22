# PyVRP-RH-D — strict-online rolling horizon 距离优化基线（B2，已冻结）

> 定义：在 strict-online rolling horizon 下以 **travel distance** 为求解目标的
> 传统强基线，最终由统一 C0 evaluator 回放并报告 D/Q/E/J_CC。
> **非冷链感知**（不修改 PyVRP cost、不注入 quality/energy 代理）；
> 冷链主表标签为 distance-optimized / non-CC-aware baseline。
> 真正的 PyVRP-RH-CC 若后续需要，单独立项。

## 正式冻结配置（protocol revision 1，2026-09-09）

机器可读记录见 [`FROZEN_CONFIG.json`](FROZEN_CONFIG.json)（绑定 9×32 证据产物
hash 与 canonical.COMPLETE）。身份链（单向绑定，无循环）：
`SOURCE_MANIFEST.json`（identity rev4：源码/依赖/环境资产 + expected_code_hash）
← `FROZEN_CONFIG.json`（绑定 SOURCE_MANIFEST + 行为配置 + 证据）←
[`FREEZE_SEAL.json`](FREEZE_SEAL.json)（最外层绑定两者）。任何配置项变化必须
升级 protocol/adapter revision，不得覆盖当前冻结版本。

```text
method_name         = PyVRP-RH-D
PyVRP               = 0.14.0
MaxIterations       = 300
solver_seed         = 0
warm_start          = false
fallback            = false
objective           = distance（PyVRP 内部；最终 C0 D/Q/E/J_CC 重放）
INT_SCALE           = 1000
optional clients    = dynamic coverage prize（按子问题界限计算）
feasibility         = fixed dynamically bounded penalty（min==max，溢出保护）
decision policy     = strict-online WAIT/CLOSE（9×32 已验证实现）
```

**预算选择依据**（修正表述，不使用「收敛」措辞）：

> 300 iterations 在 DEV-PROTO 上达到稳定可行区间；增加至 1000/3000 未形成一致
> 距离优势，因此选择 300 作为可复现的固定计算预算。rolling-horizon 是连续决策
> 组合，单个事件求解更充分不保证终局距离单调下降。300/1000/3000 的差异作为
> 预算敏感度报告，不解释成冷链方法优劣。

证据：`results/devproto_9x32_iter300/`（288/288 complete、0 违规、0 协议错误、
0 penalty warning、fallback 0、9/9 cell 确定性抽查一致）；
`results/devproto_9x2_iter{300,1000}/`（两档 15/18 decision_hash 一致，3 例
路线差异已入敏感度记录）。

## 文件

| 文件 | 作用 |
|------|------|
| `identity.py` | 环境身份强制检查（pyvrp==0.14.0 + site-packages 导入路径 + 二进制 hash + wheel sha256 + pip freeze） |
| `problem_builder.py` | DecisionView → PyVRP Model（整数化 + 界限计算 coverage prize / feasibility penalty） |
| `route_mapper.py` | Solution → PlanProposal（vt 索引身份映射 + 浮点 certificate 复核 + 防御性拒绝） |
| `pyvrp_adapter.py` | `PyVRPRHDAdapter`（propose：空 pool 快速路径 / solve / 映射） |
| `run_dcc.py` | 评估入口（common run_batch 薄封装） |
| `tests/` | 7 组回归（见下） |
| `assets/pyvrp-0.14.0-cp312-cp312-win_amd64.whl` | 冻结 wheel（sha256 见 identity.WHEEL_SHA256） |

## 环境（统一正式版本，两端一致）

- **pyvrp==0.14.0**（2026-08-20 正式发布；禁用 1.0.0a0 main 源码——`../../PyVRP/`
  上游快照仅 reference_only）
- 独立 `cc_pyvrp` conda env（Python 3.12；**不装进 MASKCO_env**）
- 服务器：单独 Python ≥3.11 环境 + pyvrp 0.14.0（无 wheel 则从官方 sdist 构建）
- adapter 启动强制：版本断言 + 导入路径必须是独立环境 site-packages
- 身份落盘：`run_dcc.py` 输出 `pyvrp_environment.json`

## 建模（冻结规则）

1. **子问题**：每决策点只含可变池（visible unserved − 非 replan 车辆 committed_next
   − 非 replan 车辆完整 frozen tail）；每 replan 车独立 vehicle type
   （start_depot=当前位置+ceil(ready_time)，end_depot=真实 depot+floor(depot_tw_end)）；
2. **整数化**：距离 round×1000；travel/service/tw_start ceil；tw_end/容量 floor；
   demand 整数不缩放。输出路线必须经原始浮点 `certify_route` 复核，
   整数可行但浮点不可行 → 拒绝；
3. **容量**：`effective_capacity = capacity − current_load`（不用 initial_load，
   避免 start-depot 卸载语义）；当前舱内货物由 C0 跟踪；
4. **覆盖-可行性尺度**（按子问题界限计算，非常数）：
   `coverage_prize = (n_pool + n_vehicles) × max_edge + 1`；
   `feasibility_penalty = n_pool × coverage_prize + distance_upper_bound + 1`；
   SolveParams penalty min==max（不靠自适应增长）；溢出保护 MAX_VALUE//4；
   尺度落盘事件记录 `solve_meta`；
5. **deferral 语义**：覆盖不了的客户留给后续事件（公共 ownership audit 记录
   deferred）；`require_complete=True` 为严格观察模式（required 客户，缺覆盖 →
   SolveError 显式失败）；
6. **空 pool 快速路径**（不调用 PyVRP）：无 future reveal → CLOSE；
   有 future reveal 且现在返仓可行 → WAIT；返仓已不可行 → CLOSE；
7. **确定性协议**：固定 `--pyvrp-seed` + `--max-iterations`，无 warm start；
   同 seed 两次运行 decision_hash 一致。固定 MaxRuntime 效率实验另立协议；
8. **失败语义**：无完整解 / infeasible / 跨车重复 / 浮点 certificate 失败 →
   显式失败并记录原因（不静默 fallback）。后续若立项
   'PyVRP-RH-D + JF1-H-F fallback'，必须单独命名并分别报告原生服务率与
   fallback 率。

## 运行

```bash
# 本地（cc_pyvrp env）
"D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" dcc_vrp/run_dcc.py \
    --data <dcc npz> --capacity 50 --num_vehicles 25 --objective coldchain \
    --objective-profile <frozen profile.json> \
    --instance-ids 0,1 --max-iterations 500 --pyvrp-seed 0 --out <dir>
```

## 测试（14 步验收的本地部分）

```bash
cd dcc_vrp/tests
python test_environment_identity.py   # 1  环境身份 + 导入路径
python test_problem_builder.py        # 2-6 单车辆/多车映射/cargo/非 depot anchor/空路线/vt 乱序
python test_integer_boundary.py       # 10 整数化边界 + 保守方向扫描
python test_adapter_integration.py    # 8,9,11,12 future 扰动/frozen/确定性/late reveal
python test_empty_pool.py             # 3  空 pool 快速路径 + WAIT/CLOSE + runner 集成
python test_penalty_protocol.py       # 4,5 容量不足可行子集 + 旧固定 prize 反例形状
python test_event4_regression.py      # 3  真实 R1 事件 4 ownership 修复证据
```

## 状态

```text
B1 公共 baseline contract：PASS
B1.1 ownership/pool 加固：PASS
B2 PyVRP-RH-D 单场景实现：PASS
B2 PyVRP-RH-D 跨场景行为 Gate：PASS（DEV-PROTO 9×32，288/288）
B2 PyVRP-RH-D 跨场景证据 Gate：PASS（完整 record 落盘 + 盘上复验 + canonical.COMPLETE）
PyVRP 正式冻结：已完成（protocol revision 1，本目录）
服务器环境与跨平台 parity：待 DEV-GATE 完成后执行
正式大样本比较：尚未开始
```
