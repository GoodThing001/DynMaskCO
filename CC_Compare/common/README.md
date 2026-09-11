# CC_Compare/common — 公共 baseline contract（B1 + B1.1 加固）

所有外部对比方法（PyVRP / RRNCO / CaDA / RouteFinder / CO-enriched ML …）共用一套
strict-online 执行与记录合同。外部方法只实现 `ExternalReplanner.propose(view)`；
事件推进、commit、服务、返仓、cold-chain trace 与终局 D/Q/E/J 全部由项目
`StrictOnlineEnv` + C0 evaluator 完成，**不信任任何外部方法自报值**。

## 模块

| 文件 | 作用 |
|------|------|
| `_bootstrap.py` | 路径引导（项目 scripts 各子目录入 sys.path + `validate_layout()`） |
| `baseline_contract.py` | 实例记录 schema（identity/events/actions/trace/outcome/audit）+ hash 工具 |
| `record_validation.py` | 记录校验：缺字段/NaN/Inf/类型/交叉一致性（单调 clock、input 子集、分层 action、certificate）→ 拒绝 |
| `method_adapter.py` | `DecisionView` + `PlanProposal` + `ExternalReplanner`（propose）+ `BridgeReplanner` 契约强制 |
| `ownership_audit.py` | 公共所有权审计（逐事件 + 终局），所有 baseline 共用 |
| `strict_online_runner.py` | `run_instance` / `run_batch`：RecordingEnv + 事件/动作装配 + 审计接线 + 落盘 |
| `trace_export.py` | 轨迹导出 + `trace_replay_check`（导出数字 vs 权威 evaluator）+ 静态路线导出 |
| `tests/` | 6 类 B1 Gate 测试（纯 NumPy，不依赖任何外部方法包） |

## 外部 adapter 接口（B1.1 P0-1：proposal 模式）

```python
from method_adapter import ExternalReplanner, PlanProposal

class MyAdapter(ExternalReplanner):
    method_name = 'pyvrp-rh'
    method_revision = '1'
    adapter_revision = '1'
    checkpoint_hash = '<sha256 of loaded weights/solver binary>'   # 加载时计算

    def propose(self, view: DecisionView) -> PlanProposal:
        # view 只含公开节点（depot + 车辆 anchor + 可见未服务客户）的特征、
        # 车辆公开快照、公开节点间距离/时间矩阵、replan_ids、has_future_reveal。
        # 未来客户的数量/身份/坐标/TW/需求在结构上不可达（node_index 越界即拒绝）。
        ...
        return PlanProposal(
            suffixes={vid: (c1, ..., 0) for vid in view.replan_ids},  # 键必须==replan_ids
            model_input_customers=tuple(sorted(喂给模型的客户)),
            fallback_triggered=False,
            model_runtime_s=0.0,     # 用 time_model() 计时
        )
```

**结构保证**：adapter 不接收 env/dataset/VehicleState 对象；proposal 是纯数据；
Bridge 校验（键集合 == replan_ids、客户 ∈ 可变池、无同车/跨车重复）后写回并重建
plan 核对 proposal。防未来**读取**是结构保证（测试 2 的 perturbation 为双保险）。

## 审计与 repair 归属（B1.1 P0-3）

- `ownership_audit.py` 逐事件检查：duplicate_suffix / duplicate_committed /
  committed_and_suffix / served_in_plan / future_in_plan / invalid_customer；
  deferred（可见未服务不在计划）合法但必须记录；终局 `terminal_unresolved`。
- outcome 字段由公共 runner 审计所得**真实整数**：
  `ownership_violations` / `terminal_unresolved` / `repair_applicable=False` /
  `audit_source='common_runner'`。**禁止 setdefault(0) 掩盖错误**；任何违规 =
  PROTOCOL_ERROR（记录被拒绝）。接入统一 fallback 层（B2+）后 repair_applicable
  切换语义。
- 审计违规类型计数与逐事件明细全部落盘于 `record['audit']`。

## 记录 schema（B1.1）

- `identity`：method/revision/checkpoint/code/data/profile hash、instance/seed、
  objective、runtime_budget、random_seed。
- `code_hash` 绑定完整计算链：`common/` 核心模块 + adapter 源码 +
  项目计算模块（strict_online_env / action_contract / recourse_snapshot /
  counterfactual_teacher / authoritative_evaluator / hard_gate /
  coldchain_evaluator / coldchain_contract / coldchain_state）。全长 sha256。
- `events`：event_id（单调唯一）、clock（非递减）、visible/served 集合与 hash、
  state_hash、replan ids（存在且唯一）、每车状态与 suffix_before、
  plan_hash_before/after、model_input_customers（⊆ visible、不 ∩ served、唯一）、
  model_runtime_s、budget_exceeded、fallback。
- `actions`（分层，禁止混合统计）：
  - `action_layer='execution'`：COMMIT/CLOSE/WAIT（env 实际执行，每决策点每车一条）；
  - `action_layer='plan_diff'`：KEEP/DEFER/INSERT/NEW_ROUTE（fleet plan diff 的
    审计性解释，每个受影响客户一条）；INSERT/NEW_ROUTE 必须带
    `certificate{feasible=True, certificate_scope='final_vehicle_suffix'}`；
    writeback_ok = 最终 plan 与 proposal 一致（DEFER 亦为 True）。
- `execution_trace` / `outcome` / `hard_vector` / `audit`：权威 evaluator 重算，
  hard_vector 与 outcome 派生一致，audit 与 outcome 一致。
- 双 hash：`decision_hash`（决策确定性，忽略身份与计时）/
  `artifact_hash`（复现身份，绑定完整身份 + 决策内容，忽略计时）。

## 运行

```python
from strict_online_runner import run_instance, run_batch, load_objective_profile
from coldchain_contract import default_pilot_profile

rec = run_instance(dataset, capacity=50, num_vehicles=25,
                   adapter_factory=lambda: MyAdapter(...), inst_idx=0,
                   objective='coldchain',              # 或 'distance'
                   profile=load_objective_profile('results/.../objective_profile.json'),
                   seed=0, data_sha256=..., instance_seed=...)
```

- `objective='coldchain'` 时必须显式 profile（拒绝 INVALIDATED v1）；协议测试可用
  `default_pilot_profile()`，正式校准/判定必须用冻结 profile。
- `runtime_budget` 目前只记录（`budget_exceeded` 逐事件标记）；B2 起执行外部
  wall-clock Gate + 超时 fallback 策略（冻结后启用）。
- B1 阶段串行运行；记录校验失败 = 抛错（no silent success）。

## 测试（B1.1）

```bash
cd CC_Compare/common/tests
python test_baseline_contract.py      # 1 schema + 14 类损坏/交叉校验拒绝
python test_no_future_leakage.py      # 2 结构隔离 + future 内容/身份/数量扰动 parity
python test_frozen_prefix.py          # 3 proposal 违规（跨车/同车重复、非 replan 键、
                                      #   缺键、未来客户、非法编号）全部拒绝 + 正例
python test_complete_service.py       # 4 每客户恰好服务一次（含 late reveal）
python test_trace_replay.py           # 5 轨迹导出对账 + 静态回放一致性
python test_adapter_determinism.py    # 6 固定 seed decision_hash 一致 + 内容敏感
```

全部通过 = B1.1 Gate 通过，可进入 B2（PyVRP-RH）。
