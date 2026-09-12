# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 用途与状态

`dcc_rh_v4/` 是 **RRNCO-Ordering-RH-D** 的 strict-online 适配工作区：把 RRNCO（`ai4co/real-routing-nco`，ICLR 2026，上游 `../rrnco/`）接入公共 strict-online 协议。

> **当前判定（2026-09-12）**：`PROVISIONAL_GO_RRNCO_ORDERING_RH_D`，`evidence_level = offline-skeleton-executed; real-model-pending`。Stage A（离线骨架）+ Stage A.1（四项 P0 收尾）已完成并提交；**Stage B（真实 `epoch_199.ckpt` 验证）尚未开始，R1 三层身份冻结未授权**。

权威进度见 [`R0_ADAPTABILITY_REPORT.md`](R0_ADAPTABILITY_REPORT.md)（结论/证据等级/后续 Gate）。

## 边界（硬规则）

- **绝不修改** `../rrnco/`（上游源码）、`../../common/`（公共合同，已冻结，改动会同时破坏 PyVRP 与 OR-Tools 冻结身份）、`../../../MASKCO_code/`。
- 新增代码只放本目录；`../dcc_vrp/` 是 legacy one-shot 适配（不合规，仅诊断参考，见其 README）。
- 真实模型推理依赖服务器 `rrnco` conda env（torch≥2.7 + rl4co + GPU）；离线测试不导入 torch/rl4co/tensordict，用 `MockPreferenceProvider`。

## 架构（5 模块流水线）

外部方法实现公共 `ExternalReplanner.propose(view) -> PlanProposal`（契约见 `../../common/method_adapter.py`）。本目录的流水线：

```
DecisionView（公共契约快照，结构上不含未来客户）
  → subproblem.py     每辆 replan 车构造「自包含」可见子问题（depot+anchor+pool），
                       未来/已服务/受保护客户结构缺席；canonical_hash 供扰动测试
  → preference.py     偏好 = 解码序列删 depot/anchor 后客户首次出现顺序；
                       MockPreferenceProvider（edd/nearest/fixed/shuffle）+ validate_ordering
  → coordinator.py    确定性多车协调器：open-new-vehicle 优先 → 归一化排名 →
                       增量距离 → vehicle_id → customer_id；deferred（非 fallback）；
                       排序非法 → EDD fallback（带原因）
  → pickup_certificate.py  独立认证 TW/容量/服务后返仓（pickup-to-depot：服务后载荷增加、
                       仅尾 0 返仓；()=WAIT / (0,)=CLOSE）；校验 pool 成员（非仅 node_ids）
  → rrnco_guided_adapter.py  组合上述 → PlanProposal；solve_meta 记录子问题 hash/
                       排序/决策/deferred/fallback
```

## 关键不变量（Stage A.1 修复确立，改动前先读）

1. **provider 每车只调一次**——adapter 预计算 `orderings` 传给 coordinator；coordinator 不持有/调用 provider；`solve_meta` 记录同源 ordering。
2. **SubProblem 无 `view` 引用**——自包含数组，provider 无法访问 sub_nodes 之外数据（结构隔离，非仅 hash 过滤）。测试断言 `not hasattr(sp, 'view')`。
3. **禁止默认 EDD**——`RRNCOGuidedAdapter(None)` 抛 `ValueError`；mock 仅测试显式注入。
4. **open-new-vehicle 优先**——已启动车（有 suffix/非 depot anchor/load>0）续用，空载 depot 闲置车才新开；外层负责 fleet packing，故命名 Ordering 而非 guided。

## 常用命令

```bash
# 全部离线测试（用 cc_ortools env，纯 NumPy，不导入 torch）
cd CC_Compare/RRNCO/dcc_rh_v4
for t in tests/test_*.py r0_5_snapshots.py; do \
  "D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" "$t"; done

# 单测
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" tests/test_coordinator.py

# late-reveal 集成（真实 StrictOnlineEnv + BridgeReplanner，6/6 complete）
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" tests/test_late_reveal_integration.py
```

测试用 `testutil.py` 构造 duck-typed view/vehicle（节点 id == 索引，depot=0，dist=欧氏，travel=dist）。

## 冻结核验（改动后必跑，确认未污染基线）

```bash
"D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" CC_Compare/PyVRP/dcc_vrp/verify_freeze.py
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" CC_Compare/OR-Tools/dcc_vrp/verify_freeze.py
"D:/AA_Py/A_Anaconda/envs/cc_ortools/python.exe" CC_Compare/OR-Tools/dcc_vrp/verify_or7_freeze.py
```

## 后续（Stage B，等 DEV-GATE 正式退出后）

`rrnco_backend.py`（加载 `epoch_199.ckpt` + 注入 anchor/time/load + `PoolStartNodes`）→ 真实三快照 → 可变规模（1/2/5/10/25/50）→ 未来扰动 → 模型贡献对照 → 公共 evaluator 重放。全部通过后才进入 R1 三层身份冻结。
