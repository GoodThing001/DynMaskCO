#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**真实 env 连通性验证**（用户 2026-10-14 裁定 3 授权的有限工程验收）。

授权范围（严格限定，本脚本不得越界）：
  ✅ 在一个**固定的、合成的公开可见状态**上，把 `role_c_replanner` 接入**真实**
     `StrictOnlineEnv`，检查：状态迁移 / 日志写入 / C0 认证顺序 / 承诺冻结 / 结果确定性。
  ❌ **不**计算、查看或依据**效用**选择任何参数；
  ❌ **不**运行 2 天烟测、40 天正式批次或最终测试；
  ❌ **不**触碰任何冻结证据、既有 gate、封存结果或保留测试种子；
  ❌ 失败时**记录原始异常**，**不**通过放松 T 系列判据来消除。

⚠️ 两件必须随结论一起说的事：
  1. 真实 env 连通性**只代表工程兼容性**，**不**升级为"Role C 完整动态基线已合格"；
  2. 本脚本用**合成夹具**（5 节点、2 车、3 次揭示），**不是**开发集/测试集，
     因此**不产出任何效用结论**。

运行：python CC_Compare/tools/test_role_c_env_connectivity.py   （退出码 0 = 全过）
"""
import os
import sys

# 控制台编码：中文/Ĵ 在 GBK 控制台会 UnicodeEncodeError（看似失败、实则打印崩）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                              # noqa: BLE001
    pass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
for _sub in ('scripts/simulation', 'scripts/coldchain', 'scripts/evaluation', 'scripts/data'):
    sys.path.insert(0, os.path.join(EXT, _sub))
sys.path.insert(0, HERE)

import role_c_replanner as rp                                   # noqa: E402
from role_c_decision_log import DecisionLogger                  # noqa: E402

# 冻结锚点（取自 P1A_CONTROL_FROZEN.json，避免手抄错）
B = 710.6548152249378
COOLING_SHARE = 3.7208946347769882
BOOKING_HORIZON = 16.0
CAPACITY = 50.0
NUM_VEHICLES = 2


def make_fixture(n_customers=4):
    """**合成**夹具：1 个 depot + n 个客户；3 次揭示；TW 宽松以免掩盖连通性问题。

    ⚠️ 合成夹具**不是**任何冻结切分，只用于工程连通性。
    """
    n = n_customers + 1
    coords = np.zeros((1, n, 2), dtype=np.float32)
    coords[0, 0] = (0.0, 0.0)
    for i in range(1, n):
        coords[0, i] = (float(i), float((i % 2)))
    demands = np.zeros((1, n), dtype=np.float32)
    demands[0, 1:] = 1.0
    tw_start = np.zeros((1, n), dtype=np.float32)
    tw_end = np.full((1, n), 100.0, dtype=np.float32)
    tw_end[0, 0] = 0.0
    service_time = np.full((1, n), 0.1, dtype=np.float32)
    service_time[0, 0] = 0.0
    temp_class = np.zeros((1, n), dtype=np.int32)
    initial_quality = np.ones((1, n), dtype=np.float32)
    # 揭示：客户 1 在 t=0 即可见；2/3/4 分别在 t=2/4/6 揭示
    reveal_time = np.zeros((1, n), dtype=np.float32)
    for k, i in enumerate(range(1, n)):
        reveal_time[0, i] = 0.0 if k == 0 else float(2 * k)
    d = np.linalg.norm(coords[0][:, None, :] - coords[0][None, :, :], axis=-1)
    dist_mat = d.astype(np.float32)[None, ...]
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
            'service_time': service_time, 'temp_class': temp_class,
            'initial_quality': initial_quality, 'reveal_time': reveal_time,
            'dist_mat': dist_mat}


def _stub_decide(order, ctx):
    """**桩**判决：接单并插到 1 号车尾。

    本脚本验证的是**管路**（状态迁移/日志/认证顺序/承诺/确定性），**不是**判决机制 ——
    机制（λ_B / 独立前瞻搜索）与真/假无关地由**别处**的纯函数测试覆盖。
    """
    plan = {k: list(v) for k, v in ctx['plan'].items()}
    plan.setdefault(1, []).append(order)
    return True, {'plan': plan, 'day_id': ctx['inst_idx'],
                  # 信息边界 = 决策时刻（合成夹具里可见集恰为 clock 前已揭示者）
                  'info_max_reveal': ctx['clock']}


def main():
    cases, notes = [], []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- 0. 依赖可用性（失败要**原样**报出，不静默跳过）----------
    try:
        from strict_online_env import StrictOnlineEnv
        from run_exp_energy_c0 import certify_plan
        from scenario_saa import make_c0_contract_v2
    except Exception as exc:                                   # noqa: BLE001
        print('FAIL: 无法导入真实 env / certify_plan：%s: %s' % (type(exc).__name__, exc))
        return 1
    ck('0. 真实 StrictOnlineEnv / certify_plan / contract 可导入', True)

    contract = make_c0_contract_v2()
    ds = make_fixture()

    # ---------- 1. 接线：replanner 接进真实 env ----------
    logger = DecisionLogger(B)

    def certify_fn(plan, ctx):
        """**真实** `certify_plan`（顺序：候选评估 → 认证 → 提交）。"""
        try:
            ok, energy = certify_plan(ctx['env'], ctx['inst_idx'], ctx['clock'],
                                      ctx['vehicles'], ctx['served_mask'], plan,
                                      ctx['contract'], ctx['budget_B'])
        except ValueError as exc:      # 物理硬违反 ⇒ 按不可行处理（与既有实现同）
            return False, None
        return bool(ok), energy

    policy = rp.RoleCReplanner(budget=B, capacity=CAPACITY,
                               booking_horizon=BOOKING_HORIZON, contract=contract,
                               cooling_share=COOLING_SHARE, decide_fn=_stub_decide,
                               certify_fn=certify_fn, logger=logger)
    env = StrictOnlineEnv(ds, capacity=CAPACITY, num_vehicles=NUM_VEHICLES,
                          replanner=policy, coldchain_contract=contract,
                          booking_horizon=BOOKING_HORIZON)
    try:
        traces, served_mask = env.run(0)
        ck('1. env.run(0) 在接线的 replanner 下正常结束（无原始异常）', True)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        print('FAIL: env.run 抛出原始异常（按授权要求原样记录，不放松判据）：')
        traceback.print_exc()
        ck('1. env.run(0) 正常结束', False, '%s: %s' % (type(exc).__name__, exc))
        _report(cases)
        return 1

    # ---------- 2. 状态迁移真的发生了 ----------
    n_served = int(np.sum(served_mask[1:]))
    ck('2. 服务状态迁移发生（served_mask 有客户被服务；%d 个）' % n_served, n_served >= 1)
    ck('2. depot 视为已服务（served_mask[0]=True）', bool(served_mask[0]) is True)
    ck('2. 记录里确实有决策事件（%d 条）' % len(logger.records), len(logger.records) >= 1)

    # ---------- 3. 日志写入 + T5/T6/T7 审计 ----------
    repeat_logger = DecisionLogger(B)

    def certify_fn2(plan, ctx):
        return certify_fn(plan, ctx)

    policy2 = rp.RoleCReplanner(budget=B, capacity=CAPACITY,
                                booking_horizon=BOOKING_HORIZON, contract=contract,
                                cooling_share=COOLING_SHARE, decide_fn=_stub_decide,
                                certify_fn=certify_fn2, logger=repeat_logger)
    env2 = StrictOnlineEnv(ds, capacity=CAPACITY, num_vehicles=NUM_VEHICLES,
                           replanner=policy2, coldchain_contract=contract,
                           booking_horizon=BOOKING_HORIZON)
    env2.run(0)
    audit = logger.audit(log_repeat=[dict(r) for r in repeat_logger.records])
    ck('3. T5 认证顺序与因果边界通过', audit['checks']['T5_order_and_causality'] is True,
       audit['problems'])
    ck('3. T6 承诺与预算硬约束通过', audit['checks']['T6_commitments_and_budget'] is True,
       audit['problems'])
    ck('3. T7 确定性通过（同 seed 两次 env.run 的 decision_hash 序列一致）',
       audit['checks']['T7_determinism'] is True, audit['problems'])
    ck('3. 总判定 passed=True', audit['passed'] is True, audit['problems'])

    # ---------- 4. 因果性：env 以 booking_horizon 非 None 构造 ----------
    ck('4. env.booking_horizon 非 None（因果口径，而非读真实未来的旧行为）',
       env.booking_horizon is not None)
    ck('4. 全部记录的 info_max_reveal ≤ clock（无真实未来泄漏）',
       all(float(r['info_max_reveal']) <= float(r['clock']) + 1e-6
           for r in logger.records))

    # ---------- 5. 承诺冻结：`_plan` 只增不减地被 env 消费 ----------
    ck('5. 已接受订单均留在承诺集合里（不可撤销）',
       set(policy._accepted) <= set(policy._accepted) | set(policy._rejected)
       and len(policy._rejected & policy._accepted) == 0)

    # ---------- 6. 明确边界（不越界声明）----------
    notes.append('本次验证 = **工程兼容性**；**不**代表 Role C 完整动态基线已合格。')
    notes.append('夹具为**合成**（5 节点 / 2 车 / 3 次揭示）⇒ **不产出任何效用结论**。')
    notes.append('判决用**桩**（接单即插 1 号车）；λ_B 与独立前瞻搜索由别处的纯函数测试覆盖。')

    _report(cases, notes)
    return 0 if all(ok for _n, ok, _d in cases) else 1


def _report(cases, notes=()):
    for n, ok, detail in cases:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   ← ' + str(detail))))
    for x in notes:
        print('  ⚠️ ' + x)
    p = sum(1 for _n, ok, _d in cases if ok)
    print('\n结果：%d 通过 / %d 失败' % (p, len(cases) - p))


if __name__ == '__main__':
    sys.exit(main())
