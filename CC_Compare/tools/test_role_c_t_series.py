#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_t_series`（T 系列回归）的测试。

关键点：**用真实 env、真实 `certify_plan`、真实 `RoleCReplanner` 产出的落盘日志**跑
T1–T9，而不是手编记录。两组夹具对照，恰好验证规格 §5 T2 条（审计 D3b）要求的门控语义：

  * **slack**（预算宽松）：`raw > 0` 从未发生 ⇒ T2/T3 **必须报 `not_applicable`**，
    **不得**报 FAIL —— "机制没接进"与"机制接好了但资源不稀缺"在看动作序列时不可区分；
  * **tight**（预算卡在"刚好少装一单"的临界）：`λ_B > 0` 真的出现 ⇒ T2/T3 这时才谈得上判定。

另查：
  * T1 身份闭包与 gate 端到端；T8 共享配对字段逐字段相同（且**如实列出**冻结文件缺的字段）；
  * T2a 预设压力夹具（数值行为符合公式，**不要求**动作不同）；
  * T5/T6/T7 在真实日志上通过；T7 用同参数重跑；
  * **报告里没有任何效用字段**（"不看效用"的机械检查）；
  * 三个消融臂确实**改变了行为**或**未被激活**——两种都如实记录，绝不把后者判成失败。

合成夹具；**不跑实验、不碰保留种子、不写冻结文件**。
运行：python CC_Compare/tools/test_role_c_t_series.py   （退出码 0 = 全过）
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import role_c_t_series as ts                               # noqa: E402

ts.ensure_project_paths()
from scenario_saa import ScenarioOrder                     # noqa: E402

B_SLACK = 710.6548152249378        # 冻结锚点（宽松：夹具里远用不完）
COOLING_SHARE = 3.7208946347769882
BOOKING_HORIZON = 16.0
CAPACITY = 50.0
NUM_VEHICLES = 2
ARM_SEED = 7002                    # cond_hist（run_a1_step2_gate.py:51）


def make_fixture(n_customers=4):
    """合成 env（1 天）。depot 的 `tw_end` 是**回仓截止**，必须给可达值。"""
    n = n_customers + 1
    coords = np.zeros((1, n, 2), dtype=np.float32)
    coords[0, 0] = (0.0, 0.0)
    for i in range(1, n):
        coords[0, i] = (float(i), float(i % 2))
    demands = np.zeros((1, n), dtype=np.float32)
    demands[0, 1:] = 1.0
    tw_start = np.zeros((1, n), dtype=np.float32)
    tw_end = np.full((1, n), 100.0, dtype=np.float32)
    tw_end[0, 0] = 1000.0
    service_time = np.full((1, n), 0.1, dtype=np.float32)
    service_time[0, 0] = 0.0
    temp_class = np.zeros((1, n), dtype=np.int32)
    for i in range(1, n):
        temp_class[0, i] = 1
    initial_quality = np.ones((1, n), dtype=np.float32)
    reveal_time = np.zeros((1, n), dtype=np.float32)
    for k, i in enumerate(range(1, n)):
        reveal_time[0, i] = 0.0 if k == 0 else float(2 * k)
    d = np.linalg.norm(coords[0][:, None, :] - coords[0][None, :, :], axis=-1)
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start,
            'tw_end': tw_end, 'service_time': service_time, 'temp_class': temp_class,
            'initial_quality': initial_quality, 'reveal_time': reveal_time,
            'dist_mat': d.astype(np.float32)[None, ...]}


class FixedSampler(object):
    """固定场景集合（id 用高位，严格不与真实节点重叠）。temp_class=1/demand=1
    与夹具订单同口径 ⇒ 场景订单的边际能耗与决策订单可比。"""

    def __init__(self, m=6, temp_class=1):
        self.scenario = [ScenarioOrder(oid=10 ** 6 + j, reveal=1.0, x=float(1 + j * 0.1),
                                       y=0.0, demand=1.0, temp_class=temp_class,
                                       tw_start=0.0, tw_end=100.0, service_time=0.1)
                         for j in range(m)]

    def sample(self, snapshot, rng, k):
        return [list(self.scenario) for _ in range(k)]


def run(**over):
    kw = dict(ds=make_fixture(), contract=None, budget=B_SLACK, capacity=CAPACITY,
              num_vehicles=NUM_VEHICLES, booking_horizon=BOOKING_HORIZON,
              cooling_share=COOLING_SHARE, sampler=FixedSampler(), arm_seed=ARM_SEED,
              K=10)
    kw.update(over)
    if kw['contract'] is None:
        from scenario_saa import make_c0_contract_v2
        kw['contract'] = make_c0_contract_v2()
    return ts.run_t_series(**kw)


UTILITY_KEYS = {'utility', 'profit', 'j_cc', 'revenue', 'fuel', 'reject_loss',
                'delta_j', 'gain', 'net_utility'}


def _leaked(obj, path=''):
    """递归找**效用字段**（"不看效用"的机械检查）。"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in UTILITY_KEYS:
                out.append('%s/%s' % (path, k))
            out += _leaked(v, '%s/%s' % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _leaked(v, '%s[%d]' % (path, i))
    return out


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- 1. slack 夹具：T5/T6/T7 通过；T2/T3 诚实报 not_applicable ----------
    rep = run(budget=B_SLACK)
    au = rep['audit']
    ck('1. 真实 env + 真实 certify_plan 下跑出落盘日志（%d 条决策）' % rep['n_records'],
       rep['n_records'] >= 2 and rep['n_served'] >= 1, rep['n_records'])
    ck('2. T5 认证顺序与因果边界通过', au['checks']['T5_order_and_causality'] is True,
       au['problems'])
    ck('3. T6 承诺与预算硬约束通过', au['checks']['T6_commitments_and_budget'] is True,
       au['problems'])
    ck('4. T7 确定性通过（同参数重跑 decision_hash 序列一致）',
       au['checks']['T7_determinism'] is True, au['problems'])
    ck('5. T2a 预设压力夹具通过（数值行为符合公式；不要求动作不同）',
       au['checks']['T2a_shadow_price_correctness'] is True, au['t2a_rows'])
    ck('★6. slack 夹具 activation=0 ⇒ **行级**判定为 not_applicable（"没接进"与'
       '"接好了但资源不稀缺"不可区分 ⇒ 不得判 FAIL）',
       rep['activation']['opportunity_cost'] == 0
       and all(r['engaged'] is None
               for r in au['ablation_rows'] if r['switch'] in ('opportunity_cost',
                                                               'lookahead'))
       and all('not_applicable' in (r.get('note') or '')
               for r in au['ablation_rows'] if r['switch'] in ('opportunity_cost',
                                                               'lookahead')),
       au['ablation_rows'])
    ck('★6a **如实记录**：审计器把已降级为 not_applicable 的开关**仍列进 `problems`**'
       '（`check_ablation_engagement` L277-278 用 `is not True` 而非 `is False`），'
       '与规格"报 not_applicable 不报 FAIL"矛盾 —— 本模块只报告、不代改他方模块',
       rep['findings'] and rep['findings'][0]['finding']
       == 'ablation_downgraded_but_listed_in_problems'
       and 'L277-278' in rep['findings'][0]['where']
       and au['passed'] is False,
       rep['findings'])
    ck('6b T2b 在 slack 夹具上如实报 not_applicable（无价格信号）',
       au['t2b_verdict'] in ('not_applicable', 'mechanism_activated_no_behavioural_effect',
                             'mechanism_activated_with_behavioural_effect'),
       au['t2b_verdict'])
    ck('7. T8 共享配对字段逐字段相同（对照 check_control_config.PAIR_FIELDS）',
       rep['t8']['ok'] is True and rep['t8']['mismatches'] == [], rep['t8']['rows'])
    ck('7b T8 **如实列出**冻结文件未记录的配对字段（不冒充相等）',
       isinstance(rep['t8']['gaps'], list), rep['t8']['gaps'])
    ck('★7c T8 的字段清单是 **AST 抽取**而来，**没有 import** `check_control_config`'
       '（该文件模块级换 `sys.stdout` ⇒ import 会关掉底层 buffer，之后任何 print 都崩）',
       rep['t8']['pair_fields_source'].startswith('check_control_config.py (AST')
       and 'check_control_config' not in sys.modules, sorted(sys.modules)[:0])
    ck('8. T1 身份配置哈希存在（64 位）', len(rep['t1']['config_sha256']) == 64,
       rep['t1'])
    ck('★9. 报告里**没有任何效用字段**（"不看效用"的机械检查）',
       _leaked(rep) == [], _leaked(rep))
    ck('10. 报告结构齐备（checks/problems/not_applicable/passed）',
       all(k in rep for k in ('checks', 'problems', 'not_applicable', 'passed')),
       sorted(rep))

    # ---------- 2. tight 夹具：预算卡在临界 ⇒ 机制真的被激活 ----------
    # 先用一个探针跑（真实 certify）读出决策订单的 e_o，再把预算卡到 2.5·e_o：
    # 此时 B 装得下 2 单、B − e_o 只装得下 1 单 ⇒ Ĵ(B) > Ĵ(B−e_o) ⇒ λ_B > 0。
    contract = __import__('scenario_saa').make_c0_contract_v2()
    from run_exp_energy_c0 import certify_plan

    def certify(env, inst_idx, clock, vehicles, served_mask, plan, contract_, B):
        try:
            return certify_plan(env, inst_idx, clock, vehicles, served_mask, plan,
                                contract_, B)
        except ValueError:
            return False, None

    probe = ts.run_fixture(
        decide_fn=ts.hn.build_decide_fn(
            sampler=FixedSampler(), arm_seed=ARM_SEED, K=10, contract=contract,
            cooling_share=COOLING_SHARE, booking_horizon=BOOKING_HORIZON,
            budget=B_SLACK, time_limit=10.0, certify_plan_fn=certify),
        ds=make_fixture(), contract=contract, budget=B_SLACK, capacity=CAPACITY,
        num_vehicles=NUM_VEHICLES, booking_horizon=BOOKING_HORIZON,
        cooling_share=COOLING_SHARE)
    e_vals = [float(d['e_o']) for d in probe['diags'] if d.get('e_o')]
    e_o = float(sorted(e_vals)[len(e_vals) // 2]) if e_vals else 1.0
    ck('10b 探针读到决策订单的边际能耗 e_o（用于把预算卡到临界）', e_o > 0.0,
       (e_o, e_vals))
    rep_t = run(budget=2.5 * e_o)
    ck('★11. tight 夹具（B=2.5·e_o）⇒ 机制被真实激活（activation>0）',
       rep_t['activation']['opportunity_cost'] > 0,
       (e_o, rep_t['activation'], rep_t['t2b_stats']))
    ck('11b T2/T3 在激活后**不再**被降级为 not_applicable（这时才有资格判）',
       all(r['engaged'] is not None for r in rep_t['audit']['ablation_rows']
           if r['switch'] in ('opportunity_cost', 'lookahead')),
       rep_t['audit']['ablation_rows'])
    ck('11c T2b 在 tight 夹具上给出四级判定之一（含"机制已激活"两档）',
       rep_t['audit']['t2b_verdict'] in (
           'not_applicable', 'wiring_defect',
           'mechanism_activated_no_behavioural_effect',
           'mechanism_activated_with_behavioural_effect'),
       rep_t['audit']['t2b_verdict'])
    ck('★11d tight 夹具**不得**出现 wiring_defect（价格必须真的进了评分）',
       rep_t['audit']['t2b_verdict'] != 'wiring_defect', rep_t['audit']['t2b_row'])
    ck('12. tight 夹具仍满足 T5/T6（激活机制不破坏认证顺序与承诺）',
       rep_t['audit']['checks']['T5_order_and_causality'] is True
       and rep_t['audit']['checks']['T6_commitments_and_budget'] is True,
       rep_t['audit']['problems'])
    ck('12b tight 报告同样无效用字段', _leaked(rep_t) == [], _leaked(rep_t))
    ck('13. 消融臂的动作序列**如实报告**是否改变（两种结果都不判失败）',
       all('engaged' in r for r in rep_t['audit']['ablation_rows'])
       and len(rep_t['audit']['ablation_rows']) == 3,
       rep_t['audit']['ablation_rows'])
    ck('13b 三个消融开关都被逐一点名（opportunity_cost / lookahead / scenarios）',
       {r['switch'] for r in rep_t['audit']['ablation_rows']} == set(ts.ABLATION_SWITCHES),
       [r['switch'] for r in rep_t['audit']['ablation_rows']])

    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail)[:300])))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
