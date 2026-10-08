# -*- coding: utf-8 -*-
"""探针：_complete_u 返回 None 的具体原因分布（真不可行根因定位）。"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collections import Counter

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, build_history,
                          build_history_times_classes, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926
P_C0 = {0: 0.0, 1: 0.0, 2: 0.0}

reasons = Counter()
firsts = []


def probe(rp):
    orig = rp._complete_u

    def wrapped(space, env, inst_idx, clock, st, wait_until):
        # 复刻 None 路径判定（不重跑 transition）：逐车逐单检查
        contract = rp.contract
        speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
        deadline = float(env.tw_end[inst_idx, 0])
        for s in st.values():
            cc = s['cc']
            if cc is not None and cc.closed:
                continue
            if not s['route'] and s['cur'] == 0:
                continue
            cur, t = s['cur'], s['cur_time']
            for o in s['route']:
                d = float(space.D[cur, o])
                depart = t if t >= space.reveal[o] else float(space.reveal[o])
                arrive = depart + d / speed
                sstart = arrive if arrive >= space.tws[o] else float(space.tws[o])
                if sstart > float(space.twe[o]) + 1e-6:
                    reasons['tw'] += 1
                    if len(firsts) < 3:
                        precheck = rp._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                                       s['load'], s['route'], deadline)
                        # 手动走一遍预检：打印违规单前几单的 t/arr/twe 轨迹
                        walk = []
                        tt, cc, ll = s['cur_time'], s['cur'], s['load']
                        inv = 1.0 / env.tw_speed
                        for oo in s['route']:
                            dd = space.D[cc, oo]
                            a2 = tt + dd * inv
                            if a2 < space.reveal[oo]:
                                a2 = space.reveal[oo]
                            if a2 < space.tws[oo]:
                                a2 = space.tws[oo]
                            walk.append((oo, round(a2, 2), round(float(space.twe[oo]), 2)))
                            tt = a2 + space.st[oo]
                            cc = oo
                            if oo == o:
                                break
                        firsts.append(('tw', o, 'precheck=' + str(precheck),
                                       'inv=' + str(inv),
                                       'speed=' + str(rp.contract.units.speed_kmph /
                                                      rp.contract.units.distance_km_per_unit),
                                       'walk=' + str(walk[-4:])))
                    return None, None
                t = sstart + float(space.st[o])
                cur = o
            wt = wait_until if (wait_until is not None and wait_until > t + 1e-9) else t
            d = float(space.D[cur, 0])
            arrive = wt + d / speed
            if arrive > deadline + 1e-6:
                reasons['return'] += 1
                if len(firsts) < 5:
                    firsts.append(('return', cur, round(wt, 2), round(arrive, 2)))
                return None, None
        return orig(space, env, inst_idx, clock, st, wait_until)

    rp._complete_u = wrapped
    return rp


def main():
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, TRAIN_SEED,
                                                   0.6, hist_tc, k=10)
    dev_ds = add_v2_initial_quality(generate_dataset(60, 200, DEV_SEED), contract)
    sub = {k: v[0:1] for k, v in dev_ds.items()}
    rp = SaaReplanner(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON, contract=contract,
                      cooling_share=cooling_share, sampler=UncondHistoricalSampler(hist),
                      reject_penalty=P_C0, K=10, time_limit=10.0, arm_seed=7001,
                      energy_pricing='marginal', future_policy='density')
    rp = probe(rp)
    env = StrictOnlineEnv(sub, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.run(0)
    print("None-reason counts:", dict(reasons))
    print("capacity check: contract cap =", contract.operational.shared_vehicle_capacity,
          "replanner cap =", rp.capacity)
    for f in firsts:
        print("  first:", f)


if __name__ == "__main__":
    main()
