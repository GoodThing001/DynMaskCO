# -*- coding: utf-8 -*-
"""A-v1 零效应根因诊断（修正后 SAA）：cond_hist vs uncond_hist。

两部分：
  1) 场景分布差异（离线、快）：同快照下两采样器的场景集合统计对比
     （数量分布均值/方差、温区/空间/时间桶边缘、总变差）。
  2) 决策同质性（在线、慢）：同两天上两臂逐决策记录（接/拒、acc_u/rej_u、
     场景数、不可行场景数、margin），计算决策一致率与分歧点特征。

用法（扩展根目录）：
  python scripts/evaluation/diag_a1_null.py --days 2 --out results/a1_diag_null
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history,
                          build_history_times_classes, make_c0_contract_v2,
                          VisibleSnapshot, ScenarioOrder, SPOT_CENTERS,
                          INFEASIBLE_SCENARIO_PENALTY)
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926
P_C = {0: 5.0, 1: 10.0, 2: 15.0}


class SpyReplanner(SaaReplanner):
    """逐决策记录接/拒、双分支场景效用和、场景统计。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.decisions = []

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        rec = dict(o=int(o), clock=float(clock),
                   n_scen=[len(s) for s in scenarios],
                   n_infeas=0, acc_sum=0.0, rej_sum=0.0, timed_out=False,
                   cert_failed=False)
        orig = self._sim_scenario
        call_i = [0]

        def wrapped(env2, i2, clk, st, scen_idx, space):
            u = orig(env2, i2, clk, st, scen_idx, space)
            if u == INFEASIBLE_SCENARIO_PENALTY:
                rec['n_infeas'] += 1
            if call_i[0] < len(scenarios):
                rec['acc_sum'] += u
            else:
                rec['rej_sum'] += u
            call_i[0] += 1
            return u

        self._sim_scenario = wrapped
        try:
            super()._saa_decide(env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)
        finally:
            self._sim_scenario = orig
        pc_o = self.reject_penalty[self._class_of(env, inst_idx, o)]
        rec['acc_u'] = rec['acc_sum']
        rec['rej_u'] = rec['rej_sum'] - len(scenarios) * pc_o
        rec['decision'] = 'accept' if o in self._accepted else 'reject'
        rec['timeouts_before'] = self.timeouts
        self.decisions.append(rec)


def _run_day(ds, contract, B, cooling_share, sampler_cls, hist, arm_seed, num_vehicles=15):
    rp = SpyReplanner(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=cooling_share,
                      sampler=sampler_cls(hist, n_neighbors=10) if sampler_cls is CondHistoricalSampler
                      else sampler_cls(hist),
                      reject_penalty=P_C, K=10, time_limit=10.0, arm_seed=arm_seed)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.run(0)
    return rp


def scenario_divergence(hist, days, clock_cuts=(2.0, 4.0, 6.0, 8.0), K=200, rng_seed=0):
    """同快照下三采样器场景集合统计对比。"""
    samplers = dict(uncond=UncondHistoricalSampler(hist),
                    cond=CondHistoricalSampler(hist, n_neighbors=10),
                    explicit=ExplicitFeatureSampler(hist))
    stats = {n: defaultdict(list) for n in samplers}
    for d_idx in days:
        day = hist[d_idx]
        for cut in clock_cuts:
            vis = [o for o in day if o.reveal <= cut + 1e-6]
            if not vis:
                continue
            snap = VisibleSnapshot(clock=cut, orders=tuple(vis), accepted=frozenset(),
                                   rejected=frozenset(), vehicles=(),
                                   energy_used=0.0, budget=700.0, booking_horizon=16.0,
                                   capacity=50.0)
            for name, smp in samplers.items():
                rng = np.random.default_rng(rng_seed)
                scens = smp.sample(snap, rng, K)
                counts = [len(s) for s in scens]
                cls = Counter(o.temp_class for s in scens for o in s)
                spot = Counter(int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
                               for s in scens for o in s)
                tbin = Counter(int(min(o.reveal, 15.99) // 0.5) for s in scens for o in s)
                stats[name]['count'].append(counts)
                stats[name]['cls'].append(cls)
                stats[name]['spot'].append(spot)
                stats[name]['tbin'].append(tbin)
    out = {}
    for name, st in stats.items():
        cnt = np.concatenate(st['count']) if st['count'] else np.array([])
        out[name] = dict(count_mean=float(cnt.mean()) if len(cnt) else None,
                         count_std=float(cnt.std()) if len(cnt) else None,
                         cls_frac={c: float(sum(x[c] for x in st['cls']) /
                                             max(1, sum(sum(x.values()) for x in st['cls'])))
                                   for c in range(3)},
                         spot_frac={b: float(sum(x[b] for x in st['spot']) /
                                             max(1, sum(sum(x.values()) for x in st['spot'])))
                                    for b in range(3)},
                         n_snap=len(st['count']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--out", default="results/a1_diag_null")
    args = ap.parse_args(argv)

    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(
        train_ds, contract, 15, TRAIN_SEED, 0.6, hist_tc, k=10)
    print(f"[info] B={B:.1f} cooling_share={cooling_share:.3f}")

    div = scenario_divergence(hist, days=range(6), K=200)
    print("[divergence]", json.dumps(div, ensure_ascii=False, indent=1))

    dev_ds = add_v2_initial_quality(generate_dataset(60, 200, DEV_SEED), contract)
    report = dict(B=B, cooling_share=cooling_share, days={})
    for i in range(min(args.days, dev_ds['coords'].shape[0])):
        sub = {k: v[i:i + 1] for k, v in dev_ds.items()}
        rp_c = _run_day(sub, contract, B, cooling_share, CondHistoricalSampler, hist, 7002)
        rp_u = _run_day(sub, contract, B, cooling_share, UncondHistoricalSampler, hist, 7001)
        acc_c, rej_c = set(rp_c._accepted), set(rp_c._rejected)
        acc_u, rej_u = set(rp_u._accepted), set(rp_u._rejected)
        agree = len(acc_c & acc_u) + len(rej_c & rej_u)
        total = len(acc_c | rej_c)
        report['days'][str(i)] = dict(
            n_orders=total, agreement_rate=round(agree / max(total, 1), 4),
            cond_accepts=sorted(acc_c), uncond_accepts=sorted(acc_u),
            cond_decisions=rp_c.decisions, uncond_decisions=rp_u.decisions)
        print(f"[day {i}] agree={agree}/{total} ({agree/max(total,1):.2f}) "
              f"cond_acc={len(acc_c)} uncond_acc={len(acc_u)}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "diag.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("saved", os.path.join(args.out, "diag.json"))


if __name__ == "__main__":
    main()
