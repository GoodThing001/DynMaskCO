# -*- coding: utf-8 -*-
"""A1 臂 2 的 V 模型训练：训练日因果回放 + 岭回归 + 先行探针（设计件《A1强动态对照_助手实施设计》§3）。

- 回放：20260925 流前 N 天，逐事件（SAA 决策口径与 cond_hist 一致），对每个场景：
  全影子 u_full 与 h=2 截断影子 u_h2(V=0)（acc/rej 两状态）→
  目标 1（续程价值）= u_full − u_h2；目标 2（效用差探针）= d = u_full_acc − u_full_rej。
  当天从采样器历史中剔除（更严格的因果回放；部署时评估日本就不在历史内）。
- 训练/评估日按天隔离：只用 20260925 流；评估日（20260926/30/01/27/28）从不出现。
- 岭回归（标准化特征）；λ 按训练日内天分折留出 MSE 选；输出 npz（w,b,f_mean,f_std,
  feature_version, identity）+ probe.json（两目标留出 Spearman/MAE，D6 风险预判）。
- 负信号如实记录、不豁免判据（清单 §0 口径）。--workers 按天并行回放。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (os.path.join(_REPO, 'scripts', 'evaluation'),
           os.path.join(_REPO, 'scripts', 'simulation'),
           os.path.join(_REPO, 'scripts', 'coldchain')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from a1_strong_controls import (RolloutVReplanner, V_FEATURE_DIM,  # noqa: E402
                                V_FEATURE_VERSION, a1_v_features)
from coldchain_evaluator_a1 import (FUEL_COST_PER_KM, KM_PER_UNIT, REV,  # noqa: E402
                                    add_v2_initial_quality)
from run_exp_encoder_v3 import compute_budget_and_dwell  # noqa: E402
from run_exp_reserve import BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT, generate_dataset  # noqa: E402
from scenario_saa import (CondHistoricalSampler, build_history,  # noqa: E402
                          build_history_times_classes, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv  # noqa: E402

TRAIN_SEED = 20260925
FIN = 1e6  # 与 INFEASIBLE 区分的有限性阈值


def _fin(x):
    return x is not None and np.isfinite(x) and x > -FIN


class VLogReplanner(RolloutVReplanner):
    """因果回放日志器：决策 = 标准全影子 SAA（与 cond_hist 同口径）；额外记录
    (特征, 续程目标) 与 (特征, 效用差探针)。超时/不可行路径不日志（目标口径不完整）。"""

    def __init__(self, *args, log_list=None, max_events=None, **kwargs):
        super().__init__(*args, v_ckpt=None, **kwargs)
        self.log_list = log_list if log_list is not None else []
        self.max_events = max_events
        self.n_logged_events = 0

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self.n_logged_events = 0

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        deadline = t0 + self.time_limit
        if time.perf_counter() > deadline:
            self._rejected.add(o)
            self.timeouts += 1
            return
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        try:
            insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles, served_mask, clock)
        except ValueError:
            insert_ok = False
            self._plan = {k: list(v) for k, v in saved_plan.items()}
        if not insert_ok:
            self._rejected.add(o)
            return
        acc_plan = {k: list(v) for k, v in self._plan.items()}
        space, scen_orders = self._build_space(env, inst_idx, scenarios)
        idx_of = {id(x): space.n_real + i for i, x in enumerate(scen_orders)}
        scen_indices = [[idx_of[id(x)] for x in scen] for scen in scenarios]
        acc_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, acc_plan)
        self._plan = saved_plan
        rej_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, saved_plan)
        pc_o = self.reject_penalty[self._class_of(env, inst_idx, o)]
        do_log = (self.max_events is None or self.n_logged_events < self.max_events)
        if do_log:
            self.n_logged_events += 1
        acc_u = 0.0
        rej_u = 0.0
        for scen_idx in scen_indices:
            if time.perf_counter() > deadline:
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                return
            a = self._sim_scenario(env, inst_idx, clock, acc_states, scen_idx, space)
            r = self._sim_scenario(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
            if do_log and _fin(a) and _fin(r):
                ra = self._h2_shadow_state(env, inst_idx, clock, acc_states, scen_idx, space)
                rr = self._h2_shadow_state(env, inst_idx, clock, rej_states, scen_idx, space)
                if ra is not None and rr is not None:
                    st_a, est_a, rej_a, _e_a, dist_a = ra
                    st_r, est_r, rej_r, _e_r, dist_r = rr
                    rev_a = sum(REV[int(space.tc[q])] for s in st_a.values() for q in s['route'])
                    rev_r = sum(REV[int(space.tc[q])] for s in st_r.values() for q in s['route'])
                    fuel_a = dist_a * KM_PER_UNIT * FUEL_COST_PER_KM
                    fuel_r = dist_r * KM_PER_UNIT * FUEL_COST_PER_KM
                    a2 = rev_a - fuel_a - rej_a
                    r2 = rev_r - fuel_r - rej_r - pc_o
                    f_a = a1_v_features(space, env, inst_idx, clock, st_a, est_a,
                                        self.budget, self.capacity, self.h)
                    f_r = a1_v_features(space, env, inst_idx, clock, st_r, est_r,
                                        self.budget, self.capacity, self.h)
                    self.log_list.append(dict(
                        f_acc=np.asarray(f_a), f_rej=np.asarray(f_r),
                        t_acc=float(a - a2), t_rej=float(r - r2), d=float(a - r),
                        clock=float(clock), tc=int(self._class_of(env, inst_idx, o))))
            acc_u += a
            rej_u += r
        if time.perf_counter() > deadline:
            self._plan = saved_plan
            self._rejected.add(o)
            self.timeouts += 1
            return
        if acc_u > rej_u:
            self._plan = acc_plan
            self._accepted.add(o)
        else:
            self._plan = saved_plan
            self._rejected.add(o)


def _replay_day(day_i, train_ds, hist_full, contract, args):
    hist_i = hist_full[:day_i] + hist_full[day_i + 1:]
    sampler = CondHistoricalSampler(hist_i, n_neighbors=10)
    rows = []
    rp = VLogReplanner(budget=args.B, capacity=args.capacity,
                       booking_horizon=BOOKING_HORIZON, contract=contract,
                       cooling_share=args.cooling_share, sampler=sampler,
                       reject_penalty={0: 5.0, 1: 10.0, 2: 15.0},
                       K=args.K, time_limit=1e9, arm_seed=7002,
                       shadow_mode='greedy', energy_pricing='marginal',
                       future_policy='density', standby_orders=args.standby_orders,
                       h=args.h, log_list=rows, max_events=args.max_events_per_day)
    ds = add_v2_initial_quality(
        {k: v[day_i:day_i + 1] for k, v in train_ds.items()}, contract)
    env = StrictOnlineEnv(ds, capacity=args.capacity, num_vehicles=args.num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.run(0)
    for row in rows:
        row['day'] = day_i
        row['f_acc'] = row['f_acc'].tolist()
        row['f_rej'] = row['f_rej'].tolist()
    return rows


def _probe(feat, tgt, day_ids, n_held):
    """按天留出：最后 n_held 天为留出集；报告 Spearman/MAE（vs 均值基线 MAE）。"""
    X = np.asarray(feat, dtype=np.float64)
    y = np.asarray(tgt, dtype=np.float64)
    d = np.asarray(day_ids, dtype=np.int64)
    if len(y) < 40 or np.unique(d).size < 2:
        return None
    held = d >= (int(d.max()) - n_held + 1)
    tr = ~held
    if held.sum() < 10 or tr.sum() < 10:
        return None
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    X1 = np.concatenate([Xs[tr], np.ones((int(tr.sum()), 1))], 1)
    coef, *_ = np.linalg.lstsq(X1, y[tr], rcond=None)
    yhat = np.hstack([Xs[held], np.ones((int(held.sum()), 1))]) @ coef
    b = float(y[tr].mean())
    yh, yv = y[held], yhat
    if np.std(yh) < 1e-12:
        return None
    rho = float(np.corrcoef(yh, yv)[0, 1]) if len(yh) > 2 else float('nan')
    return dict(n_train=int(tr.sum()), n_held=int(held.sum()), spearman=float(rho),
                mae=float(np.mean(np.abs(yh - yv))),
                mae_mean_baseline=float(np.mean(np.abs(yh - b))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-days', type=int, default=8,
                    help='20260925 流回放天数（训练/探针共用）')
    ap.add_argument('--max-events-per-day', type=int, default=60)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--h', type=float, default=2.0)
    ap.add_argument('--n-orders', type=int, default=200)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=15)
    ap.add_argument('--rho', type=float, default=0.60)
    ap.add_argument('--n-held-days', type=int, default=2)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    contract = make_c0_contract_v2()
    n_stream = max(16, args.max_days)
    train_ds = generate_dataset(n_stream, args.n_orders, TRAIN_SEED)
    hist_full = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, TRAIN_SEED, args.rho, hist_tc, k=10)
    args.B = float(B)
    args.cooling_share = float(cooling_share)
    # 训练日均单量（marginal 定价固定待命预留 = cooling_share × N）
    mean_orders = float(np.mean([len(d) for d in hist_full]))
    args.standby_orders = mean_orders

    day_ids = list(range(args.max_days))
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import get_context
        ctx = get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            per_day = pool.starmap(
                _replay_day, [(i, train_ds, hist_full, contract, args) for i in day_ids])
    else:
        per_day = [_replay_day(i, train_ds, hist_full, contract, args) for i in day_ids]
    rows = [r for day_rows in per_day for r in day_rows]
    print('replay wall %.0fs, rows=%d' % (time.time() - t0, len(rows)), flush=True)
    if len(rows) < 40:
        raise SystemExit('回放样本过少（%d）——max-days/max-events 过小或全场景不可行' % len(rows))

    F = np.vstack([np.array([r['f_acc'] for r in rows]),
                   np.array([r['f_rej'] for r in rows])])
    T1 = np.array([r['t_acc'] for r in rows] + [r['t_rej'] for r in rows])
    D1 = np.array([r['day'] for r in rows] + [r['day'] for r in rows])
    F2 = np.array([r['f_acc'] for r in rows])
    T2 = np.array([r['d'] for r in rows])
    D2 = np.array([r['day'] for r in rows])
    # 目标 3（决策相关量）：续程价值差 cont_acc − cont_rej，特征 = f_acc − f_rej——
    # 臂 2 的 SAA 决策实际依赖的是 V 在 acc/rej 两状态间的差值，不是单侧特征直接预测边际。
    F3 = np.array([np.asarray(r['f_acc'], float) - np.asarray(r['f_rej'], float)
                   for r in rows])
    T3 = np.array([r['t_acc'] - r['t_rej'] for r in rows])
    D3 = np.array([r['day'] for r in rows])

    probe = {
        'continuation_value': _probe(F, T1, D1, args.n_held_days),
        'utility_diff_margin': _probe(F2, T2, D2, args.n_held_days),
        'continuation_delta': _probe(F3, T3, D3, args.n_held_days),
    }

    # 岭回归 λ 选择（训练日内天分折；留出天不参与任何拟合）
    held = D1 >= (int(D1.max()) - args.n_held_days + 1)
    tr = ~held
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
    Fs = (F - mu) / sd
    best = (None, float('inf'))
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
        errs = []
        for fd in np.unique(D1[tr]):
            va = D1[tr] == fd
            trr = D1[tr] != fd
            X1 = np.concatenate([Fs[tr][trr], np.ones((int(trr.sum()), 1))], 1)
            aug = np.vstack([X1, np.sqrt(lam) * np.eye(X1.shape[1])])
            aug[X1.shape[0]:, -1] = 0.0   # 截距不惩罚
            coef, *_ = np.linalg.lstsq(
                aug, np.concatenate([T1[tr][trr], np.zeros(X1.shape[1])]), rcond=None)
            pred = Fs[tr][va] @ coef[:-1] + coef[-1]
            errs.append(float(np.mean((T1[tr][va] - pred) ** 2)))
        mse = float(np.mean(errs))
        if mse < best[1]:
            best = (lam, mse)
    lam = best[0]
    X1 = np.concatenate([Fs[tr], np.ones((int(tr.sum()), 1))], 1)
    aug = np.vstack([X1, np.sqrt(lam) * np.eye(X1.shape[1])])
    aug[X1.shape[0]:, -1] = 0.0   # 截距不惩罚
    coef, *_ = np.linalg.lstsq(aug, np.concatenate([T1[tr], np.zeros(X1.shape[1])]),
                               rcond=None)
    w, b = coef[:-1], coef[-1]

    src = {}
    for name, sub in (('train_a1_rollout_v.py', 'training'),
                      ('a1_strong_controls.py', 'simulation'),
                      ('scenario_saa.py', 'evaluation'),
                      ('strict_online_env.py', 'simulation')):
        p = os.path.join(_REPO, 'scripts', sub, name)
        if os.path.isfile(p):
            src[name] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    identity = {
        'train_seed': TRAIN_SEED, 'max_days': args.max_days,
        'max_events_per_day': args.max_events_per_day, 'K': args.K, 'h': args.h,
        'n_rows': int(len(rows)), 'lambda': float(lam), 'lambda_cv_mse': float(best[1]),
        'feature_version': V_FEATURE_VERSION, 'feature_dim': V_FEATURE_DIM,
        'source_sha256': src, 'n_held_days': int(args.n_held_days),
        'day_ids_used': sorted(int(x) for x in np.unique(D1)),
        'standby_orders': float(args.standby_orders), 'budget_B': float(B),
    }
    np.savez(os.path.join(args.out, 'v_model.npz'), w=w, b=float(b), f_mean=mu, f_std=sd,
             feature_version=V_FEATURE_VERSION)
    with open(os.path.join(args.out, 'probe.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(identity=identity, probe=probe,
                       note='目标1=续程价值（V 训练目标）；目标2=V 特征→效用差（清单 §0 风险探针，'
                            '历史 D6 Spearman≈−0.04）；目标3=续程价值差（acc−rej 状态对，'
                            '臂 2 决策实际依赖的量）；负信号如实记录、不豁免判据'),
                  f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(dict(identity=identity, probe=probe), indent=2,
                     ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    main()
