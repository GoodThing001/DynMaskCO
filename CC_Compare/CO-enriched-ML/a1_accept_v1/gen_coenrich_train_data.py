"""生成 CO-enriched-ML 的 A-v1 重训数据（可见池特征 + A-v1 接单标签）。

为什么需要「重训数据」
--------------------
原文 ML-CO 的监督信号是**辅助下界解**（`experiments/*/best-sol.json`，由全天已知的
PC-HGS 预先算出）在该 epoch 的派车集合 `n_i`：即「该请求现在值得派吗」。A-v1 没有等价
预解资产，且原文无 accept/reject。因此按用户批准的口径，用 **A-v1 的强非学习参考臂**
（`UncondDynamicReplanner`：无条件未来预算门槛 + 真实 C0 certify）在**训练天**上跑出的
接单集合作为标签：

    label(order, event) = 1  iff  参考臂最终接受了该订单

含义是「参考臂在该揭示事件会接的订单」= 原文『值得现在派』在 A-v1 的最接近对应物。
**这是行为克隆口径，不是原文的 anticipative-lower-bound 最优派车集合**（A-v1 无该资产），
必须在 RESULTS.md 中如实标注。

数据口径与外部臂完全一致
------------------------
- 合同 `make_c0_contract_v2()`；预算 B / cooling_share 取自既有 gate.json（只读）；
- 环境 `StrictOnlineEnv(capacity=50, num_vehicles=15, tw_speed=SPEED_KMH/KM_PER_UNIT,
  booking_horizon=BOOKING_HORIZON)`；
- 订单数 200、rho 与外部臂同；**训练天 seed = 20260920**（≠ 开发集 20260926，
  也 ≠ 保留测试 20260927/28）；
- 特征 = 部署同源：用**冻结的** `OrderingAcceptReplanner._build_view`（未绑定方法调用）
  构造 view → `subproblem.build_subproblem` → `coenrich_features.compute_features`。
  与 provider 在线路径逐字节同源，避免「训练/部署特征不一致」。

不修改任何冻结文件：所有项目代码只读导入；本文件在 `a1_accept_v1/` 内。

用法（服务器，cc_compare env）：
    cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
    /home/hzeng/envs/cc_compare/bin/python \
        ../CC_Compare/CO-enriched-ML/a1_accept_v1/gen_coenrich_train_data.py \
        --days 8 --out ../CC_Compare/CO-enriched-ML/a1_accept_v1/data/train_pool.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CC = os.path.dirname(os.path.dirname(_HERE))                     # CC_Compare/
_ROOT = os.path.dirname(_CC)                                      # 仓库根
_EXT = os.path.join(_ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
for _p in (_HERE, os.path.join(_EXT, 'scripts'),
           os.path.join(_EXT, 'scripts', 'coldchain'),
           os.path.join(_EXT, 'scripts', 'simulation'),
           os.path.join(_EXT, 'scripts', 'evaluation'),
           os.path.join(_CC, 'RRNCO', 'dcc_rh_v4')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality            # noqa: E402
from run_exp_reserve import (generate_dataset, BOOKING_HORIZON,      # noqa: E402
                             SPEED_KMH, KM_PER_UNIT)
from run_exp_encoder_v3 import compute_budget_and_dwell              # noqa: E402
from scenario_saa import make_c0_contract_v2                         # noqa: E402
from strict_online_env import StrictOnlineEnv                        # noqa: E402
from run_exp_rule_control import UncondDynamicReplanner              # noqa: E402
from rrnco_accept_replanner import OrderingAcceptReplanner           # noqa: E402
from subproblem import build_subproblem                              # noqa: E402
from coenrich_features import compute_features                       # noqa: E402

TRAIN_DATA_SEED = 20260920     # 训练天（≠ dev 20260926，≠ final 20260927/28）
HIST_SEED = 20260925           # 项目约定的 train seed（仅用于 hist/lam 标定）
DEV_SEED = 20260926


class RecordingUncondReplanner(UncondDynamicReplanner):
    """参考臂 + 逐揭示事件记录（可见池 → 特征）。标签在当天结束后回填。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.events = []          # list[(inst_idx, clock, pool, X)]

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = sorted({int(i) for i in visible_ids
                          if not served_mask[i] and int(i) not in reserved
                          and int(i) not in self._accepted
                          and int(i) not in self._rejected},
                         key=lambda i: env.reveal_time[inst_idx, i])
        if new_set:
            pool = sorted({int(c) for c in self._accepted if not served_mask[c]}
                          | set(new_set))
            try:
                # 未绑定方法调用：与部署路径 **同一份** view 构造代码（零重复实现）
                view = OrderingAcceptReplanner._build_view(
                    self, env, inst_idx, clock, vehicles, pool)
                sp = build_subproblem(view, view.vehicles[0])
                X, names = compute_features(sp, feature_set='dynamicstatic')
                pool_eff = [int(c) for c in sp.pool_customer_ids]
                if pool_eff:
                    # X 行 0 = depot；其余行与 pool_eff 同序
                    self.events.append((int(inst_idx), float(clock), pool_eff,
                                        set(int(c) for c in new_set),
                                        np.asarray(X[1:1 + len(pool_eff)], float),
                                        names))
            except Exception as exc:                      # noqa: BLE001
                sys.stderr.write('[gen] event 特征失败(跳过): %r\n' % (exc,))
        super().on_reveal(env, inst_idx, clock, vehicles, served_mask, visible_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=8)
    ap.add_argument('--n-orders', type=int, default=200)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=15)
    ap.add_argument('--rho', type=float, default=0.60)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=TRAIN_DATA_SEED)
    ap.add_argument('--baseline-from', default=os.path.join(
        _EXT, 'results', 'a1_step2_gate_c1_20260926', 'gate.json'))
    ap.add_argument('--label-scope', choices=['new', 'pool'], default='new',
                    help="new=只标注本次新揭示订单(默认，标签均衡)；pool=标注整池")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    contract = make_c0_contract_v2()
    baseline = json.load(open(args.baseline_from, encoding='utf-8'))
    B = float(baseline['budget']['B'])
    cooling_share = float(baseline['budget']['cooling_share'])
    print('budget B=%.4f cooling_share=%.6f  (from %s)'
          % (B, cooling_share, args.baseline_from))

    # hist / lam 标定：与 run_exp_a1_compare 同法（train seed 20260925）
    train_ds = generate_dataset(200, args.n_orders, HIST_SEED)
    hist = []
    for i in range(200):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))

    gate_ds = add_v2_initial_quality(
        generate_dataset(args.days, args.n_orders, args.seed), contract)
    print('训练天 seed=%d days=%d n_orders=%d' % (args.seed, args.days, args.n_orders))

    rp = RecordingUncondReplanner(budget=B, capacity=args.capacity,
                                  booking_horizon=BOOKING_HORIZON, contract=contract,
                                  cooling_share=cooling_share, hist=hist, k=args.k)
    env = StrictOnlineEnv(gate_ds, capacity=args.capacity,
                          num_vehicles=args.num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract,
                          booking_horizon=BOOKING_HORIZON)

    Xs, ys, ctxs, names = [], [], [], None
    t0 = time.time()
    for i in range(args.days):
        rp.events = []
        rp._reset_if_new(-1)               # 强制清空（跨天复用同一 replanner）
        traces, _ = env.run(i)
        accepted = set(int(c) for c in rp._accepted)
        n_ev = len(rp.events)
        n_rows = 0
        for (inst_idx, clock, pool, new_set, X, nm) in rp.events:
            names = nm
            for row, c in zip(X, pool):
                if args.label_scope == 'new' and c not in new_set:
                    continue        # 只标注「本次新揭示」的订单（默认）
                Xs.append(row)
                ys.append(1.0 if c in accepted else 0.0)
                ctxs.append((i, clock, c))
                n_rows += 1
        print('day %2d: 事件=%-4d 行=%-5d 参考接受=%d/%d  served=%d'
              % (i, n_ev, n_rows, len(accepted),
                 int(np.sum(gate_ds['demands'][i][1:] > 0)), len(traces)))
    X = np.asarray(Xs, float)
    y = np.asarray(ys, float)
    print('总样本 %d，正例率 %.4f，耗时 %.1fs' % (len(y), y.mean(), time.time() - t0))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, X=X, y=y,
             ctx_day=np.array([c[0] for c in ctxs], int),
             ctx_clock=np.array([c[1] for c in ctxs], float),
             ctx_cust=np.array([c[2] for c in ctxs], int),
             feature_names=np.array(names or [], dtype=object),
             meta=np.array([json.dumps({
                 'seed': args.seed, 'days': args.days, 'n_orders': args.n_orders,
                 'capacity': args.capacity, 'num_vehicles': args.num_vehicles,
                 'rho': args.rho, 'k': args.k, 'B': B,
                 'cooling_share': cooling_share,
                 'baseline_from': os.path.abspath(args.baseline_from),
                 'reference_arm': 'UncondDynamicReplanner',
                 'label': 'accepted_by_reference_arm (行为克隆口径)',
                 'label_scope': args.label_scope,
                 'feature_set': 'dynamicstatic(A-v1 可见池口径)',
                 'n_features': int(X.shape[1]),
             })], dtype=object))
    print('saved', args.out, X.shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
