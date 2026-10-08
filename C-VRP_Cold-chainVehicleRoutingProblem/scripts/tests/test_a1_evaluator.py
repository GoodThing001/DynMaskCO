"""A-v1 评价链认证：数据接通 + 鱼可售/番茄不评价 + 合法拒单 + A-v1 效用 + 负例。"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_contract import ColdChainContractV2
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from strict_online_env import StrictOnlineEnv, Replanner


def _small_dataset():
    # depot(0) + 鱼(class1, node1) + 番茄(class0, node2)，都在 0.5 揭示；批量维度=1
    return {
        'coords': np.array([[[0.5, 0.5], [0.6, 0.5], [0.4, 0.5]]], dtype=np.float32),
        'demands': np.array([[0.0, 2.0, 2.0]], dtype=np.float32),
        'tw_start': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        'tw_end': np.array([[22.0, 22.0, 22.0]], dtype=np.float32),
        'service_time': np.array([[0.0, 0.05, 0.05]], dtype=np.float32),
        'reveal_time': np.array([[1e6, 0.5, 0.5]], dtype=np.float32),
        'temp_class': np.array([[0, 1, 0]], dtype=np.int32),
    }


def _single(ds):
    """batch dataset -> 单实例（评价器用）。"""
    return {k: v[0] for k, v in ds.items()}


class _Base(Replanner):
    def __init__(self, capacity):
        self.capacity = capacity
        self._accepted = set()
        self._rejected = set()

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved = env.get_reserved_customers(vehicles)
        to_assign = [int(i) for i in visible_ids if not served_mask[i]
                     and int(i) not in reserved and int(i) in self._accepted]
        to_assign.sort(key=lambda i: env.tw_end[inst_idx, i])
        active = [v for v in vehicles if v.status in ('idle', 'ready')
                  and (replan_ids is None or v.vehicle_id in replan_ids)]
        st = {v.vehicle_id: dict(cur=v.current_node, cur_time=v.ready_time,
                                 load=float(v.current_load), route=[])
              for v in active}
        for o in to_assign:
            best_v, best_pos, best_incr = None, None, float('inf')
            for vid, s in st.items():
                route = s['route']
                for pos in range(len(route) + 1):
                    pred = s['cur'] if pos == 0 else route[pos - 1]
                    succ = 0 if pos == len(route) else route[pos]
                    incr = (env.dist_mat[inst_idx, pred, o] + env.dist_mat[inst_idx, o, succ]
                            - env.dist_mat[inst_idx, pred, succ])
                    if incr < best_incr:
                        best_v, best_pos, best_incr = vid, pos, incr
            if best_v is not None:
                st[best_v]['route'].insert(best_pos, o)
        for v in active:
            s = st[v.vehicle_id]
            v.mutable_suffix = s['route'] + [0] if s['route'] else [0]


class AcceptAll(_Base):
    """接受所有可插入订单。"""

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        for i in visible_ids:
            if not served_mask[i]:
                self._accepted.add(int(i))


class RejectTomato(_Base):
    """真实拒单：拒绝番茄(node2)，接受鱼(node1)。"""

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        for i in visible_ids:
            if served_mask[i]:
                continue
            if int(i) == 2:
                self._rejected.add(int(i))
            else:
                self._accepted.add(int(i))


class RejectNothing(_Base):
    """既不接受也不登记拒绝（漏登记，负例）。"""

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        pass


def _run(ds, contract, replanner):
    ds = add_v2_initial_quality(ds, contract)
    env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=2, tw_speed=1.5,
                          replanner=replanner, coldchain_contract=contract,
                          booking_horizon=16.0)
    traces, served_mask = env.run(0)
    accepted = set(replanner._accepted)
    rejected = set(replanner._rejected)
    return traces, accepted, rejected, ds


def test_initial_quality():
    c = ColdChainContractV2()
    ds = add_v2_initial_quality(_small_dataset(), c)
    assert ds['initial_quality'][0][1] == 9.0
    assert ds['initial_quality'][0][2] == 1.0
    print("[PASS] initial_quality: fish=9, tomato=1")
    return True


def test_end_to_end():
    c = ColdChainContractV2()
    ds = _small_dataset()
    traces, accepted, rejected, ds = _run(ds, c, AcceptAll(50.0))
    served = {s.node for t in traces for s in t.services}
    assert 1 in served and 2 in served, served
    records = [r for t in traces if t.final_coldchain_state
               for r in t.final_coldchain_state.delivered_to_depot]
    fish = [r for r in records if r.temp_class == 1][0]
    tomato = [r for r in records if r.temp_class == 0][0]
    assert abs(fish.initial_quality - 9.0) < 1e-6
    assert fish.salable is True
    assert tomato.salable is None
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 0.0, 1: 0.0, 2: 0.0})
    assert result['hard_feasible'], result['failures']
    assert result['n_fish_unsalable'] == 0
    assert result['salable_rate']['0'] is None
    assert result['salable_rate']['1'] == 1.0
    assert abs(result['revenue'] - 30.0) < 1e-6
    print(f"[PASS] end-to-end: revenue={result['revenue']:.1f} utility={result['utility']:.2f}")
    return True


def test_legal_rejection():
    # 真实拒单：RejectTomato 拒绝番茄、接受鱼，断言终局可行 + 拒绝损失
    c = ColdChainContractV2()
    ds = _small_dataset()
    traces, accepted, rejected, ds = _run(ds, c, RejectTomato(50.0))
    assert accepted == {1} and rejected == {2}, (accepted, rejected)
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 5.0, 2: 5.0})
    assert result['hard_feasible'], result['failures']
    assert result['reject_loss'] == 5.0
    print(f"[PASS] legal rejection (real reject): hard_feasible=True, reject_loss={result['reject_loss']:.1f}")
    return True


def test_unaccounted_order():
    # 负例：RejectNothing 不接受也不拒绝 -> unaccounted_order，hard_feasible=False
    c = ColdChainContractV2()
    ds = _small_dataset()
    traces, accepted, rejected, ds = _run(ds, c, RejectNothing(50.0))
    assert accepted == set() and rejected == set()
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 5.0, 1: 5.0, 2: 5.0})
    assert not result['hard_feasible']
    assert 'unaccounted_order' in result['failures'], result['failures']
    print(f"[PASS] unaccounted_order negative: failures={result['failures']}")
    return True


def test_missing_final_state():
    # 负例：移除已服务轨迹的 final_coldchain_state -> missing_final_state + accepted_not_delivered
    c = ColdChainContractV2()
    ds = _small_dataset()
    traces, accepted, rejected, ds = _run(ds, c, AcceptAll(50.0))
    for t in traces:
        t.final_coldchain_state = None
    result = evaluate_trace_a1(traces, _single(ds), c, accepted, rejected, B=1e9,
                               reject_penalty_by_class={0: 0.0, 1: 0.0, 2: 0.0})
    assert not result['hard_feasible']
    assert 'missing_final_state' in result['failures'], result['failures']
    print(f"[PASS] missing_final_state negative: failures={result['failures']}")
    return True


if __name__ == '__main__':
    r = {
        'initial_quality': test_initial_quality(),
        'end_to_end': test_end_to_end(),
        'legal_rejection': test_legal_rejection(),
        'unaccounted_order': test_unaccounted_order(),
        'missing_final_state': test_missing_final_state(),
    }
    ok = all(r.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
