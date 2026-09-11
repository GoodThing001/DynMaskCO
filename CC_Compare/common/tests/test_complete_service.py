"""B1 Gate 测试 4：full service——所有客户恰好服务一次（含动态 reveal / 多车）。

   - 容量收紧（capacity_tight）迫使多车：complete / n_unserved=0 / n_duplicate=0；
   - 带 mid-horizon reveal 的实例同样 100% 服务；
   - 独立 duplicate Gate：从 trace 重算每个客户被服务次数（不信任 outcome 自报）。
"""
import os
import sys
from collections import Counter

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import helpers
from strict_online_runner import run_instance
from coldchain_contract import default_pilot_profile


def _run(ds):
    return run_instance(ds, capacity=50, num_vehicles=6,
                        adapter_factory=helpers.GreedyEDDAdapter, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=0, data_sha256='t', instance_seed=0)


def _trace_service_counts(rec):
    counts = Counter()
    for v in rec['execution_trace']['vehicles']:
        for s in v['services']:
            if s['picked_order_id'] is not None:
                counts[s['picked_order_id']] += 1
    return counts


def test_static_full_service():
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=1)
    rec = _run(ds)
    assert rec['outcome']['complete'], '静态合成实例未 complete'
    assert rec['outcome']['n_unserved'] == 0
    assert rec['outcome']['n_duplicate'] == 0
    assert rec['outcome']['tw_feasible'] and rec['outcome']['capacity_feasible']
    assert rec['outcome']['depot_return_feasible']
    assert rec['outcome']['all_orders_picked']
    assert rec['outcome']['all_cargo_delivered_to_depot']
    assert rec['outcome']['terminal_manifests_empty']
    assert all(rec['hard_vector'].values()), f'hard vector 存在 False: {rec["hard_vector"]}'

    counts = _trace_service_counts(rec)
    universe = [i for i in range(1, len(ds['demands'])) if ds['demands'][i] > 0]
    for c in universe:
        assert counts[c] == 1, f'客户 {c} 被服务 {counts[c]} 次'
    print('  静态合成实例：所有客户恰好服务一次')


def test_dynamic_full_service():
    ds = helpers.make_synthetic_dataset(capacity_tight=True, reveal_spec={5: 5.0, 6: 6.0},
                                        seed=2)
    rec = _run(ds)
    assert rec['outcome']['complete'], '动态合成实例未 complete（late reveal 客户丢失）'
    assert rec['outcome']['n_unserved'] == 0 and rec['outcome']['n_duplicate'] == 0
    counts = _trace_service_counts(rec)
    universe = [i for i in range(1, len(ds['demands'])) if ds['demands'][i] > 0]
    for c in universe:
        assert counts[c] == 1, f'客户 {c} 被服务 {counts[c]} 次'
    # reveal 事件确实发生（事件层记录了 revealed_customer_ids）
    revealed = [c for ev in rec['events'] for c in ev['revealed_customer_ids']]
    assert 5 in revealed and 6 in revealed, '事件层未记录 reveal'
    print('  动态合成实例：late reveal 客户 100% 服务且 reveal 事件被记录')


def main():
    test_static_full_service()
    test_dynamic_full_service()
    print('PASS test_complete_service')


if __name__ == '__main__':
    main()
