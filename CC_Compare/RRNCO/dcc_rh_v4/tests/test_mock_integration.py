"""test_mock_integration.py — 适配器端到端 + provider 调用次数 + 无默认 EDD + 无 Torch。"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from rrnco_guided_adapter import RRNCOGuidedAdapter
from preference import MockPreferenceProvider
from testutil import make_view


def _mock_view(n_vehicles=2):
    coords = [(0, 0), (5, 0), (10, 0), (20, 0)]
    specs = [(i + 1, 0, 0.0, 0.0, 'ready', ()) for i in range(n_vehicles)]
    return make_view(coords, [0, 3, 4, 0], [0, 0, 0, 0], [100, 100, 100, 100],
                     [0, 0, 0, 0], specs, list(range(1, n_vehicles + 1)),
                     10.0, 100.0, [1, 2], True)


class CountingProvider:
    def __init__(self, orderings):
        self.orderings = orderings
        self.calls = 0

    def order(self, subproblem):
        self.calls += 1
        return self.orderings[subproblem.vehicle_id]


class AlternatingProvider:
    def __init__(self):
        self.calls = 0

    def order(self, subproblem):
        self.calls += 1
        return (1, 2) if self.calls % 2 == 1 else (2, 1)


class RaisingProvider:
    def order(self, subproblem):
        raise RuntimeError('backend boom')


def test_provider_called_exactly_once_per_vehicle():
    view = _mock_view()
    prov = CountingProvider({1: (1, 2), 2: (2, 1)})
    RRNCOGuidedAdapter(prov).propose(view)
    assert prov.calls == 2, f'provider 应每车调用一次，实际 {prov.calls}'
    print('  provider 每车恰好调用一次')


def test_audit_ordering_matches_decision_source():
    """交替 provider：审计 ordering 必须与实际 suffix 同源（不再二次调用）。"""
    view = _mock_view()
    prov = AlternatingProvider()
    prop = RRNCOGuidedAdapter(prov).propose(view)
    assert prov.calls == 2, f'provider 调用 {prov.calls}'
    # 决策用到的 ordering = 第一次调用返回的 (1,2)；solve_meta 必须记录同源 ordering
    assert list(prop.solve_meta['model_orderings']['1']) == [1, 2]
    assert prop.suffixes[1] == (1, 2, 0), prop.suffixes
    print('  审计 ordering 与实际决策同源')


def test_provider_exception_triggers_fallback_with_reason():
    view = _mock_view(n_vehicles=1)
    prop = RRNCOGuidedAdapter(RaisingProvider()).propose(view)
    assert prop.fallback_triggered is True
    assert 'RuntimeError' in prop.solve_meta['fallback_reason']
    assert 'boom' in prop.solve_meta['fallback_reason']
    print('  provider 异常 → 带原因 fallback')


def test_no_default_provider():
    try:
        RRNCOGuidedAdapter(None)
        raise AssertionError('默认 provider 未被拒绝')
    except ValueError:
        pass
    print('  无 provider → ValueError（禁止默认 EDD 冒充 RRNCO）')


def test_propose_returns_plan_and_not_mutate_view():
    view = _mock_view()
    before = [(v.vehicle_id, v.mutable_suffix, v.load, v.ready_time)
              for v in view.vehicles]
    prop = RRNCOGuidedAdapter(MockPreferenceProvider('edd')).propose(view)
    assert set(prop.suffixes.keys()) == {1, 2}
    assert prop.fallback_triggered is False
    after = [(v.vehicle_id, v.mutable_suffix, v.load, v.ready_time)
             for v in view.vehicles]
    assert before == after, 'adapter 修改了 view.vehicles'
    print('  propose → PlanProposal；adapter 未改 view（真实 bridge 写保护见 Stage B）')


def test_no_torch_import():
    assert 'torch' not in sys.modules
    print('  未导入 torch/rl4co/tensordict')


def main():
    test_provider_called_exactly_once_per_vehicle()
    test_audit_ordering_matches_decision_source()
    test_provider_exception_triggers_fallback_with_reason()
    test_no_default_provider()
    test_propose_returns_plan_and_not_mutate_view()
    test_no_torch_import()
    print('PASS test_mock_integration')


if __name__ == '__main__':
    main()
