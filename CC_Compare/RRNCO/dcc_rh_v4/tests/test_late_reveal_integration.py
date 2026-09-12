"""test_late_reveal_integration.py — 公共 strict_online_runner late-reveal 集成（6/6 complete）。

用真实 StrictOnlineEnv + BridgeReplanner 跑一个合成实例：t=0 揭示 4 单，t=5 再
揭示 2 单；适配器用 MockPreferenceProvider（edd）。验证 open-new-vehicle 协调不再
把订单扩散到多辆车导致后续服务能力丧失，最终 6/6 complete。
"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import run_instance, load_objective_profile
from rrnco_guided_adapter import RRNCOGuidedAdapter
from preference import MockPreferenceProvider

_PROJECT = _bootstrap.PROJECT_EXTENSION_ROOT
PROFILE = os.path.join(_PROJECT, 'results', 'o0cc', 'scale_v2',
                       'objective_profile.json')


def _dataset():
    # 7 节点：depot(0) + 客户 1..6；客户 5,6 在 t=5 才揭示
    coords = np.array([[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]],
                      dtype=np.float32)
    demands = np.array([[0, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
    tw_start = np.zeros((1, 7), dtype=np.float32)
    tw_end = np.full((1, 7), 100.0, dtype=np.float32)
    service = np.zeros((1, 7), dtype=np.float32)
    reveal = np.array([[0, 0, 0, 0, 0, 5.0, 5.0]], dtype=np.float32)
    temp_class = np.zeros((1, 7), dtype=np.int32)
    initial_quality = np.ones((1, 7), dtype=np.float32)
    return {'coords': coords, 'demands': demands, 'tw_start': tw_start,
            'tw_end': tw_end, 'service_time': service, 'reveal_time': reveal,
            'temp_class': temp_class, 'initial_quality': initial_quality}


def test_late_reveal_complete():
    ds = _dataset()
    profile = load_objective_profile(PROFILE)
    rec = run_instance(
        ds, capacity=50, num_vehicles=3,
        adapter_factory=lambda: RRNCOGuidedAdapter(MockPreferenceProvider('edd')),
        inst_idx=0, objective='coldchain', profile=profile,
        seed=0, data_sha256='d' * 64,
        adapter_module_path=os.path.join(_DCC_VRP, 'rrnco_guided_adapter.py'),
        instance_seed=0, scene_instance_id='late_reveal_test')
    assert rec['outcome']['complete'], (rec['outcome']['complete'],
                                        rec['outcome']['n_unserved'],
                                        rec['outcome'])
    assert rec['outcome']['n_unserved'] == 0
    assert rec['stats']['fallback_triggered_events'] == 0
    print('  late-reveal 6/6 complete（0 fallback）')


def main():
    test_late_reveal_complete()
    print('PASS test_late_reveal_integration')


if __name__ == '__main__':
    main()
