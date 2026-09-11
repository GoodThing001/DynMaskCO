"""B1 Gate 测试 6：固定 seed 重复运行完全一致（deterministic parity）。

   - 同 seed 两次运行 → decision_hash 相同（决策与执行内容逐字节一致）；
   - 不同 seed 且 adapter 用 np.random 时 → 结构合法但 decision_hash 可不同（
     证明 decision_hash 确实对决策内容敏感，而不是常数）。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import numpy as np

import helpers
from strict_online_runner import run_instance
from coldchain_contract import default_pilot_profile


def _run(ds, seed):
    return run_instance(ds, capacity=50, num_vehicles=4,
                        adapter_factory=helpers.GreedyEDDAdapter, inst_idx=0,
                        objective='coldchain', profile=default_pilot_profile(),
                        seed=seed, data_sha256='t', instance_seed=0)


def _strip_timing(rec):
    import baseline_contract as bc
    return bc._strip_timing({k: v for k, v in rec.items()
                             if k not in bc._HASH_EXCLUDE_TOP})


class NoisyGreedyAdapter(helpers.GreedyEDDAdapter):
    pass  # helpers 提供 propose 版（车辆顺序噪声）


def test_same_seed_identical():
    ds = helpers.make_synthetic_dataset(capacity_tight=True,
                                        reveal_spec={5: 5.0, 6: 5.0}, seed=7)
    rec1 = _run(ds, seed=42)
    rec2 = _run(ds, seed=42)
    assert rec1['decision_hash'] == rec2['decision_hash'], \
        '同 seed 两次运行 decision_hash 不一致（非确定）'
    assert rec1['actions'] == rec2['actions']
    # events 含计时字段（model_runtime_s），剔除后逐字段比较
    assert _strip_timing({'events': rec1['events']}) == _strip_timing({'events': rec2['events']})
    assert rec1['outcome'] == rec2['outcome']
    assert rec1['hard_vector'] == rec2['hard_vector']
    print('  同 seed 两次运行：decision_hash / actions / events(剔除计时) / outcome 一致')


def test_hash_is_content_sensitive():
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=7)
    rec_a = run_instance(ds, capacity=50, num_vehicles=4,
                         adapter_factory=NoisyGreedyAdapter, inst_idx=0,
                         objective='coldchain', profile=default_pilot_profile(),
                         seed=1, data_sha256='t', instance_seed=0)
    rec_b = run_instance(ds, capacity=50, num_vehicles=4,
                         adapter_factory=NoisyGreedyAdapter, inst_idx=0,
                         objective='coldchain', profile=default_pilot_profile(),
                         seed=2, data_sha256='t', instance_seed=0)
    # 噪声输入集不同 → 记录内容不同（若 hash 是常数会在此失败）
    assert rec_a['decision_hash'] != rec_b['decision_hash'], \
        'decision_hash 对决策内容不敏感（不同 seed 仍相同）'
    print('  decision_hash 对决策内容敏感（不同 seed 的车辆顺序噪声产生不同 hash）')


def main():
    test_same_seed_identical()
    test_hash_is_content_sensitive()
    print('PASS test_adapter_determinism')


if __name__ == '__main__':
    main()
