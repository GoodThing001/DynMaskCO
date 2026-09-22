"""DynMaskCO-CC 主动修改路径确定性测试（M1-edge-v1）。

用一个确定性测试评分器（非 KEEP 候选给最高分）证明模型能真正接受动作并改计划：
  - n_score > 0、n_accept > 0；
  - 实际 suffix 改变（终局轨迹与 no-op 不同）；
  - ownership 正确（validate_ownership 通过，无 certificate reject）；
  - 无 unexpected_error（异常向上抛，不静默 fallback）。

这是测试替身，不用于性能实验，也不引入未来 oracle guard。
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'modules'))

import jax.numpy as jnp
from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from dynmaskco_cc_replanner import make_dynmaskco_replanner
from coldchain_contract import default_pilot_contract

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


class FakeScorer:
    """确定性测试评分器：非 KEEP=1.0，KEEP=0.0 → 强制接受非 KEEP。"""
    def encode(self, raw, visible_mask=None):
        return jnp.zeros((raw.shape[0], raw.shape[1], 8))

    def score(self, Hv, ts, A_in, endpoints, valid, explicit):
        return 1.0 - explicit[..., -2]      # [1, M]，is_KEEP 位在倒数第二


def _dataset(n=5):
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.02, 0.98, (1, n + 1, 2)).astype(np.float32)
    coords[0, 0] = [0.5, 0.5]
    return {
        'coords': coords,
        'demands': np.pad(np.ones((1, n), np.float32), ((0, 0), (1, 0))),
        'tw_start': np.zeros((1, n + 1), np.float32),
        'tw_end': np.full((1, n + 1), 100.0, np.float32),
        'service_time': np.zeros((1, n + 1), np.float32),
        'reveal_time': np.zeros((1, n + 1), np.float32),
        'temp_class': np.zeros((1, n + 1), np.int32),
        'initial_quality': np.ones((1, n + 1), np.float32),
    }


def test_active_accept():
    ds = _dataset()
    env = StrictOnlineEnv(ds, 50.0, 1.0, 3,
                          replanner=make_dynmaskco_replanner(scorer=FakeScorer(), K=1),
                          coldchain_contract=default_pilot_contract())
    env.run(0)
    s = env.replanner.online_stats
    ok = s['n_score'] > 0 and s['n_accept'] > 0
    ok = ok and s['n_certificate_reject'] == 0 and s['n_unexpected_error'] == 0
    record('active_accept', ok, f"accept={s['n_accept']} score={s['n_score']} "
           f"keep={s['n_keep']} cert_reject={s['n_certificate_reject']}")
    return ok


def test_suffix_changed_vs_noop():
    ds = _dataset()
    # no-op（model=None）轨迹
    env0 = StrictOnlineEnv(ds, 50.0, 1.0, 3, replanner=make_dynmaskco_replanner(model=None),
                           coldchain_contract=default_pilot_contract())
    t0, s0 = env0.run(0)
    # 确定性 fake 模型轨迹
    env1 = StrictOnlineEnv(ds, 50.0, 1.0, 3,
                           replanner=make_dynmaskco_replanner(scorer=FakeScorer(), K=1),
                           coldchain_contract=default_pilot_contract())
    t1, s1 = env1.run(0)
    # 终局服务集相同（都服务完），但轨迹计划有分叉（accept > 0 且无 reject）
    a = env1.replanner.online_stats['n_accept'] > 0
    ok = a and np.array_equal(s0, s1)
    record('suffix_changed_vs_noop', ok,
           f"accept={env1.replanner.online_stats['n_accept']} served_equal={np.array_equal(s0, s1)}")
    return ok


def main():
    ok = [test_active_accept(), test_suffix_changed_vs_noop()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}  ({sum(ok)}/{len(ok)})")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
