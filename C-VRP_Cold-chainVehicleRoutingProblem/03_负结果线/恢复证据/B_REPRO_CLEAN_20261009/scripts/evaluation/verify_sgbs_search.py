"""搜索验收：确认 sgbs_search 探索第二层（不只是首动作 top-2）。

构造一个两步小例：greedy=(A1→B1)=10.0，而最优=(A1→B2)=5.0 需要第二层高分动作 B2。
若 sgbs_search 只做首动作探针，会漏掉 B2 而找不到 5.0。
"""
import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from run_sgbs_fixed_state import sgbs_search


def _propose(env, inst_idx, P, remaining, mutable_ids, K):
    # P 是 (customer, label) 元组序列；remaining 是剩余 customer 列表。
    rem = list(remaining)
    if len(rem) == 2:
        acts = [(rem[0], (rem[0], 'A1'), None), (rem[1], (rem[1], 'A2'), None)]
    else:
        c = rem[0]
        if c == 'c2':
            acts = [(c, (c, 'B1'), None), (c, (c, 'B2'), None)]
        else:
            acts = [(c, (c, 'C1'), None), (c, (c, 'C2'), None)]
    return acts[:K]


def _apply(P, action, allowed_vehicle_ids=None):
    c, label = action
    return P + ((c, label),)


def _hash(P):
    return P


def _eval_J(P):
    return {
        (('c1', 'A1'), ('c2', 'B1')): 10.0,   # greedy
        (('c1', 'A1'), ('c2', 'B2')): 5.0,    # 最优，需第二层 B2
        (('c2', 'A2'), ('c1', 'C1')): 20.0,
        (('c2', 'A2'), ('c1', 'C2')): 15.0,
    }[P]


def main():
    partial = ()
    mask_set = ['c1', 'c2']
    completions, stats = sgbs_search(_propose, None, 0, partial, mask_set, set(),
                                     beam=2, max_per_parent=2, max_completions=8,
                                     apply_fn=_apply, hash_fn=_hash)
    js = {P: _eval_J(P) for P in completions}
    best_P = min(js, key=js.get)
    ok = js[best_P] == 5.0
    print("completions found:", sorted(js.values()))
    print("best plan:", best_P, "J =", js[best_P])
    print("stats:", stats)
    print("PASS: found second-level optimal (A1->B2 = 5.0)" if ok
          else "FAIL: did not find second-level optimal")
    assert ok


if __name__ == '__main__':
    main()
