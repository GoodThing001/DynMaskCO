"""
JF2 Step 9（α=0 regression）— 启发式路径（model=None）必须精确 == JF1-H。

这是 Gate-0 的核心：JF2 assigner 在 α=0（score=base_score=-travel）+ sound mask + greedy 时，
必须逐实例 exact 复现 JF1-H。model 路径的 α=0 退化由 test_jf2_head.py 的 test_alpha_zero_degrades_to_base
覆盖（head 输出 == base_score），两者合起来保证「full JF2 在 α=0 时 == JF1-H」。

纯 NumPy（启发式路径），可本地/服务器跑：

    python scripts/tests/test_jf2_alpha0.py --num_instances 128
"""
import sys, os, argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_BASE), 'simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(_BASE), 'models'))

from strict_online_env import StrictOnlineEnv
from joint_fleet import JointAssignmentReplanner
from jf2 import JF2Replanner


def _fingerprint(ds, rp, n):
    out = []
    for i in range(n):
        env = StrictOnlineEnv(ds, 50, 1.0, 25, replanner=rp)
        traces, served = env.run(i)
        fp = tuple((tr.vehicle_id, tuple(sr.node for sr in tr.services))
                   for tr in sorted(traces, key=lambda t: t.vehicle_id))
        out.append((fp, served))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_instances', type=int, default=128)
    args = parser.parse_args()

    ds = dict(np.load(args.data))
    tw_max = float(ds['tw_end'].max())
    n = min(args.num_instances, ds['coords'].shape[0])

    ref = _fingerprint(ds, JointAssignmentReplanner('heuristic'), n)
    jf2 = _fingerprint(ds, JF2Replanner(ds, 50, tw_max=tw_max, tw_speed=1.0), n)

    differ = sum(1 for (a, _), (b, _) in zip(ref, jf2) if a != b)
    served_same = all(np.array_equal(sa, sb) for (_, sa), (_, sb) in zip(ref, jf2))

    print(f"=== JF2 alpha=0 regression（{n} instances）===")
    print(f"instances_differ={differ}/{n}  served_identical={served_same}")
    if differ == 0 and served_same:
        print("PASS: JF2(model=None) == JF1-H（exact）→ Gate-0 启发式路径成立")
    else:
        print(f"FAIL: {differ} 实例 assignment 不同")


if __name__ == '__main__':
    main()
