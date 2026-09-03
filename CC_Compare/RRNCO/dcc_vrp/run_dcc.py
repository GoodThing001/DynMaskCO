"""RRNCO 在 DCC-VRP 协议上的评估脚本（不修改 RRNCO 原始代码）。

用法（服务器, RRNCO conda env）:
    cd /home/hzeng/project/MASKCO-Main/CC_Compare/RRNCO
    CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
        --ckpt checkpoints/rcvrptw/epoch_199.ckpt \
        --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
        --mask_future --capacity 50

输出: 每实例 cost（纯行驶距离）、TW feas、Cap feas、gap vs opt_cost, 以及全量均值。
可行性由 RRNCO 解码器的 action mask 逐步保证（TW+容量+回程），本脚本另用原始数据独立复核。

流程镜像 test.py（known-good），差异：
  1. 关掉 dihedral8 增强（n_aug=1）。
  2. 逐实例 batch=1（可见性异构），num_starts = 可见客户数。
  3. normalize=False（成本可比），cost = -max reward = 纯欧氏距离。
"""
import os
import sys

# 让 `import rrnco` 可用（RRNCO 根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # RRNCO/
sys.path.insert(0, ROOT)

import argparse
import time
import warnings

import numpy as np
import torch

from rrnco.utils import patch_torchrl_specs
patch_torchrl_specs()                       # torchrl 新旧 API 兼容（CompositeSpec→Composite 等）

from rrnco.models import RRNet
from rl4co.utils.ops import batchify

from dcc_env import DCCRMTVRPEnv
from dcc_data import load_dcc_tensordict

warnings.filterwarnings("ignore")


def verify_feasibility(coords, tw_start, tw_end, service, demands, capacity, actions):
    """用原始数据独立复核一条 route 的 TW/容量可行性 + 完整性 + 距离。

    返回 (tw_ok, cap_ok, n_visited, route_dist)：
      n_visited  实际服务到的不同客户数（depot=0 不计）
      route_dist 原始欧氏距离路线总长（应与 harness 报告的 cost 一致，作交叉校验）
    """
    tw_ok, cap_ok = True, True
    t, load, pos, dist_sum = 0.0, 0.0, 0, 0.0
    visited = set()
    for a in actions:
        a = int(a)
        if a == 0:                      # 回到 depot: 计入 return 边，重置车辆状态
            dist_sum += float(np.linalg.norm(coords[pos] - coords[0]))
            t, load, pos = 0.0, 0.0, 0
            continue
        visited.add(a)
        dist = float(np.linalg.norm(coords[pos] - coords[a]))
        dist_sum += dist
        arrive = t + dist
        start = max(arrive, float(tw_start[a]))
        if start > float(tw_end[a]) + 1e-6:
            tw_ok = False
        t = start + float(service[a])
        load += float(demands[a])
        if load > float(capacity) + 1e-6:
            cap_ok = False
        pos = a
    if pos != 0:                        # 以客户结尾：补回 depot 的最后一段
        dist_sum += float(np.linalg.norm(coords[pos] - coords[0]))
    return tw_ok, cap_ok, len(visited), dist_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="RRNCO checkpoint (*.ckpt)")
    ap.add_argument("--data", required=True, help="DCC .npz 文件")
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--mask_future", action="store_true", help="non-anticipatory（未来节点屏蔽）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_instances", type=int, default=None, help="只跑前 N 个实例（smoke 用）")
    opts = ap.parse_args()

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    device = torch.device("cuda:0" if "cuda" in opts.device and torch.cuda.is_available() else "cpu")

    # ---- 装载模型 + checkpoint（mirror test.py）----
    print("Loading checkpoint from", opts.ckpt)
    model = RRNet.load_from_checkpoint(
        opts.ckpt, map_location="cpu", strict=False, load_baseline=False, weights_only=False
    )
    policy = model.policy.to(device).eval()

    env = DCCRMTVRPEnv(generator_params={"num_loc": 50})

    # ---- 装载数据（TensorDict + 原始数组用于复核）----
    td_all = load_dcc_tensordict(opts.data, capacity=opts.capacity, mask_future=opts.mask_future, device="cpu")
    raw = np.load(opts.data, allow_pickle=True)
    B = td_all.batch_size[0]
    B = B if opts.max_instances is None else min(B, opts.max_instances)

    costs, tw_feas, cap_feas, complete_feas, gaps = [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(B):
            if i % 10 == 0:
                print(f"  [{i+1}/{B}]", flush=True)
            td_i = td_all[i:i + 1].to(device)
            td_reset = env.reset(td_i).to(device)
            num_visible = int(td_i["visible_mask"].bool().sum().item()) - 1   # 排除 depot
            assert num_visible >= 1, f"instance {i} has no visible customer"

            out = policy(
                td_reset, env, return_actions=True, phase="val",
                calc_reward=False, num_starts=num_visible,
            )
            # 与 test.py 一致：显式 batchify 后算 reward（normalize=False → 纯距离）
            td_batch = batchify(td_reset, num_visible)
            reward = env.get_reward(td_batch, out["actions"])      # (num_visible,)
            best = int(reward.argmax().item())
            cost = float(-reward[best].item())

            best_actions = out["actions"][best].cpu().numpy()
            tw_ok, cap_ok, n_visited, route_dist = verify_feasibility(
                raw["coords"][i], raw["tw_start"][i], raw["tw_end"][i], raw["service_time"][i],
                raw["demands"][i], opts.capacity, best_actions,
            )
            num_customers = int(raw["demands"].shape[1]) - 1
            complete = (n_visited == num_customers)
            if abs(route_dist - cost) > 1e-3:
                print(f"  [!] inst {i}: route_dist={route_dist:.4f} != cost={cost:.4f}")
            opt = float(raw["opt_costs"][i])
            costs.append(cost); tw_feas.append(tw_ok); cap_feas.append(cap_ok)
            complete_feas.append(complete)
            gaps.append((cost - opt) * 100.0 / opt)

    print(f"\n{'='*72}")
    print(f"RRNCO  DCC-VRP 结果  ({'non-anticipatory' if opts.mask_future else 'clairvoyant'})")
    print(f"  data      : {opts.data}")
    print(f"  instances : {B}")
    print(f"  cost      : {np.mean(costs):.4f}  (mean pure travel distance)")
    print(f"  TW feas   : {100.0*float(np.mean(tw_feas)):.1f}%")
    print(f"  Cap feas  : {100.0*float(np.mean(cap_feas)):.1f}%")
    print(f"  complete  : {100.0*float(np.mean(complete_feas)):.1f}%  (all customers served)")
    print(f"  gap vs greedy: {np.mean(gaps):.2f}%")
    print(f"  elapsed   : {time.time()-t0:.1f}s")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
