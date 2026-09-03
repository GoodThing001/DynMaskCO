"""CaDA 在 DCC-VRP 协议上的评估脚本（不修改 CaDA 原始代码）。

用法（服务器, CaDA conda env）:
    cd /home/hzeng/project/MASKCO-Main/CC_Compare/CaDA
    CUDA_VISIBLE_DEVICES=0 python dcc_vrp/run_dcc.py \
        --ckpt 50/result/2024-1111-1139/checkpoint-300.pt \
        --data ../../C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz \
        --mask_future --capacity 50

输出: 每实例 cost（纯行驶距离）、TW feas、Cap feas、gap vs opt_cost, 以及全量均值。
可行性由 CaDA 解码器的 action mask 逐步保证（TW+容量），本脚本另用原始数据独立复核。
"""
import os
import sys

# 让 CaDA 的 50/（envs, model）与 CaDA/（utils）可 import
ROOT50 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CaDA/
sys.path.insert(0, ROOT50)
sys.path.insert(0, os.path.join(ROOT50, "50"))

import argparse
import time

import numpy as np
import torch
import yaml

# torchrl 新旧 API 兼容（必须在 import dcc_env → CaDA envs.env 之前执行）
import _torchrl_compat  # noqa: F401

from model import VRPModel, reshape_by_heads, PrecomputedCache
from utils.functions import batchify
from dcc_env import DCCEnv
from dcc_data import load_dcc_tensordict


def decode_greedy(model, td, env):
    """复刻 VRPModel.forward 的贪心解码, 但额外返回 actions（用于独立复核可行性）。"""
    args = model.args
    prompt = model.prompt_net(td)["prompt"]
    node_embed = model.encoder(td, prompt)
    num_starts, action = env.select_start_nodes(td)
    td = batchify(td, num_starts)
    actions_list = [action]
    td.set("action", action)
    td = env.step(td)["next"]
    decoder_k = reshape_by_heads(model.decoder.Wk(node_embed), head_num=args.model_params["head_num"])
    decoder_v = reshape_by_heads(model.decoder.Wv(node_embed), head_num=args.model_params["head_num"])
    decoder_single_head_k = node_embed.transpose(1, 2)
    cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)
    step = 0
    while not td["done"].all():
        logprobs, mask = model.decoder(td, cache, num_starts)
        select = VRPModel.greedy(logprobs, mask)
        td.set("action", select)
        actions_list.append(select)
        td = env.step(td)["next"]
        step += 1
        if step > 10000:  # 安全阀（正常 ~2N 步内必收敛）
            raise RuntimeError("decode did not converge")
    actions = torch.stack(actions_list, 1)          # (num_starts, seq)
    td.set("reward", env.get_reward(td, actions))
    return td["reward"], actions


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
    ap.add_argument("--ckpt", required=True, help="CaDA checkpoint (*.pt)")
    ap.add_argument("--data", required=True, help="DCC .npz 文件")
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--mask_future", action="store_true", help="non-anticipatory（未来节点屏蔽）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_instances", type=int, default=None, help="只跑前 N 个实例（smoke 用）")
    opts = ap.parse_args()

    # ---- 组装 args（镜像 CaDA run.py 的 config 装载）----
    with open(os.path.join(ROOT50, "50", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace()
    for k, v in cfg.items():
        setattr(args, k, v)
    args.model_params["sqrt_embedding_dim"] = args.model_params["embedding_dim"] ** 0.5
    args.env["generator_params"]["num_loc"] = 50
    args.log = print

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    if opts.device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # ---- 装载模型 + checkpoint ----
    model = VRPModel(args)
    ckpt = torch.load(opts.ckpt, map_location=opts.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    env = DCCEnv(generator_params=args.env["generator_params"], check_solution=False)

    # ---- 装载数据（TensorDict + 原始数组用于复核）----
    td_all = load_dcc_tensordict(opts.data, capacity=opts.capacity, mask_future=opts.mask_future, device=opts.device)
    raw = np.load(opts.data, allow_pickle=True)
    B = td_all.batch_size[0]
    B = B if opts.max_instances is None else min(B, opts.max_instances)

    costs, tw_feas, cap_feas, complete_feas, gaps = [], [], [], [], []
    t0 = time.time()
    for i in range(B):
        if i % 10 == 0:
            print(f"  [{i+1}/{B}]", flush=True)
        td_i = td_all[i:i + 1]
        td_reset = env.reset(td=td_i)
        with torch.no_grad():
            reward, actions = decode_greedy(model, td_reset, env)   # (num_starts,), (num_starts, seq)
        best = int(reward.argmax().item())                          # 最大 reward = 最小 cost
        cost = float(-reward[best].item())
        best_actions = actions[best].cpu().numpy()
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
    print(f"CaDA  DCC-VRP 结果  ({'non-anticipatory' if opts.mask_future else 'clairvoyant'})")
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
