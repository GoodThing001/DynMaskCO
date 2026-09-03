"""
Week 4 v6 Day 3-4: 约束消融实验（Constraint Ablation）

三个扰动实验，验证「模型/管线决策因果依赖约束参数」：
  1. temp_swap — 打乱非 depot 已知客户的 temp_class（温度标签全局重排）
     测：type_embed → edge_logits → 路线是否改变（模型层信号，预期在窄 TW 下弱）
  2. tw_relax  — 客户时间窗宽度 ×2（depot horizon 不动）
     测：beam/2-opt 是否有更多优化空间（管线层，预期 cost 下降）
  3. capacity  — 数据不动，只改 decode 的 --capacity {40,60,70}
     测：车辆数与容量反相关（管线层）

三个实验都复用已有 baseline 路线（8 runs × 40 cycles，capacity=50，原始 TW/temp）
作为对比基准，只需对扰动数据重新 decode 后用 compare 子命令比对。

edge_overlap 稳健性修正（2026-08-22）：
  原始 overlap 被两个因素污染——① 目标函数平坦（大量近乎等价的近优解）；② 解码
  随机性（baseline 与扰动在不同会话解码，C++ RNG 种子逐调用递增）。于是 overlap 的
  绝对值无法区分「扰动的真实效应」与「纯解码随机性」。

  修正方案：加「噪声地板」对照 —— 用**相同数据、不同 seed** 再解码一次得到
  routes_noise，度量纯解码随机性造成的重叠度 noise_overlap。然后：
      effect_gap = edge_overlap - noise_overlap
  effect_gap 明显为负 = 扰动效应超出解码随机性（真实存在）；effect_gap ≈ 0 = 扰动
  效应与随机性无法区分。这是让 overlap 变成可解释信号的正确口径。

用法：
  # 生成扰动数据（CPU，秒级）
  python analysis/constraint_ablation.py perturb --ptype temp_swap \
      --data data/p0_fix/dcc_50_r1_edod05_test.npz \
      --out data/p0_fix/dcc_50_r1_edod05_test_tempswap.npz
  python analysis/constraint_ablation.py perturb --ptype tw_relax \
      --data data/p0_fix/dcc_50_r1_edod05_test.npz \
      --out data/p0_fix/dcc_50_r1_edod05_test_twrelax.npz

  # 对比原始 vs 扰动路线（CPU，秒级；--routes_noise 可选，提供噪声地板）
  python analysis/constraint_ablation.py compare \
      --data data/p0_fix/dcc_50_r1_edod05_test.npz \
      --routes_orig logs/week4/routes_typed_edge_r1_edod05.npz \
      --routes_pert logs/week4/routes_tempswap_r1.npz \
      --routes_noise logs/week4/routes_baseline_seed43_r1.npz \
      --out logs/week4/ablation_tempswap_r1
"""

import sys, os, argparse, csv
import numpy as np


# ─────────────────────────────────────────────────────────────
# 基础工具（与 edge_influence.py 一致）
# ─────────────────────────────────────────────────────────────
def compute_dist_mat(coords):
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return np.sqrt((diff ** 2).sum(-1) + 1e-10)


def route_cost(route, dist_mat):
    """扁平路线（depot=0 分隔）的纯行驶距离。"""
    num_nodes = dist_mat.shape[0]
    cost = 0.0
    prev = 0
    for x in route:
        node = int(x)
        if node == 0:
            if prev != 0:
                cost += dist_mat[prev, 0]
            prev = 0
            continue
        if node >= num_nodes:
            continue
        cost += dist_mat[prev, node]
        prev = node
    if prev != 0:
        cost += dist_mat[prev, 0]
    return cost


def route_edges(route, num_nodes):
    """扁平路线 -> 有向边集合（含 depot 往返）。"""
    edges = set()
    prev = 0
    for x in route:
        node = int(x)
        if node == 0:
            if prev != 0:
                edges.add((prev, 0))
            prev = 0
            continue
        if node >= num_nodes:
            continue
        edges.add((prev, node))
        prev = node
    if prev != 0:
        edges.add((prev, 0))
    return edges


def count_vehicles(route, num_nodes):
    """车辆数 = 非空 route segment 数。"""
    segments = 0
    cur = 0
    for x in route:
        node = int(x)
        if node == 0:
            if cur > 0:
                segments += 1
            cur = 0
        elif node < num_nodes:
            cur += 1
    if cur > 0:
        segments += 1
    return segments


# ─────────────────────────────────────────────────────────────
# 扰动函数
# ─────────────────────────────────────────────────────────────
def perturb_temp_swap(data, seed=0):
    """打乱非 depot 已知客户的 temp_class（温度标签全局重排）。"""
    rng = np.random.default_rng(seed)
    temp_class = data['temp_class'].copy()
    if 'reveal_time' in data:
        reveal = data['reveal_time']
    else:
        reveal = np.zeros_like(temp_class, dtype=np.float32)
    n_nodes = temp_class.shape[1]
    n_swapped = 0
    for b in range(temp_class.shape[0]):
        idxs = [i for i in range(1, n_nodes) if reveal[b, i] == 0]
        if len(idxs) < 2:
            continue
        orig = temp_class[b, idxs].copy()
        rng.shuffle(orig)
        temp_class[b, idxs] = orig
        n_swapped += 1
    return temp_class, n_swapped


def perturb_tw_relax(data, factor=2.0):
    """客户时间窗宽度 ×factor（depot horizon 不动）。"""
    tw_start = data['tw_start'].copy()
    tw_end = data['tw_end'].copy()
    mid = (tw_start + tw_end) / 2.0
    half = (tw_end - tw_start) / 2.0
    tw_start[:, 1:] = mid[:, 1:] - half[:, 1:] * factor
    tw_end[:, 1:] = mid[:, 1:] + half[:, 1:] * factor
    return tw_start, tw_end


# ─────────────────────────────────────────────────────────────
# 子命令：perturb
# ─────────────────────────────────────────────────────────────
def cmd_perturb(args):
    data = np.load(args.data)
    if args.ptype == 'temp_swap':
        new_temp, n = perturb_temp_swap(data, seed=args.seed)
        out = {k: data[k] for k in data.files}
        out['temp_class'] = new_temp
        print(f'[temp_swap] 打乱 {n}/{data["temp_class"].shape[0]} 实例的非 depot 已知客户温度标签')
    elif args.ptype == 'tw_relax':
        new_start, new_end = perturb_tw_relax(data, factor=args.factor)
        out = {k: data[k] for k in data.files}
        out['tw_start'] = new_start
        out['tw_end'] = new_end
        print(f'[tw_relax] 客户时间窗宽度 ×{args.factor}（depot 不动）')
    else:
        raise SystemExit(f'未知 ptype: {args.ptype}（可选 temp_swap | tw_relax）')

    np.savez(args.out, **out)
    print(f'[保存] 扰动数据 -> {args.out}')
    # 打印扰动前后统计，方便 QC
    print('  扰动前 tw_start 范围:', data['tw_start'][:, 1:].min(), '~', data['tw_start'][:, 1:].max(),
          '| tw_end 范围:', data['tw_end'][:, 1:].min(), '~', data['tw_end'][:, 1:].max())
    if 'temp_class' in out:
        vals, counts = np.unique(out['temp_class'], return_counts=True)
        print('  扰动后 temp_class 分布:', dict(zip(vals.tolist(), counts.tolist())))


# ─────────────────────────────────────────────────────────────
# 子命令：compare
# ─────────────────────────────────────────────────────────────
def cmd_compare(args):
    data = np.load(args.data)
    orig = np.load(args.routes_orig)['routes']
    pert = np.load(args.routes_pert)['routes']
    noise = None
    if args.routes_noise:
        noise = np.load(args.routes_noise)['routes']
    coords = data['coords']
    dist_mats = compute_dist_mat(coords)

    n_inst = min(orig.shape[0], pert.shape[0], coords.shape[0])
    if noise is not None:
        n_inst = min(n_inst, noise.shape[0])
    if args.num_instances is not None:
        n_inst = min(args.num_instances, n_inst)

    print(f'[约束消融对比] 原始={os.path.basename(args.routes_orig)}')
    print(f'[约束消融对比] 扰动={os.path.basename(args.routes_pert)}  实例数={n_inst}')
    if noise is not None:
        print(f'[约束消融对比] 噪声地板（同数据不同 seed）={os.path.basename(args.routes_noise)}')
    print(f'[约束消融对比] 口径：edge_overlap=有向边 Jaccard；cost=纯距离')

    rows = []
    for b in range(n_inst):
        num_nodes = dist_mats[b].shape[0]
        e1 = route_edges(orig[b], num_nodes)
        e2 = route_edges(pert[b], num_nodes)
        inter = len(e1 & e2)
        union = len(e1 | e2)
        overlap = inter / union if union > 0 else 0.0

        noise_ov = None
        if noise is not None:
            e3 = route_edges(noise[b], num_nodes)
            inter_n = len(e1 & e3)
            union_n = len(e1 | e3)
            noise_ov = inter_n / union_n if union_n > 0 else 0.0

        c1 = route_cost(orig[b], dist_mats[b])
        c2 = route_cost(pert[b], dist_mats[b])
        delta = (c2 - c1) / c1 * 100 if c1 > 0 else 0.0
        v1 = count_vehicles(orig[b], num_nodes)
        v2 = count_vehicles(pert[b], num_nodes)
        rows.append((b, overlap, c1, c2, delta, v1, v2, noise_ov))

    ov = np.array([r[1] for r in rows])
    deltas = np.array([r[4] for r in rows])
    v1s = np.array([r[5] for r in rows])
    v2s = np.array([r[6] for r in rows])

    print('\n' + '=' * 62)
    print('约束消融统计结果')
    print('=' * 62)
    print(f'edge_overlap  mean={ov.mean():.3f}  median={np.median(ov):.3f}  '
          f'min={ov.min():.3f}  max={ov.max():.3f}')
    if noise is not None:
        noise_ovs = np.array([r[7] for r in rows])
        print(f'noise_overlap（同数据不同 seed，随机性地板）mean={noise_ovs.mean():.3f}  '
              f'median={np.median(noise_ovs):.3f}')
        print(f'effect_gap（edge_overlap − noise_overlap，负=扰动效应超出随机性）'
              f'mean={ov.mean() - noise_ovs.mean():+.3f}')
    print(f'cost Δ(%)      mean={deltas.mean():+.3f}  median={np.median(deltas):+.3f}  '
          f'min={deltas.min():+.3f}  max={deltas.max():+.3f}')
    print(f'车辆数         原始 mean={v1s.mean():.2f}  扰动 mean={v2s.mean():.2f}  '
          f'Δ={v2s.mean() - v1s.mean():+.2f}')

    # 解读提示
    print('\n解读提示:')
    print('  temp_swap: overlap 越低 => 模型对温度越敏感；若 overlap≈1 => 温度在窄 TW 下是弱信号')
    print('  tw_relax : cost Δ 越负 => 更宽 TW 带来更多优化空间；overlap 低 => 路线结构改变')
    print('  capacity : 车辆数 Δ 与容量反向 => 管线正确执行容量约束')
    if noise is not None:
        print('  稳健性  : effect_gap 明显为负 => 扰动效应超出解码随机性（真实）；'
              'effect_gap≈0 => 扰动效应与随机性无法区分')

    if args.out:
        csv_path = args.out + '_compare.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['instance', 'edge_overlap', 'orig_cost', 'pert_cost',
                      'cost_delta_pct', 'orig_vehicles', 'pert_vehicles']
            if noise is not None:
                header.append('noise_overlap')
            w.writerow(header)
            w.writerows(rows)
        print(f'\n[保存] 明细 CSV -> {csv_path}')


def main():
    ap = argparse.ArgumentParser(description='Week 4 约束消融（扰动 + 对比）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('perturb', help='生成扰动数据')
    p1.add_argument('--ptype', required=True, choices=['temp_swap', 'tw_relax'])
    p1.add_argument('--data', required=True)
    p1.add_argument('--out', required=True)
    p1.add_argument('--factor', type=float, default=2.0, help='tw_relax 宽度倍数')
    p1.add_argument('--seed', type=int, default=0)
    p1.set_defaults(func=cmd_perturb)

    p2 = sub.add_parser('compare', help='对比原始 vs 扰动路线')
    p2.add_argument('--data', required=True)
    p2.add_argument('--routes_orig', required=True)
    p2.add_argument('--routes_pert', required=True)
    p2.add_argument('--routes_noise', type=str, default=None,
                    help='噪声地板路线（同数据不同 seed 的第二次解码），度量解码随机性')
    p2.add_argument('--num_instances', type=int, default=None)
    p2.add_argument('--out', type=str, default=None)
    p2.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
