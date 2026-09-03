"""
Week 4 v6: 边影响分析（Edge Influence via Counterfactual Relocate）

对模型输出的每条路线，对每个客户节点 v 做反事实 relocate：
  移除 v（其前驱边 (prev_v, v) 断开）→ 贪心重插入到成本最低的位置 → 计算 Δcost。
  Δcost 越大 = v 当前位置越关键 = 进入 v 的边越不可替代。
  Δcost < 0  = v 当前位置有害（挪走反而更好，说明模型此处决策欠优）。

修正 第四周实施指南.md 示例代码的两个 bug：
  1. 读 --save_routes 输出的模型实际路线（routes 字段），不是 data['routes'] 贪心标签
  2. dist_mat 从 coords 现算（.npz 没有 dist_mat 字段）

用法：
  python analysis/edge_influence.py \
      --data data/p0_fix/dcc_50_r1_edod05_test.npz \
      --routes logs/week4/routes_typed_edge_r1_edod05.npz \
      [--num_instances 128] [--tw_check] \
      [--out logs/week4/edge_influence_r1]
"""

import sys, os, argparse, csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def compute_dist_mat(coords):
    """(N, nodes, 2) -> (N, nodes, nodes) 欧氏距离矩阵。"""
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return np.sqrt((diff ** 2).sum(-1) + 1e-10)


def route_cost(route, dist_mat):
    """单条扁平路线（depot=0 分隔）的总行驶距离，含每次返回 depot。"""
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


def extract_route_segments(route, num_nodes):
    """把扁平路线拆成多条车辆 route（以 depot=0 分隔），返回 list[list[int]]。"""
    segments = []
    cur = []
    for x in route:
        node = int(x)
        if node == 0:
            if cur:
                segments.append(cur)
                cur = []
        elif node < num_nodes:
            cur.append(node)
    if cur:
        segments.append(cur)
    return segments


def flatten_segments(segments):
    """车辆 route 列表 -> 扁平序列（每条 route 后补 0 分隔）。"""
    flat = []
    for seg in segments:
        flat.extend(seg)
        flat.append(0)
    return flat


def _segment_feasible(seg, tw_start, tw_end, service_time, dist_mat):
    """检查一条车辆 route 是否满足时间窗（从 depot 出发，可等待）。"""
    t = 0.0
    prev = 0
    for node in seg:
        t += dist_mat[prev, node] + service_time[prev]
        if t < tw_start[node]:
            t = tw_start[node]
        if t > tw_end[node]:
            return False
        prev = node
    return True


def relocate_node(segments, node_v, dist_mat, tw_start=None, tw_end=None,
                  service_time=None, tw_check=False):
    """
    反事实 relocate：把客户 v 从当前位置移除，再贪心插入到同一辆车内成本最低的位置。
    返回 (delta_cost, category) 或 (None, None)（v 未服务）。
    delta_cost = 新总成本 - 原总成本；>0 表示 v 原位置更优（挪走变差）。
    """
    # 1. 找到 v 所在车辆及其位置
    for si, seg in enumerate(segments):
        if node_v in seg:
            vi = seg.index(node_v)
            removed_seg = seg[:vi] + seg[vi + 1:]
            break
    else:
        return None, None  # v 未服务

    orig_cost = route_cost(flatten_segments(segments), dist_mat)

    # 2. 在 removed_seg 内遍历所有插入位置（pos=-1 最前 … len-1 最后），选成本增量最小
    best_inc = float('inf')
    best_pos = 0
    for pos in range(-1, len(removed_seg)):
        a = 0 if pos < 0 else removed_seg[pos]
        b = 0 if pos + 1 >= len(removed_seg) else removed_seg[pos + 1]
        inc = dist_mat[a, node_v] + dist_mat[node_v, b] - dist_mat[a, b]
        if inc >= best_inc:
            continue
        if tw_check:
            cand = ([] if pos < 0 else removed_seg[:pos + 1]) + [node_v] + \
                   ([] if pos + 1 >= len(removed_seg) else removed_seg[pos + 1:])
            if not _segment_feasible(cand, tw_start, tw_end, service_time, dist_mat):
                continue
        best_inc = inc
        best_pos = pos

    # 3. 插入
    if best_pos == -1:
        rebuilt = [node_v] + removed_seg
    else:
        rebuilt = removed_seg[:best_pos + 1] + [node_v] + removed_seg[best_pos + 1:]

    new_segments = segments.copy()
    new_segments[si] = rebuilt
    new_cost = route_cost(flatten_segments(new_segments), dist_mat)
    return new_cost - orig_cost, None


def categorize(delta_pct, critical_thresh, neutral_thresh):
    if delta_pct > critical_thresh:
        return 'critical'
    elif delta_pct > neutral_thresh:
        return 'beneficial'
    elif delta_pct > -neutral_thresh:
        return 'neutral'
    else:
        return 'harmful'


def main():
    ap = argparse.ArgumentParser(description='Week 4 边影响分析（反事实 relocate）')
    ap.add_argument('--data', required=True, help='数据 .npz')
    ap.add_argument('--routes', required=True, help='--save_routes 输出的路线 .npz')
    ap.add_argument('--num_instances', type=int, default=None, help='分析前 N 个实例（默认全部）')
    ap.add_argument('--tw_check', action='store_true', help='重建时做时间窗可行性检查（更慢）')
    ap.add_argument('--critical_thresh', type=float, default=10.0, help='critical 阈值 %（默认 10）')
    ap.add_argument('--neutral_thresh', type=float, default=0.1, help='neutral 阈值 %（默认 0.1）')
    ap.add_argument('--out', type=str, default=None, help='输出前缀（写 CSV + 直方图）')
    args = ap.parse_args()

    data = np.load(args.data)
    routes_npz = np.load(args.routes)
    routes = routes_npz['routes']
    coords = data['coords']
    dist_mats = compute_dist_mat(coords)
    has_tw = ('tw_start' in data) and ('tw_end' in data) and ('service_time' in data)

    n_inst = routes.shape[0] if args.num_instances is None else min(args.num_instances, routes.shape[0])
    print(f'[边影响分析] 数据={os.path.basename(args.data)} 路线={os.path.basename(args.routes)}')
    print(f'[边影响分析] 实例数={n_inst}  tw_check={args.tw_check}  critical>{args.critical_thresh}%  neutral<=±{args.neutral_thresh}%')
    print(f'[边影响分析] 口径说明：Δcost = relocate(v) 后成本 - 原成本，纯距离、不含 TW/容量（除非 --tw_check）')

    rows = []          # (instance, node, delta_cost, delta_pct, category)
    n_skipped = 0

    for b in range(n_inst):
        route = routes[b]
        dist_mat = dist_mats[b]
        num_nodes = dist_mat.shape[0]
        segments = extract_route_segments(route, num_nodes)
        served = sorted({node for seg in segments for node in seg})

        if has_tw:
            tw_start = data['tw_start'][b]
            tw_end = data['tw_end'][b]
            service_time = data['service_time'][b]
        else:
            tw_start = tw_end = service_time = None

        base_cost = route_cost(route, dist_mat)
        for v in served:
            delta, _ = relocate_node(segments, v, dist_mat,
                                     tw_start, tw_end, service_time, args.tw_check)
            if delta is None:
                n_skipped += 1
                continue
            delta_pct = (delta / base_cost * 100) if base_cost > 0 else 0.0
            cat = categorize(delta_pct, args.critical_thresh, args.neutral_thresh)
            rows.append((b, v, delta, delta_pct, cat))

        if (b + 1) % 20 == 0 or b == n_inst - 1:
            print(f'  [{b + 1}/{n_inst}] 累计 {len(rows)} 条边')

    if not rows:
        print('没有分析到任何边，检查 --routes 文件是否为空或格式不符。')
        return

    # 统计
    deltas = np.array([r[3] for r in rows])  # delta_pct
    cats = [r[4] for r in rows]
    cat_count = {c: cats.count(c) for c in ['critical', 'beneficial', 'neutral', 'harmful']}
    total = len(rows)

    print('\n' + '=' * 62)
    print('边影响统计结果')
    print('=' * 62)
    print(f'总边数: {total}  (跳过未服务节点 {n_skipped})')
    print(f'Δcost(%)  mean={deltas.mean():+.3f}  median={np.median(deltas):+.3f}  '
          f'std={deltas.std():.3f}  min={deltas.min():+.3f}  max={deltas.max():+.3f}')
    print('\n分类分布:')
    for c in ['critical', 'beneficial', 'neutral', 'harmful']:
        n = cat_count[c]
        print(f'  {c.capitalize():<11} {n:5d}  ({n / total * 100:5.1f}%)')

    # case study：最 critical 的边
    by_delta = sorted(rows, key=lambda r: -r[3])
    print('\nTop 5 最 critical 的边（Δcost 最大，最不可替代）:')
    for inst, node, dc, dp, cat in by_delta[:5]:
        print(f'  inst={inst:<4} node={node:<3} Δcost={dc:+.3f}  Δpct={dp:+.2f}%')

    n_harmful = cat_count['harmful']
    if n_harmful > 0:
        by_harmful = sorted(rows, key=lambda r: r[3])
        print(f'\nTop 5 最 harmful 的边（Δcost 最小，移除反而更好，共 {n_harmful} 条）:')
        for inst, node, dc, dp, cat in by_harmful[:5]:
            print(f'  inst={inst:<4} node={node:<3} Δcost={dc:+.3f}  Δpct={dp:+.2f}%')
    else:
        print('\n无 harmful 边：所有 relocate 都不会让成本下降（模型决策在距离意义上一致合理）。')

    # 保存
    if args.out:
        csv_path = args.out + '_influences.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['instance', 'node', 'delta_cost', 'delta_pct', 'category'])
            w.writerows(rows)
        print(f'\n[保存] 明细 CSV -> {csv_path}')

        if _HAS_MPL:
            png_path = args.out + '_hist.png'
            plt.figure(figsize=(9, 5))
            plt.hist(deltas, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', lw=2, label='No change')
            plt.axvline(args.critical_thresh, color='orange', linestyle='--', lw=1,
                        label=f'Critical (+{args.critical_thresh:.0f}%)')
            plt.xlabel('Cost Change (%)')
            plt.ylabel('Number of Edges')
            plt.title('Edge Influence Distribution (Counterfactual Relocate)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            print(f'[保存] 直方图 -> {png_path}')
        else:
            print('[提示] 未安装 matplotlib，跳过直方图。')


if __name__ == '__main__':
    main()
