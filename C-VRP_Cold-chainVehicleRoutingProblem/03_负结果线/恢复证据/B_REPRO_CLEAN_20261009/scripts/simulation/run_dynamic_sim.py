"""
Phase H: 完整 MaskCO pipeline (EDD + TW 2opt + C++) 接入动态仿真。

用法:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u simulation/run_dynamic_sim.py \
        --data dcc_50_r1_edod05_test.npz --ckpt phasec_st/step50000.ckpt \
        --edod 0.5 --num_instances 8
"""

import sys, os, argparse, time, numpy as np
_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.dirname(_BASE)
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'lib'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from cvrptw_utils import coord_normalize_visible
from lib import cvrp_two_opt, CVRPPartialInsertion, cvrp_eval_cost
from simulation.rolling_horizon import RollingHorizonSimulator

# 加载 C++ TW 算子
_CPP = None
try:
    import cvrptw_ops as _CPP
    print("[SIM] C++ TW operators loaded")
except ImportError:
    print("[SIM] C++ TW operators NOT found, using Python fallback")

from decoding.cvrptw import (
    evaluate_tw_feasibility, tw_repair_edd, tw_aware_two_opt_py,
    _cpp_repair_edd, _cpp_two_opt, _CPP as _decode_cpp,
)
# Sync C++ state
if _CPP is None:
    import decoding.cvrptw as _d
    _d._CPP = None


def build_full_maskco_replanner(dataset, capacity, model, tw_max, tw_speed=1.0):
    """
    构建完整 MaskCO 重规划函数：
    encode → decode_step → C++ insertion → EDD repair → TW-aware 2opt → 返回新路线。
    """
    coords = dataset['coords'].astype(np.float32)
    demands_raw = dataset['demands'].astype(np.float32)
    tw_start_raw = dataset['tw_start'].astype(np.float32)
    tw_end_raw = dataset['tw_end'].astype(np.float32)
    service_time = dataset.get('service_time',
        np.zeros_like(demands_raw, dtype=np.float32))
    temp_class = dataset.get('temp_class',
        np.zeros_like(demands_raw, dtype=np.float32))
    has_reveal = 'reveal_time' in dataset
    has_visible_mask = 'visible_mask' in dataset
    if has_visible_mask:
        visible_mask_ds = dataset['visible_mask'].astype(np.float32)

    # 预计算距离矩阵
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    num_nodes = coords.shape[1]

    # JIT 编译 encode（支持可见性掩码）
    @jax.jit
    def encode_fn(raw_features, visible_mask=None):
        raw_features = raw_features.at[..., :2].set(
            coord_normalize_visible(raw_features[..., :2], visible_mask))
        if visible_mask is not None and hasattr(model, 'type_embed'):
            return model.encode(raw_features, visible_mask=visible_mask)
        return model.encode(raw_features)

    # JIT 编译 decode_step
    @jax.jit
    def decode_step_fn(features, adjmat, timestep):
        logits = model.decode(features, timestep, adjmat.astype(jnp.float32))
        probs = jax.nn.softmax(logits, axis=-1)
        # 按概率排序候选边
        flat = probs.reshape(num_nodes * num_nodes)
        top_indices = jnp.argsort(flat, descending=True)[:200]
        edges = jnp.stack(jnp.divmod(top_indices, num_nodes), axis=-1)
        return edges

    def replanner(inst_idx, pending_orders, frozen_nodes):
        """
        对实例 inst_idx 进行重规划。
        pending_orders: Order 对象列表
        frozen_nodes: set of frozen node indices
        """
        pending_ids = [o.oid for o in pending_orders if o.oid not in frozen_nodes]
        if len(pending_ids) < 2:
            # 少于2个待规划节点 → 贪心即可
            return _greedy_route(inst_idx, pending_ids, frozen_nodes,
                                 coords, dist_mat, tw_start_raw, tw_end_raw,
                                 service_time, tw_speed)

        # 1. 构建 7D features（单实例）+ 应用可见性掩码
        feat_arrays = [
            coords[inst_idx:inst_idx+1],
            (demands_raw[inst_idx:inst_idx+1] / capacity)[..., None],
            (tw_start_raw[inst_idx:inst_idx+1] / tw_max)[..., None],
            (tw_end_raw[inst_idx:inst_idx+1] / tw_max)[..., None],
            (temp_class[inst_idx:inst_idx+1] / 2.0)[..., None],
        ]
        if has_reveal:
            reveal = dataset['reveal_time'].astype(np.float32)
            feat_arrays.append((reveal[inst_idx:inst_idx+1] / tw_max)[..., None])
        raw_features = np.concatenate(feat_arrays, axis=-1).astype(np.float32)

        # === 可见性掩码 (P0-2 修复): 未来订单特征置零 ===
        vis_mask_inst = None
        if has_visible_mask:
            vis_mask_inst = visible_mask_ds[inst_idx:inst_idx+1]  # (1, nodes)
            vis = vis_mask_inst[..., None]  # (1, nodes, 1)
            raw_features[..., 2:] = raw_features[..., 2:] * vis
            raw_features[..., :2] = raw_features[..., :2] * vis + (1.0 - vis) * 0.5

        # 2. Encode（传递可见性掩码）
        features = np.array(encode_fn(
            jnp.array(raw_features),
            visible_mask=jnp.array(vis_mask_inst) if vis_mask_inst is not None else None,
        ))

        # 3. 构建部分解 adjmat（冻结节点→已知边）
        adjmat = np.zeros((1, num_nodes, num_nodes), dtype=np.float32)
        frozen_list = sorted(frozen_nodes)
        for k in range(len(frozen_list) - 1):
            a, b = frozen_list[k], frozen_list[k+1]
            adjmat[0, a, b] = 1.0

        # 4. Decode → 候选边
        timestep = np.array([0.5], dtype=np.float32)
        candidate_edges = np.array(decode_step_fn(
            jnp.array(features), jnp.array(adjmat), jnp.array(timestep)
        ))[0]  # (num_candidates, 2)

        # 5. TW 过滤（复用 cvrptw 的 TW 可行性矩阵）
        from decoding.cvrptw import compute_tw_feasibility_matrix
        tw_feas = np.array(compute_tw_feasibility_matrix(
            coords[inst_idx:inst_idx+1],
            tw_start_raw[inst_idx:inst_idx+1],
            tw_end_raw[inst_idx:inst_idx+1],
            service_time[inst_idx:inst_idx+1],
            speed=tw_speed,
        ))[0]  # (nodes, nodes)

        # 过滤不可行边
        feasible_mask = tw_feas[candidate_edges[:, 0], candidate_edges[:, 1]]
        candidate_edges = candidate_edges[feasible_mask]

        # 6. C++ Partially Greedy Insertion
        insertion = CVRPPartialInsertion(1, num_nodes, num_workers=1)
        # 初始状态：冻结节点作为已知边
        frozen_edges = []
        for k in range(len(frozen_list) - 1):
            frozen_edges.append([frozen_list[k], frozen_list[k+1]])
        if frozen_edges:
            insertion.set_state(np.array([frozen_edges], dtype=np.int32))

        insertion.insert(
            np.array(candidate_edges[:200]).reshape(1, -1, 2),
            2 * (num_nodes - 1)
        )

        # 7. 获取初始解
        try:
            sols = insertion.get_sols()
        except AssertionError:
            # 插入不完整 → 贪心 fallback
            return _greedy_route(inst_idx, pending_ids, frozen_nodes,
                                 coords, dist_mat, tw_start_raw, tw_end_raw,
                                 service_time, tw_speed)

        # 8. EDD 修复
        if _CPP is not None:
            sols = _cpp_repair_edd(sols,
                coords[inst_idx:inst_idx+1],
                tw_start_raw[inst_idx:inst_idx+1],
                tw_end_raw[inst_idx:inst_idx+1],
                service_time[inst_idx:inst_idx+1],
                tw_speed)
        else:
            sols = tw_repair_edd(sols,
                coords[inst_idx:inst_idx+1],
                tw_start_raw[inst_idx:inst_idx+1],
                tw_end_raw[inst_idx:inst_idx+1],
                service_time[inst_idx:inst_idx+1],
                speed=tw_speed)

        # 9. TW-aware 2-opt
        if _CPP is not None:
            sols = _cpp_two_opt(sols,
                dist_mat[inst_idx:inst_idx+1],
                coords[inst_idx:inst_idx+1],
                tw_start_raw[inst_idx:inst_idx+1],
                tw_end_raw[inst_idx:inst_idx+1],
                service_time[inst_idx:inst_idx+1],
                4, tw_speed)
        else:
            sols = tw_aware_two_opt_py(sols,
                dist_mat[inst_idx:inst_idx+1],
                demands_raw[inst_idx:inst_idx+1],
                capacity, 3.0, 4,
                coords[inst_idx:inst_idx+1],
                tw_start_raw[inst_idx:inst_idx+1],
                tw_end_raw[inst_idx:inst_idx+1],
                service_time[inst_idx:inst_idx+1],
                speed=tw_speed)

        return np.array(sols[0], dtype=np.int32)

    return replanner


def _greedy_route(inst_idx, pending_ids, frozen_nodes,
                  coords, dist_mat, tw_start, tw_end, service_time, speed):
    """贪心构造回退方案。"""
    route = [0]
    current, cur_time = 0, 0.0
    unvisited = set(pending_ids) - frozen_nodes
    while unvisited:
        best, best_d = None, float('inf')
        for j in unvisited:
            d = dist_mat[inst_idx, current, j]
            arr = cur_time + service_time[inst_idx, current] + d / speed
            arr = max(arr, tw_start[inst_idx, j])
            if arr <= tw_end[inst_idx, j] and d < best_d:
                best_d, best = d, j
        if best is None:
            route.append(0); current = 0; cur_time = 0; continue
        route.append(best); unvisited.remove(best)
        cur_time = max(cur_time + service_time[inst_idx, current] + dist_mat[inst_idx, current, best] / speed, tw_start[inst_idx, best])
        current = best
    route.append(0)
    return np.array(route, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--edod', type=float, default=0.5)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--horizon', type=float, default=24.0)
    parser.add_argument('--replan_interval', type=float, default=2.0)
    parser.add_argument('--pending_threshold', type=int, default=3)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--output', type=str, default='simulation/logs/')
    parser.add_argument('--tw_speed', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"=== Phase H: Full MaskCO Pipeline Simulation ===")
    print(f"  CKPT: {args.ckpt}")

    # 加载模型
    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    try:
        model = nnx.merge(nnx.graphdef(model), params)
        print("  Model loaded.")
    except Exception as e:
        print(f"  WARNING: {e}")

    # 加载数据
    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    sim = RollingHorizonSimulator(dataset, capacity=args.capacity, horizon=args.horizon)

    # 构建完整 MaskCO replanner
    maskco_fn = build_full_maskco_replanner(dataset, args.capacity, model, tw_max, args.tw_speed)

    # 贪心 baseline
    def greedy_fn(inst_idx, pending_orders, frozen_nodes):
        pending_ids = [o.oid for o in pending_orders if o.oid not in frozen_nodes]
        if not pending_ids:
            return None
        return _greedy_route(inst_idx, pending_ids, frozen_nodes,
                             dataset['coords'].astype(np.float32),
                             np.sqrt(((dataset['coords'][:, :, None, :] - dataset['coords'][:, None, :, :]) ** 2).sum(axis=-1)).astype(np.float32),
                             dataset['tw_start'].astype(np.float32),
                             dataset['tw_end'].astype(np.float32),
                             dataset.get('service_time', np.zeros_like(dataset['demands'], dtype=np.float32)),
                             args.tw_speed)

    # 跑 3 种策略
    results = {}
    for strategy, fn in [('static', None), ('full_reopt', greedy_fn), ('maskco_full', maskco_fn)]:
        print(f"\n--- {strategy} ---")
        logs = []
        t0 = time.time()

        for i in range(min(args.num_instances, sim.num_instances)):
            orders = sim._build_orders(i)
            known_ids = [o.oid for o in orders if o.status == 'known']
            if len(known_ids) < 5:
                continue

            initial = np.array([0] + known_ids[:15] + [0], dtype=np.int32)

            def model_fn(idx, pending, frozen):
                return fn(idx, pending, frozen) if fn else None

            log = sim.simulate(i, initial, model_fn,
                              replan_interval=args.replan_interval,
                              strategy=strategy,
                              pending_threshold=args.pending_threshold)
            logs.append(log)

        elapsed = time.time() - t0
        completed = [l['final_completed'] / max(l['final_total_orders'], 1) for l in logs]
        dists = [l['final_distance'] for l in logs]
        spoils = [l['final_spoilage'] for l in logs]
        replans = [l['replan_count'] for l in logs]

        print(f"  Instances:  {len(logs)}")
        print(f"  Completed:  {np.mean(completed):.1%}")
        print(f"  Distance:   {np.mean(dists):.1f}")
        print(f"  Legacy spoilage proxy (非C0): {np.mean(spoils):.3f}")
        print(f"  Replans:    {np.mean(replans):.1f}")
        print(f"  Time:       {elapsed:.0f}s")

        results[strategy] = {
            'completed': completed,
            'distance': dists,
            'legacy_spoilage_proxy': spoils,
            'replans': replans,
        }

    # 汇总
    print(f"\n{'='*75}")
    print(f"{'Strategy':<16} {'Completed':>10} {'Distance':>10} {'LegacyProxy':>12} {'Replans':>8}")
    print(f"{'-'*75}")
    for s in ['static', 'full_reopt', 'maskco_full']:
        if s in results:
            r = results[s]
            print(f"{s:<16} {np.mean(r['completed']):>9.1%} {np.mean(r['distance']):>10.1f} "
                  f"{np.mean(r['legacy_spoilage_proxy']):>12.3f} {np.mean(r['replans']):>8.1f}")

    np.savez(os.path.join(args.output, f'dynamic_sim_full_edod{str(args.edod).replace(".","")}.npz'),
             metric_schema=np.asarray('legacy-rolling-horizon-proxy-v1'),
             coldchain_authoritative=np.asarray(False), **results)


if __name__ == '__main__':
    main()
