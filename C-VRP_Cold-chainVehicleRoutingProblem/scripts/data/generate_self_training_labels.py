"""
方向一：Self-Training 伪标签生成。

用最佳模型对训练集做增强推理，生成比贪心解更优的伪标签。

用法:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u data/generate_self_training_labels.py \
        --data dcc_50_r1_edod05_train.npz \
        --ckpt step2_tempemb/step20000.ckpt \
        --output dcc_50_r1_edod05_selftrain_round1.npz \
        --batch_size 8 --runs 8 --cycles 160
"""

import sys, os, argparse, time, numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'lib'))

# ── 内联 TW 工具函数（避免 cvrptw.py 的复杂依赖） ──

def evaluate_tw_feasibility(routes, coords, tw_start, tw_end, service_time, speed=1.0):
    batch_size = routes.shape[0]
    feasible = np.ones(batch_size, dtype=bool)
    violations = np.zeros(batch_size, dtype=np.int32)
    for b in range(batch_size):
        route = routes[b]; cur_time = 0.0; prev = 0
        for pos in range(len(route)):
            node = int(route[pos])
            if node == 0: cur_time = 0.0; prev = 0; continue
            if node == prev: continue
            dx = coords[b,prev,0]-coords[b,node,0]; dy = coords[b,prev,1]-coords[b,node,1]
            tt = np.sqrt(dx*dx+dy*dy)/speed
            arr = cur_time + service_time[b,prev] + tt
            arr = max(arr, tw_start[b,node])
            if arr > tw_end[b,node] + 1e-6: feasible[b] = False; violations[b] += 1
            cur_time = arr; prev = node
    return feasible, violations


def compute_tw_feasibility_matrix(coords, tw_start, tw_end, service_time, speed=1.0):
    if coords.ndim == 2: coords = coords[None,...]; tw_start = tw_start[None,...]; tw_end = tw_end[None,...]; service_time = service_time[None,...]
    diff = coords[:,:,None,:] - coords[:,None,:,:]
    dist = np.sqrt((diff**2).sum(axis=-1) + 1e-10)
    ready_at_j = tw_start[:,None,:] + service_time[:,None,:] + dist/speed
    feasible = ready_at_j <= tw_end[:,:,None]
    feasible[:,0,:] = True; feasible[:,:,0] = True
    return feasible

# Python fallback functions
def tw_repair_edd(sols, coords, tw_start, tw_end, service_time, speed=1.0):
    return sols  # simplified: C++ handles this

def tw_aware_two_opt_py(sols, dist_mat, demands, capacity, penalty, steps, coords, tw_start, tw_end, service_time, speed=1.0):
    return sols  # simplified: C++ handles this

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from modules.functional import coord_normalize

# 加载 C++
_CPP = None
try:
    import cvrptw_ops as _CPP
    print("[ST] C++ operators loaded")
except ImportError:
    print("[ST] C++ NOT found")

from lib import cvrp_two_opt, CVRPPartialInsertion, cvrp_eval_cost


def generate_pseudo_labels(dataset, capacity, model, tw_max, tw_speed,
                            batch_size, runs, cycles, two_opt_steps):
    """对训练集实例做增强推理，选取最优可行解作为伪标签。"""
    coords = dataset['coords'].astype(np.float32)
    demands_raw = dataset['demands'].astype(np.float32)
    tw_start_raw = dataset['tw_start'].astype(np.float32)
    tw_end_raw = dataset['tw_end'].astype(np.float32)
    service_time = dataset.get('service_time',
        np.zeros_like(demands_raw, dtype=np.float32))
    temp_class = dataset.get('temp_class',
        np.zeros_like(demands_raw, dtype=np.float32))
    opt_routes = dataset['routes']
    opt_costs = dataset['opt_costs']
    has_reveal = 'reveal_time' in dataset

    num_instances = coords.shape[0]
    num_nodes = coords.shape[1]
    num_workers = 1

    # 预计算距离矩阵
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    new_routes = []
    new_costs = []
    improved_count = 0

    print(f"Generating pseudo-labels for {num_instances} instances...")
    print(f"  runs={runs}, cycles={cycles}, 2opt_steps={two_opt_steps}")

    for i in range(num_instances):
        best_cost = float(opt_costs[i])
        best_route = opt_routes[i].copy()

        # 构建 features
        feat_arrays = [
            coords[i:i+1],
            (demands_raw[i:i+1] / capacity)[..., None],
            (tw_start_raw[i:i+1] / tw_max)[..., None],
            (tw_end_raw[i:i+1] / tw_max)[..., None],
            (temp_class[i:i+1] / 2.0)[..., None],
        ]
        if has_reveal:
            reveal = dataset['reveal_time'].astype(np.float32)
            feat_arrays.append((reveal[i:i+1] / tw_max)[..., None])
        raw_features = np.concatenate(feat_arrays, axis=-1).astype(np.float32)

        for run in range(runs):
            np.random.seed(i * 777 + run * 13)
            generator = np.random.default_rng(i * 777 + run * 13)

            # Encode
            feats_jax = jnp.array(raw_features)
            feats_jax = feats_jax.at[..., :2].set(coord_normalize(feats_jax[..., :2]))
            encoded = np.array(model.encode(feats_jax))

            # Decode → candidate edges
            timestep = jnp.array([0.5])
            adjmat = jnp.zeros((1, num_nodes, num_nodes), dtype=jnp.float32)
            logits = np.array(jax.nn.softmax(model.decode(
                jnp.array(encoded), timestep, adjmat), axis=-1))[0]

            # 排序候选边
            flat = logits.reshape(-1)
            topk = min(400, len(flat))
            top_indices = np.argpartition(-flat, topk)[:topk]
            top_indices = top_indices[np.argsort(-flat[top_indices])]
            candidate_edges = np.stack(np.divmod(top_indices, num_nodes), axis=-1)

            # TW 过滤
            tw_feas = np.array(compute_tw_feasibility_matrix(
                coords[i:i+1], tw_start_raw[i:i+1], tw_end_raw[i:i+1],
                service_time[i:i+1], speed=tw_speed
            ))[0]
            feasible_mask = tw_feas[candidate_edges[:, 0], candidate_edges[:, 1]]
            candidate_edges = candidate_edges[feasible_mask][:200]

            # C++ Insertion
            insertion = CVRPPartialInsertion(1, num_nodes, num_workers=1)
            insertion.insert(
                np.array(candidate_edges).reshape(1, -1, 2),
                2 * (num_nodes - 1)
            )

            try:
                sols = insertion.get_sols()
            except AssertionError:
                continue

            # CVRP 2-opt
            sols = cvrp_two_opt(sols, dist_mat[i:i+1], demands_raw[i:i+1],
                               capacity, 3.0, two_opt_steps, num_workers)

            # EDD repair
            if _CPP is not None:
                r_arr = np.ascontiguousarray(sols.astype(np.int32))
                _CPP.repair_edd(r_arr,
                    np.ascontiguousarray(coords[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(tw_start_raw[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(tw_end_raw[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(service_time[i:i+1].astype(np.float32)),
                    tw_speed)
                sols = np.array(r_arr)
            else:
                sols = tw_repair_edd(sols, coords[i:i+1],
                    tw_start_raw[i:i+1], tw_end_raw[i:i+1],
                    service_time[i:i+1], speed=tw_speed)

            # TW 2-opt
            if _CPP is not None:
                r_arr = np.ascontiguousarray(sols.astype(np.int32))
                _CPP.two_opt(r_arr,
                    np.ascontiguousarray(dist_mat[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(coords[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(tw_start_raw[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(tw_end_raw[i:i+1].astype(np.float32)),
                    np.ascontiguousarray(service_time[i:i+1].astype(np.float32)),
                    4, 50, tw_speed, np.random.randint(0, 2**31))
                sols = np.array(r_arr)
            else:
                sols = tw_aware_two_opt_py(sols, dist_mat[i:i+1],
                    demands_raw[i:i+1], capacity, 3.0, 4,
                    coords[i:i+1], tw_start_raw[i:i+1], tw_end_raw[i:i+1],
                    service_time[i:i+1], speed=tw_speed)

            # 评估
            cost = cvrp_eval_cost(sols, dist_mat[i:i+1],
                                  demands_raw[i:i+1], capacity, num_workers)[0]

            # TW 检查
            tw_feas, tw_viol = evaluate_tw_feasibility(
                sols, coords[i:i+1], tw_start_raw[i:i+1], tw_end_raw[i:i+1],
                service_time[i:i+1], speed=tw_speed
            )

            if cost < 1e9 and cost < best_cost:
                if tw_feas[0]:  # 优先选 TW 可行的
                    best_cost = cost
                    best_route = sols[0].copy()

        if best_cost < float(opt_costs[i]):
            improved_count += 1

        new_routes.append(best_route)
        new_costs.append(best_cost)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_instances} | improved: {improved_count} "
                  f"({improved_count/(i+1)*100:.1f}%) | "
                  f"avg cost: {np.mean(new_costs[:i+1]):.2f} vs orig {np.mean(opt_costs[:i+1]):.2f}")

    # 对齐 route 长度
    max_len = max(len(r) for r in new_routes)
    routes_padded = np.stack([
        np.pad(r, (0, max_len - len(r)), constant_values=0) if len(r) < max_len
        else r[:max_len]
        for r in new_routes
    ], axis=0).astype(np.int32)

    print(f"\nDone. Improved {improved_count}/{num_instances} "
          f"({improved_count/num_instances*100:.1f}%) instances.")
    print(f"  Avg cost: {np.mean(opt_costs):.2f} → {np.mean(new_costs):.2f} "
          f"({(np.mean(opt_costs) - np.mean(new_costs)) / np.mean(opt_costs) * 100:.1f}% lower)")

    return routes_padded, np.array(new_costs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--tw_speed', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--runs', type=int, default=8)
    parser.add_argument('--cycles', type=int, default=160)
    parser.add_argument('--two_opt_steps', type=int, default=4)
    args = parser.parse_args()

    # 加载模型
    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    try:
        model = nnx.merge(nnx.graphdef(model), params)
        print("Model loaded.")
    except Exception as e:
        print(f"WARNING: {e}")

    # 加载数据
    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    print(f"Data: {args.data} ({dataset['coords'].shape[0]} instances)")

    t0 = time.time()
    new_routes, new_costs = generate_pseudo_labels(
        dataset, args.capacity, model, tw_max, args.tw_speed,
        args.batch_size, args.runs, args.cycles, args.two_opt_steps,
    )

    # 保存
    out = dict(dataset)
    out['routes'] = new_routes
    out['opt_costs'] = new_costs
    np.savez_compressed(args.output, **out)
    print(f"Saved: {args.output} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
