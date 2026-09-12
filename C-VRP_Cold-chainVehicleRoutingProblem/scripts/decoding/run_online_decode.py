"""
端到端验证：StrictOnlineDecoder + Resource Beam replanner 在真实 DCC 数据上运行。

验证严格 non-anticipatory 滚动时域：
  encode（动态可见性）→ decode → edge logits → Resource Beam generate_from_state
  → feasible suffix → execute until next event → freeze → 循环

用法:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u scripts/decoding/run_online_decode.py \
        --data data/baseline/50_node/test/dcc_50_r1_edod05_test.npz \
        --ckpt ckpts/p0_fix/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --num_instances 8 --beam_width 16 --capacity 50
"""

import sys, os, argparse, time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

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
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from online_decode import StrictOnlineDecoder, build_replanner
from strict_online_env import GreedyReplanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--method', type=str, default=None,
                        choices=['edd', 'nn', 'model', 'model_shuffle'],
                        help='H0=edd / H1=nn / H2=model / H3=model_shuffle')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='model/model_shuffle 必需；edd/nn 不需要')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=16)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--tw_speed', type=float, default=1.0)
    parser.add_argument('--decode_seed', type=int, default=42)
    # deprecated（向后兼容）
    parser.add_argument('--use_greedy', action='store_true', default=False,
                        help='[deprecated] 等价 --method edd')
    parser.add_argument('--incumbent_builder', type=str, default=None,
                        help='[deprecated] 用 --method 代替')
    args = parser.parse_args()

    # 兼容旧 flag
    if args.method is None:
        args.method = 'edd' if args.use_greedy else 'model'
    if args.method in ('model', 'model_shuffle') and args.ckpt is None:
        parser.error(f"--method {args.method} 需要 --ckpt")

    print(f"=== StrictOnlineDecoder | method={args.method} ===")
    print(f"  data: {args.data}")

    # 1. 加载数据
    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    print(f"  tw_max = {tw_max}")

    # 2. 加载模型（仅 model / model_shuffle）
    model = None
    if args.method in ('model', 'model_shuffle'):
        params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
        if model_config is None:
            model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
            model_config.encoder_input_dim = 7
        model_config.dtype = 'float32'
        model = model_config.construct_model()
        if params is None:
            raise RuntimeError(f"load_ckpt 返回空 params: {args.ckpt}")
        try:
            model = nnx.merge(nnx.graphdef(model), params)
            print("  Model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"checkpoint 合并失败: {args.ckpt}\n  原始错误: {e}\n"
                f"  请确认 ckpt 与模型结构/encoder_input_dim 一致。"
            )

    # 3. 构建 replanner
    replan_fn = build_replanner(args.method, dataset, args.capacity, model, tw_max,
                                args.tw_speed, args.beam_width, args.decode_seed)

    # 4. 构建 StrictOnlineDecoder
    decoder = StrictOnlineDecoder(
        dataset, capacity=args.capacity, model=model, tw_max=tw_max,
        tw_speed=args.tw_speed, beam_width=args.beam_width,
        replan_fn=replan_fn,
    )

    # 5. 跑 decode，汇总结果
    n = min(args.num_instances, decoder.num_instances)
    results = []
    t0 = time.time()
    for inst_idx in range(n):
        route, metrics = decoder.decode_instance(inst_idx)
        results.append(metrics)
        print(f"  [{inst_idx+1}/{n}] complete={metrics['complete']} "
              f"cost={metrics['distance_cost']:.2f} "
              f"cap_feas={metrics['capacity_feasible']} "
              f"tw_feas={metrics['tw_feasible']} "
              f"vehicles={metrics['vehicle_count']}")

    elapsed = time.time() - t0

    # 6. 汇总
    complete_rate = np.mean([r['complete'] for r in results])
    cap_feas_rate = np.mean([r['capacity_feasible'] for r in results])
    tw_feas_rate = np.mean([r['tw_feasible'] for r in results])
    depot_return_rate = np.mean([r['depot_return_feasible'] for r in results])
    mean_cost = np.mean([r['distance_cost'] for r in results])
    mean_unserved = np.mean([r['n_unserved'] for r in results])
    mean_vehicles = np.mean([r['vehicle_count'] for r in results])

    print(f"\n{'='*60}")
    print(f"StrictOnlineDecoder 结果（strict non-anticipatory）")
    print(f"{'='*60}")
    print(f"  instances:      {n}")
    print(f"  complete rate:  {complete_rate:.1%}")
    print(f"  cap feas rate:  {cap_feas_rate:.1%}")
    print(f"  TW feas rate:   {tw_feas_rate:.1%}")
    print(f"  depot return:   {depot_return_rate:.1%}")
    print(f"  mean cost:      {mean_cost:.2f}")
    print(f"  mean unserved:  {mean_unserved:.2f}")
    print(f"  mean vehicles:  {mean_vehicles:.2f}")
    print(f"  elapsed:        {elapsed:.1f}s")


if __name__ == '__main__':
    main()
