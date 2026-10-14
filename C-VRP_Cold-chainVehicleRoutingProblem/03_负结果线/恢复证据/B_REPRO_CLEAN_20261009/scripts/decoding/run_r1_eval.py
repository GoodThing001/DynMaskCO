"""
R1: strict-online baseline 矩阵评估（D3 正式评估）

用 StrictOnlineDecoder（事件驱动滚动时域 + visible incumbent → mask → reconstruct
→ resource beam）在 5 train seed × 3 type × 3 EDoD = 45 cells 上评估，输出 CSV。

这是新协议的正式评估口径（区别于 cvrptw.py 的 offline 单发评估）。

用法:
    python scripts/decoding/run_r1_eval.py --mode smoke   # 16 实例/cell 冒烟
    python scripts/decoding/run_r1_eval.py --mode full    # 128 实例/cell 正式

输出:
    results/r1_baseline/strict_online_matrix.csv
    (列: type, edod, train_seed, complete, cap_feas, tw_feas, depot_return,
          cost, unserved, vehicles)
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

from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from online_decode import StrictOnlineDecoder, build_maskco_replanner


def load_model(ckpt_path):
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError(f"load_ckpt 返回空 params: {ckpt_path}")
    try:
        model = nnx.merge(nnx.graphdef(model), params)
    except Exception as e:
        # P1 #17：ckpt 与模型结构不匹配时，绝不静默用随机模型跑 R1 正式评估
        raise RuntimeError(
            f"checkpoint 合并失败（params 与 model 结构不匹配）: {ckpt_path}\n"
            f"  原始错误: {e}\n"
            f"  请确认 --use_edge 与训练时 USE_EDGE 一致（typed_v1_edge vs typed_v1）。"
        )
    return model


def eval_cell(data_path, ckpt_path, capacity, beam_width, num_instances, tw_speed=1.0, decode_seed=42):
    dataset = dict(np.load(data_path))
    tw_max = float(dataset['tw_end'].max())
    model = load_model(ckpt_path)
    replan_fn = build_maskco_replanner(dataset, capacity, model, tw_max, tw_speed, beam_width,
                                       decode_seed=decode_seed)
    decoder = StrictOnlineDecoder(
        dataset, capacity=capacity, model=model, tw_max=tw_max,
        tw_speed=tw_speed, beam_width=beam_width, replan_fn=replan_fn,
    )
    n = min(num_instances, decoder.num_instances)
    metrics = []
    for inst_idx in range(n):
        route, m = decoder.decode_instance(inst_idx)
        metrics.append(m)
    agg = {
        'complete': np.mean([m['complete'] for m in metrics]),
        'cap_feas': np.mean([m['capacity_feasible'] for m in metrics]),
        'tw_feas': np.mean([m['tw_feasible'] for m in metrics]),
        'depot_return': np.mean([m['depot_return_feasible'] for m in metrics]),
        'cost': np.mean([m['distance_cost'] for m in metrics]),
        'unserved': np.mean([m['n_unserved'] for m in metrics]),
        'vehicles': np.mean([m['vehicle_count'] for m in metrics]),
    }
    return agg, metrics


def main():
    parser = argparse.ArgumentParser(description='R1 strict-online baseline matrix eval')
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'full'])
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='val=开发/模型选择（默认），test=方法冻结后一次性正式报告')
    parser.add_argument('--decode_seed', type=int, default=42,
                        help='固定 decode RNG（P1-2，可复现）')
    parser.add_argument('--ckpt_step', type=int, default=50000,
                        help='checkpoint 步数（50000=正式，2000=smoke 流程验证）')
    parser.add_argument('--root', type=str,
                        default='/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
    parser.add_argument('--use_edge', type=int, default=1,
                        help='1=typed_v1_edge checkpoint, 0=typed_v1')
    parser.add_argument('--run_tag', type=str, default='r1_5_baseline',
                        help='checkpoint 目录版本（r1_5_baseline=干净重训，r1_baseline=legacy）')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=16)
    parser.add_argument('--tw_speed', type=float, default=1.0)
    args = parser.parse_args()

    data_dir = os.path.join(args.root, 'data', 'baseline', '50_node', args.split)
    tag = 'typed_v1_edge' if args.use_edge else 'typed_v1'
    ckpt_dir = os.path.join(args.root, 'ckpts', args.run_tag, tag, 'phase3c')
    out_dir = os.path.join(args.root, 'results', 'r1_baseline')
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 123, 999, 2025, 2026]
    types = ['r1', 'c1', 'rc1']
    edods = ['02', '05', '08']
    n_inst = 16 if args.mode == 'smoke' else 128

    csv_path = os.path.join(out_dir, 'strict_online_matrix.csv')
    header = ('type,edod,train_seed,complete,cap_feas,tw_feas,depot_return,'
              'cost,unserved,vehicles')
    with open(csv_path, 'w') as f:
        f.write(header + '\n')

    inst_csv_path = os.path.join(out_dir, 'strict_online_instances.csv')
    inst_header = ('type,edod,train_seed,decode_seed,instance_id,'
                   'complete,tw_feasible,cap_feasible,depot_return,'
                   'unserved,late,distance,vehicles')
    with open(inst_csv_path, 'w') as f:
        f.write(inst_header + '\n')

    print(f"=== R1 strict-online matrix eval | mode={args.mode} | tag={tag} ===")
    print(f"  {len(seeds)} seeds × {len(types)} types × {len(edods)} EDoDs = "
          f"{len(seeds)*len(types)*len(edods)} cells, {n_inst} inst/cell")

    t0 = time.time()
    n_cells = 0
    for seed in seeds:
        ck = os.path.join(ckpt_dir, f'seed{seed}', f'step{args.ckpt_step}.ckpt')
        if not os.path.exists(ck):
            if args.mode == 'full':
                raise FileNotFoundError(
                    f"Formal R1 requires all 5 seeds: missing {ck}")
            print(f"  [SKIP] seed={seed} no ckpt: {ck}")
            continue
        for t in types:
            for e in edods:
                data = os.path.join(data_dir, f'dcc_50_{t}_edod{e}_{args.split}.npz')
                print(f"  [{t} edod={e} seed={seed}] ...", flush=True)
                agg, per_inst = eval_cell(data, ck, args.capacity, args.beam_width,
                                          n_inst, args.tw_speed, args.decode_seed)
                row = (f"{t.upper()},{float(e)/10},{seed},"
                       f"{agg['complete']:.3f},{agg['cap_feas']:.3f},"
                       f"{agg['tw_feas']:.3f},{agg['depot_return']:.3f},"
                       f"{agg['cost']:.2f},{agg['unserved']:.2f},{agg['vehicles']:.2f}")
                with open(csv_path, 'a') as f:
                    f.write(row + '\n')
                with open(inst_csv_path, 'a') as f:
                    for iid, m in enumerate(per_inst):
                        f.write(
                            f"{t.upper()},{float(e)/10},{seed},{args.decode_seed},{iid},"
                            f"{int(m['complete'])},{int(m['tw_feasible'])},"
                            f"{int(m['capacity_feasible'])},{int(m['depot_return_feasible'])},"
                            f"{m['n_unserved']},{m['tw_late_count']},"
                            f"{m['distance_cost']:.2f},{m['vehicle_count']}\n")
                print(f"      complete={agg['complete']:.1%} tw_feas={agg['tw_feas']:.1%} "
                      f"cost={agg['cost']:.2f} unserved={agg['unserved']:.2f}")
                n_cells += 1

    print(f"\n=== DONE: {n_cells} cells in {time.time()-t0:.0f}s ===")
    print(f"  CSV: {csv_path}")

    # ═══════════════════════════════════════════════════════
    # 汇总：按 type × EDoD 聚合（mean ± std over train seeds）
    # ═══════════════════════════════════════════════════════
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.shape[0] > 0:
            g = df.groupby(['type', 'edod']).agg(
                cost_mean=('cost', 'mean'), cost_std=('cost', 'std'),
                complete=('complete', 'mean'),
                tw_feas=('tw_feas', 'mean'),
                cap_feas=('cap_feas', 'mean'),
                depot_return=('depot_return', 'mean'),
                unserved=('unserved', 'mean'),
                vehicles=('vehicles', 'mean'),
                n_seed=('train_seed', 'nunique'),
            ).reset_index()
            print("\n=== R1 strict-online 汇总（5-seed mean ± std）===")
            print(f"{'Type':<6} {'EDoD':<6} {'Cost':>14} {'Complete':>9} "
                  f"{'TW':>7} {'Cap':>7} {'Depot':>7} {'Unserv':>7}")
            print('-' * 70)
            for _, r in g.iterrows():
                print(f"{r['type']:<6} {r['edod']:<6} "
                      f"{r['cost_mean']:>7.2f} ±{r['cost_std']:>5.2f} "
                      f"{r['complete']:>8.1%} {r['tw_feas']:>6.1%} "
                      f"{r['cap_feas']:>6.1%} {r['depot_return']:>6.1%} "
                      f"{r['unserved']:>6.2f}")
            # overall
            ov = df.groupby('train_seed').agg(cost=('cost', 'mean')).reset_index()
            print('-' * 70)
            print(f"{'ALL':<6} {'':<6} {ov['cost'].mean():>7.2f} ±{ov['cost'].std():>5.2f} "
                  f"{df['complete'].mean():>8.1%} {df['tw_feas'].mean():>6.1%} "
                  f"{df['cap_feas'].mean():>6.1%} {df['depot_return'].mean():>6.1%} "
                  f"{df['unserved'].mean():>6.2f}")
            print(f"\n  (n_seed={df['train_seed'].nunique()}, cells={df.shape[0]})")
    except Exception as e:
        print(f"\n  [WARN] 汇总失败: {e}")


if __name__ == '__main__':
    main()
