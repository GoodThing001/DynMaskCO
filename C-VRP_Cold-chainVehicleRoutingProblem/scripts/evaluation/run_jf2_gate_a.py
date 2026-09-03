"""
JF2 Phase 3C — VAL128 Gate A：JF1-H vs JF2-M0-v2（downstream utility）。

同 strict-online skeleton / candidate builder / sequencing / guard / decode seed，唯一变量 =
assignment score。输出 service + cost 的 paired 比较，判断 Gate A。

注意：M0-v2 训练时 encode H 用 edge_feat=None（无 energy_mat 边特征），故 eval 也去掉
energy_mat 保持一致（head 是在无 edge 特征的 H 上训练的）。

用法（服务器）：
    python scripts/evaluation/run_jf2_gate_a.py \
      --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
      --base_ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
      --head_ckpt results/jf2/m0_seed42/fleet_head.ckpt \
      --num_instances 128 --out results/jf2/m0_v2_gate_a
"""
import sys, os, argparse, time, pickle, csv
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))

from flax import nnx
from strict_online_env import StrictOnlineEnv
from authoritative_evaluator import evaluate_execution_trace
from joint_fleet import JointAssignmentReplanner
from jf2 import JF2Replanner
from fleet_assignment_head import FleetAssignmentHead
from fleet_features import Fv, Fp


def load_base_model(ckpt_path):
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError(f"load_ckpt 返回空 params: {ckpt_path}")
    return nnx.merge(nnx.graphdef(model), params)


def load_fleet_head(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    head = FleetAssignmentHead(embed_dim=ck['embed_dim'], veh_dim=ck['veh_dim'],
                               pair_dim=ck['pair_dim'], hidden_dim=ck['hidden_dim'],
                               rngs=nnx.Rngs(0))
    head = nnx.merge(nnx.graphdef(head), ck['params'])
    return head, ck.get('alpha', 1.0)


def run_method(replanner, dataset, n):
    per = []
    for i in range(n):
        env = StrictOnlineEnv(dataset, 50, 1.0, 25, replanner=replanner)
        traces, served = env.run(i)
        m = evaluate_execution_trace(
            traces, env.coords[i], env.tw_start[i], env.tw_end[i],
            env.service_time[i], env.demands[i], 50, speed=1.0, dist_mat=env.dist_mat[i])
        per.append(m)
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{n}", flush=True)
    return per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--base_ckpt', required=True)
    parser.add_argument('--head_ckpt', required=True)
    parser.add_argument('--num_instances', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=None,
                        help='覆盖 head ckpt 的 alpha（residual 校正 sweep 用）')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    n = min(args.num_instances, dataset['coords'].shape[0])

    base_model = load_base_model(args.base_ckpt)
    head, alpha = load_fleet_head(args.head_ckpt)
    if args.alpha is not None:
        alpha = args.alpha

    # M0-v2 训练用 edge_feat=None，eval 也去掉 energy_mat 保持一致
    dataset_no_edge = {k: v for k, v in dataset.items() if k != 'energy_mat'}

    jf1h = JointAssignmentReplanner('heuristic')
    jf2 = JF2Replanner(dataset_no_edge, 50, tw_max=tw_max, tw_speed=1.0,
                       model=base_model, fleet_head=head, alpha=alpha)

    print(f"=== JF1-H ===", flush=True)
    t0 = time.time()
    r1 = run_method(jf1h, dataset_no_edge, n)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"=== JF2-M0-v2 (alpha={alpha}) ===", flush=True)
    t0 = time.time()
    r2 = run_method(jf2, dataset_no_edge, n)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    def mean(xs): return float(np.mean(xs))
    c1 = mean([m['distance_cost'] for m in r1 if m['complete']])
    c2 = mean([m['distance_cost'] for m in r2 if m['complete']])
    comp1 = mean([m['complete'] for m in r1])
    comp2 = mean([m['complete'] for m in r2])
    un1 = mean([m['n_unserved'] for m in r1])
    un2 = mean([m['n_unserved'] for m in r2])

    deltas = [r2[i]['distance_cost'] - r1[i]['distance_cost'] for i in range(n)]
    improved = sum(1 for d in deltas if d < -1e-6)
    equal = sum(1 for d in deltas if abs(d) <= 1e-6)
    worsened = sum(1 for d in deltas if d > 1e-6)
    med_delta = float(np.median(deltas))

    print(f"\n=== JF2-M0-v2 VAL128 Gate A ===")
    print(f"  JF1-H      : cost={c1:.4f} complete={comp1:.1%} unserved={un1:.3f}")
    print(f"  JF2-M0-v2  : cost={c2:.4f} complete={comp2:.1%} unserved={un2:.3f}")
    print(f"  paired delta (JF2 - JF1-H): mean={np.mean(deltas):+.4f} median={med_delta:+.4f}")
    print(f"  improved={improved} equal={equal} worsened={worsened}")
    gate_a_cost = "PASS" if c2 < 24.5019 else "FAIL"
    gate_a_service = "PASS" if (comp2 == 1.0 and un2 == 0.0) else "FAIL"
    print(f"  SERVICE GATE: {gate_a_service}")
    print(f"  COST GATE A : {gate_a_cost}  ({c2:.4f} vs 24.5019)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'gate_a.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'jf1h_cost', 'jf2_cost', 'jf1h_complete', 'jf2_complete',
                    'jf1h_unserved', 'jf2_unserved', 'delta'])
        for i in range(n):
            w.writerow([i, f"{r1[i]['distance_cost']:.4f}", f"{r2[i]['distance_cost']:.4f}",
                        int(r1[i]['complete']), int(r2[i]['complete']),
                        r1[i]['n_unserved'], r2[i]['n_unserved'], f"{deltas[i]:+.4f}"])
    with open(os.path.join(args.out, 'gate_a_summary.txt'), 'w') as f:
        f.write(f"JF1-H cost={c1:.4f} complete={comp1:.1%}\n")
        f.write(f"JF2 cost={c2:.4f} complete={comp2:.1%}\n")
        f.write(f"paired delta mean={np.mean(deltas):+.4f} median={med_delta:+.4f}\n")
        f.write(f"improved={improved} equal={equal} worsened={worsened}\n")
        f.write(f"SERVICE GATE={gate_a_service} COST GATE A={gate_a_cost}\n")
    print(f"  saved: {args.out}/gate_a.csv + gate_a_summary.txt")


if __name__ == '__main__':
    main()
