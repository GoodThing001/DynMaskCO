"""
HFR-M0 Step 5 — VAL128 Gate A：JF1-H vs HFR-M0（downstream utility）。

同 strict-online skeleton / candidate builder / guard / decode seed，唯一变量 = fleet partition
+ route reconstruction scorer（JF1-H min-travel vs HFR-M0 hierarchical G→A）。输出 service +
cost 的 paired 比较，判断 Gate A（§47-50）。

与训练一致：HFR forward 用 edge_feat=None + adjmat=None + timestep=0（train_hfr_m0.py 已对齐）。

用法（服务器）：
    python scripts/evaluation/run_hfr_gate_a.py \
      --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
      --base_ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
      --group_ckpt results/hfr/m0_seed42/group_head.ckpt \
      --adapter_ckpt results/hfr/m0_seed42/route_adapter.ckpt \
      --num_instances 128 --out results/hfr/m0_gate_a
"""
import sys, os, argparse, time, pickle, csv
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.dirname(_BASE)
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))

from flax import nnx
from strict_online_env import StrictOnlineEnv
from authoritative_evaluator import evaluate_execution_trace
from joint_fleet import JointAssignmentReplanner
from hfr_replanner import HFRReplanner
from hierarchical_fleet_route import GroupingHead, RouteResidualAdapter


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


def load_head(ckpt_path, cls):
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    head = cls(d_model=ck['d_model'], hidden=ck['hidden'], rngs=nnx.Rngs(0))
    return nnx.merge(nnx.graphdef(head), ck['params'])


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
    parser.add_argument('--group_ckpt', required=True)
    parser.add_argument('--adapter_ckpt', required=True)
    parser.add_argument('--num_instances', type=int, default=128)
    parser.add_argument('--beta', type=float, default=None,
                        help='A_joint 的 log_sigmoid(G) 权重；None=读 group_ckpt 同目录 config.json')
    parser.add_argument('--partition_mode', type=str, default='group', choices=['group', 'travel'],
                        help='2×2 消融：assignment 的 partition 项用 GroupAffinity 还是 min-travel')
    parser.add_argument('--insert_mode', type=str, default='affinity',
                        choices=['affinity', 'travel', 'group_only', 'group_logit'],
                        help='插入/排序增益：A_joint / cheapest-insertion / A_group(ΔA+logσ(G)) / 纯 logσ(G)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    n = min(args.num_instances, dataset['coords'].shape[0])

    base_model = load_base_model(args.base_ckpt)
    group_head = load_head(args.group_ckpt, GroupingHead)
    route_adapter = load_head(args.adapter_ckpt, RouteResidualAdapter)

    beta = args.beta
    if beta is None:
        cfg_path = os.path.join(os.path.dirname(args.group_ckpt), 'config.json')
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as f:
                beta = json.load(f).get('beta', 1.0)
        else:
            beta = 1.0

    jf1h = JointAssignmentReplanner('heuristic')
    hfr = HFRReplanner(dataset, 50, tw_max=tw_max, tw_speed=1.0,
                       model=base_model, group_head=group_head,
                       route_adapter=route_adapter, beta=beta,
                       partition_mode=args.partition_mode, insert_mode=args.insert_mode)

    print(f"=== JF1-H ===", flush=True)
    t0 = time.time()
    r1 = run_method(jf1h, dataset, n)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"=== HFR-M0 (beta={beta}, partition={args.partition_mode}, "
          f"insert={args.insert_mode}) ===", flush=True)
    t0 = time.time()
    r2 = run_method(hfr, dataset, n)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    c1 = float(np.mean([m['distance_cost'] for m in r1 if m['complete']]))
    c2 = float(np.mean([m['distance_cost'] for m in r2 if m['complete']]))
    comp1 = float(np.mean([m['complete'] for m in r1]))
    comp2 = float(np.mean([m['complete'] for m in r2]))
    un1 = float(np.mean([m['n_unserved'] for m in r1]))
    un2 = float(np.mean([m['n_unserved'] for m in r2]))

    deltas = [r2[i]['distance_cost'] - r1[i]['distance_cost'] for i in range(n)]
    improved = sum(1 for d in deltas if d < -1e-6)
    equal = sum(1 for d in deltas if abs(d) <= 1e-6)
    worsened = sum(1 for d in deltas if d > 1e-6)
    med_delta = float(np.median(deltas))

    print(f"\n=== HFR-M0 VAL128 Gate A ===")
    print(f"  JF1-H   : cost={c1:.4f} complete={comp1:.1%} unserved={un1:.3f}")
    print(f"  HFR-M0  : cost={c2:.4f} complete={comp2:.1%} unserved={un2:.3f}")
    print(f"  paired delta (HFR - JF1-H): mean={np.mean(deltas):+.4f} median={med_delta:+.4f}")
    print(f"  improved={improved} equal={equal} worsened={worsened}")
    gate_service = "PASS" if (comp2 == 1.0 and un2 == 0.0) else "FAIL"
    gate_cost = "PASS" if c2 < c1 else "FAIL"
    print(f"  SERVICE GATE: {gate_service}")
    print(f"  COST GATE A : {gate_cost}  ({c2:.4f} vs {c1:.4f})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'gate_a.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'jf1h_cost', 'hfr_cost', 'jf1h_complete', 'hfr_complete',
                    'jf1h_unserved', 'hfr_unserved', 'delta'])
        for i in range(n):
            w.writerow([i, f"{r1[i]['distance_cost']:.4f}", f"{r2[i]['distance_cost']:.4f}",
                        int(r1[i]['complete']), int(r2[i]['complete']),
                        r1[i]['n_unserved'], r2[i]['n_unserved'], f"{deltas[i]:+.4f}"])
    with open(os.path.join(args.out, 'gate_a_summary.txt'), 'w') as f:
        f.write(f"JF1-H cost={c1:.4f} complete={comp1:.1%}\n")
        f.write(f"HFR-M0 cost={c2:.4f} complete={comp2:.1%}\n")
        f.write(f"beta={beta} partition={args.partition_mode} insert={args.insert_mode}\n")
        f.write(f"paired delta mean={np.mean(deltas):+.4f} median={med_delta:+.4f}\n")
        f.write(f"improved={improved} equal={equal} worsened={worsened}\n")
        f.write(f"SERVICE GATE={gate_service} COST GATE A={gate_cost}\n")
    print(f"  saved: {args.out}/gate_a.csv + gate_a_summary.txt")


if __name__ == '__main__':
    main()
