"""JF1-H-F 的 service/hard audit（阶段 A：验证 hard-feasible baseline + 全过程状态守恒）。

在 DEV-CAL 9×128 上跑 JF1-H-F，逐实例统计显式 hard vector + repair 全过程审计：

hard vector（全 True 才算实例 hard-pass）：
  complete / n_unserved==0 / n_duplicate==0 / tw_feasible / capacity_feasible /
  depot_return_feasible / temperature_hard_feasible / all_orders_picked /
  all_cargo_delivered_to_depot / terminal_manifests_empty / trace_accounting_consistent /
  distance_accounting_consistent / terminal_unresolved==0 / ownership_violations==0

阶段 A Gate：全部实例 hard-pass（terminal_unresolved=0 + ownership_violations=0 + 冷链硬约束）。

产物：
  service_audit.json  —— cell 聚合 + all_pass
  manifest.json       —— 运行元数据 + 数据/代码/合同 hash + 每 cell D/Q/E 统计
  repair_events.jsonl —— 逐事件 repair 记录
  instances/<cell>__<idx>.json —— 逐实例 hard vector + 指标
"""
import argparse, os, json, sys, hashlib, time, datetime
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from coldchain_contract import default_pilot_contract, load_coldchain_contract
from coldchain_evaluator import evaluate_coldchain_trace


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _hard_vector(out, rp, served_mask):
    """显式 hard vector。每个键 True 才算实例 hard-pass。"""
    terminal_unresolved = sum(1 for c in rp.deferred_customers if not served_mask[int(c)])
    return {
        'complete': bool(out.get('complete')),
        'n_unserved_zero': int(out.get('n_unserved', -1)) == 0,
        'n_duplicate_zero': int(out.get('n_duplicate', -1)) == 0,
        'tw_feasible': bool(out.get('tw_feasible')),
        'capacity_feasible': bool(out.get('capacity_feasible')),
        'depot_return_feasible': bool(out.get('depot_return_feasible')),
        'temperature_hard_feasible': bool(out.get('temperature_hard_feasible')),
        'all_orders_picked': bool(out.get('all_orders_picked')),
        'all_cargo_delivered_to_depot': bool(out.get('all_cargo_delivered_to_depot')),
        'terminal_manifests_empty': bool(out.get('terminal_manifests_empty')),
        'trace_accounting_consistent': bool(out.get('trace_accounting_consistent')),
        'distance_accounting_consistent': bool(out.get('distance_accounting_consistent')),
        'terminal_unresolved_zero': int(terminal_unresolved) == 0,
        'ownership_violations_zero': int(rp.repair_stats['ownership_violations']) == 0,
    }


def audit_cell(data_path, capacity, num_vehicles, slack, out_dir, repair_fp, manifest_ctx,
               contract=None):
    dataset = dict(np.load(data_path))
    contract = contract or default_pilot_contract()
    rp = JF1HRepairReplanner(slack_vehicles=slack)
    env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                          replanner=rp, coldchain_contract=contract)
    n = dataset['coords'].shape[0]
    cell = os.path.splitext(os.path.basename(data_path))[0]

    hard_fail = 0
    terminal_unresolved = 0
    ownership_violations = 0
    attempts = successes = failures = 0
    deferred_then_reassigned = 0
    deferred_then_served = 0
    reasons = {}
    dist_incr = 0.0
    dists, quals, energies = [], [], []

    inst_dir = os.path.join(out_dir, 'instances')
    os.makedirs(inst_dir, exist_ok=True)

    for i in range(n):
        traces, served_mask = env.run(i)
        out = evaluate_coldchain_trace(traces, {
            'coords': env.coords[i], 'tw_start': env.tw_start[i], 'tw_end': env.tw_end[i],
            'service_time': env.service_time[i], 'demands': env.demands[i],
            'dist_mat': env.dist_mat[i], 'temp_class': env.temp_class[i],
            'initial_quality': env.initial_quality[i], 'speed': env.tw_speed,
        }, contract)

        hv = _hard_vector(out, rp, served_mask)
        inst_pass = all(hv.values())
        hard_fail += (not inst_pass)
        term = sum(1 for c in rp.deferred_customers if not served_mask[int(c)])
        terminal_unresolved += term
        ownership_violations += rp.repair_stats['ownership_violations']
        attempts += rp.repair_stats['attempts']
        successes += rp.repair_stats['successes']
        failures += rp.repair_stats['failures']
        deferred_then_reassigned += rp.repair_stats['deferred_then_reassigned']
        deferred_then_served += rp.repair_stats['deferred_then_served']
        for k, v in rp.repair_stats['reasons'].items():
            reasons[k] = reasons.get(k, 0) + v
        dist_incr += rp.repair_stats['repair_distance_increment']
        dists.append(float(out['distance_cost']))
        quals.append(float(out['quality_loss']))
        energies.append(float(out['energy_kwh']))

        # 逐实例产物
        with open(os.path.join(inst_dir, f'{cell}__{i}.json'), 'w') as f:
            json.dump({
                'scene_instance_id': f'{cell}__{i}', 'cell': cell, 'instance_index': i,
                'hard_pass': inst_pass, 'hard_vector': hv,
                'distance_cost': out['distance_cost'],
                'quality_loss': out['quality_loss'],
                'energy_kwh': out['energy_kwh'],
                'n_unserved': out.get('n_unserved'),
                'terminal_unresolved': term,
                'ownership_violations': rp.repair_stats['ownership_violations'],
                'repair_stats': dict(rp.repair_stats),
                'contract_hash': out.get('contract_hash'),
            }, f, indent=2)
        # repair_events.jsonl（逐事件）
        for ev in rp.repair_events:
            repair_fp.write(json.dumps({
                'scene_instance_id': f'{cell}__{i}', 'cell': cell, 'instance_index': i,
                **ev,
            }) + '\n')

    def _stats(xs):
        a = np.asarray(xs, dtype=float)
        return {
            'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            'median': float(np.median(a)),
            'IQR': float(np.percentile(a, 75) - np.percentile(a, 25)),
            'n': int(len(a)),
        }

    return {
        'data': os.path.basename(data_path), 'n': n,
        'n_hard_fail': hard_fail,
        'n_terminal_unresolved': terminal_unresolved,
        'n_ownership_violations': ownership_violations,
        'repair_attempts': attempts, 'repair_successes': successes,
        'repair_failures': failures,
        'deferred_then_reassigned': deferred_then_reassigned,
        'deferred_then_served': deferred_then_served,
        'reasons': reasons, 'repair_distance_increment': round(dist_incr, 4),
        'distance': _stats(dists), 'quality_loss': _stats(quals), 'energy_kwh': _stats(energies),
        'data_sha256': _sha256_file(data_path),
        'sample_count': n,
        'all_hard_pass': hard_fail == 0,
    }


def _validate_inputs(data_paths):
    """强制输入恰好 9 个唯一 DEV-CAL cell（R1/C1/RC1 × EDoD 0.2/0.5/0.8），每个 128 实例。"""
    expected = {f'dcc_50_{t}_edod{e}_devcal.npz'
                for t in ('r1', 'c1', 'rc1') for e in ('02', '05', '08')}
    names = [os.path.basename(p) for p in data_paths]
    if len(names) != len(set(names)):
        raise ValueError(f"重复 data 文件：{names}")
    got = set(names)
    if got != expected:
        raise ValueError(f"输入 cell 不符：missing={sorted(expected - got)} "
                         f"extra={sorted(got - expected)}")
    for p in data_paths:
        d = np.load(p)
        n = int(d['coords'].shape[0])
        d.close()
        if n != 128:
            raise ValueError(f"{os.path.basename(p)} 实例数={n}，期望 128")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--slack', type=int, default=1)
    parser.add_argument('--coldchain-contract', default=None,
                        help='calibrated contract JSON（缺省回退 pilot）')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    _validate_inputs(args.data)
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    os.makedirs(args.out, exist_ok=True)
    repair_fp = open(os.path.join(args.out, 'repair_events.jsonl'), 'w')
    started = time.time()

    results = []
    for d in args.data:
        r = audit_cell(d, args.capacity, args.num_vehicles, args.slack, args.out,
                       repair_fp, None, contract)
        results.append(r)
        print(f"  {r['data']}: hard_fail={r['n_hard_fail']}/{r['n']} "
              f"terminal_unresolved={r['n_terminal_unresolved']} "
              f"ownership_viol={r['n_ownership_violations']} "
              f"repair(att={r['repair_attempts']},succ={r['repair_successes']},"
              f"fail={r['repair_failures']},reass={r['deferred_then_reassigned']},"
              f"served={r['deferred_then_served']})", flush=True)

    repair_fp.close()

    total_hard = sum(r['n_hard_fail'] for r in results)
    total_unresolved = sum(r['n_terminal_unresolved'] for r in results)
    total_viol = sum(r['n_ownership_violations'] for r in results)
    all_pass = (total_hard == 0 and total_unresolved == 0 and total_viol == 0)
    print(f"\n  TOTAL: hard_fail={total_hard} terminal_unresolved={total_unresolved} "
          f"ownership_viol={total_viol} -> "
          f"{'STAGE-A PASS' if all_pass else 'FAIL'}", flush=True)

    # manifest.json
    _src = os.path.join(_CVRPTW, 'scripts')
    manifest = {
        'run': {
            'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
            'wall_seconds': round(time.time() - started, 2),
            'baseline': 'JF1-H-F',
            'slack_vehicles': args.slack,
            'capacity': args.capacity,
            'num_vehicles': args.num_vehicles,
        },
        'contract_hash': contract.contract_hash,
        'code_sha256': {
            'jf1h_repair': _sha256_file(os.path.join(_src, 'simulation', 'jf1h_repair.py')),
            'action_contract': _sha256_file(os.path.join(_src, 'simulation', 'action_contract.py')),
            'strict_online_env': _sha256_file(os.path.join(_src, 'simulation', 'strict_online_env.py')),
            'joint_fleet': _sha256_file(os.path.join(_src, 'simulation', 'joint_fleet.py')),
            'coldchain_state': _sha256_file(os.path.join(_src, 'coldchain', 'coldchain_state.py')),
            'coldchain_evaluator': _sha256_file(os.path.join(_src, 'evaluation', 'coldchain_evaluator.py')),
            'coldchain_contract': _sha256_file(os.path.join(_src, 'coldchain', 'coldchain_contract.py')),
            'audit_baseline_service': _sha256_file(os.path.join(_src, 'evaluation', 'audit_baseline_service.py')),
        },
        'cells': results,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.out, 'service_audit.json'), 'w') as f:
        json.dump({'all_pass': all_pass, 'total_hard_fail': total_hard,
                   'total_terminal_unresolved': total_unresolved,
                   'total_ownership_violations': total_viol,
                   'slack_vehicles': args.slack,
                   'cells': results}, f, indent=2)


if __name__ == '__main__':
    main()
