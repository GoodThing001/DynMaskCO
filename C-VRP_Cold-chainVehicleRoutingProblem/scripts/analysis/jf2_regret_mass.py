"""
JF2 Regret Mass Decomposition（Phase 3C Branch C1 诊断，Layer 1：state-level counterfactual）。

回答导师 01 文档 §8 的问题：1866 个真实非等价 disagreement 的 downstream regret 是否高度集中
（少量高杠杆决策承担大部分 regret）？这决定下一步走 regret-weighted preference 还是 partition。

对每个真实分歧 (j, k_H=min-travel 车, k*=OR teacher 车)：
  J*         = 当前 FleetState 下 OR-joint 的 recourse cost（base，teacher 即 k*）
  J(j→k_H)   = 强制 j 归 k_H（k_H anchor 移到「刚服务完 j」），再 OR 优化剩余 mutable assignment
  R_j        = J(j→k_H) - J*     （>0 = OR 选择更好；<0 = OR label 在 counterfactual 口径下不优于 JF1-H）

输出（按导师要求拆正负）：
  N disagreement / N positive / near-zero / negative
  mean/median/p90/p95/p99 signed regret
  mean positive regret / mean negative regret
  Top 1/5/10/20% positive-regret mass（Lorenz concentration）
  Σ R+ / Σ |R|  （disagreement 集合里真正「可学习的正价值」占比）

用法（服务器，需 OR-Tools）：
    python scripts/analysis/jf2_regret_mass.py \
        --states results/jf2/data/r1_edod05/states_train_0.npz \
        --teacher results/jf2/data/r1_edod05/teacher_vehicle.npz \
        --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
        --time_limit_ms 50 --max_disagreements -1 \
        --out results/jf2/regret_mass
"""
import sys, os, argparse, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'baselines'))

from strict_online_env import StrictOnlineEnv, VehicleState
from ortools_rolling_horizon import ORToolsReplanner

STATUS_INV = {0: 'idle', 1: 'ready', 2: 'committed', 3: 'returning', 4: 'closed'}
PENALTY = 1e6  # counterfactual infeasible 的 regret 惩罚


def reconstruct_vehicles(a, s, K, N):
    vs = []
    for k in range(K):
        tail = [int(j) for j in range(N) if a['tail_mask'][s, k, j]]
        vs.append(VehicleState(
            vehicle_id=k, status=STATUS_INV.get(int(a['vehicle_status'][s, k]), 'closed'),
            current_node=int(a['vehicle_node'][s, k]), ready_time=float(a['vehicle_ready'][s, k]),
            current_load=float(a['vehicle_load'][s, k]),
            committed_next=(None if int(a['vehicle_comm_next'][s, k]) == -1
                            else int(a['vehicle_comm_next'][s, k])),
            committed_finish=(None if a['vehicle_comm_finish'][s, k] == -1.0
                              else float(a['vehicle_comm_finish'][s, k])),
            mutable_suffix=tail,
        ))
    return vs


def visible_ids_of(dataset, inst_idx, clock):
    N = dataset['coords'].shape[1]
    return [i for i in range(1, N) if dataset['demands'][inst_idx, i] > 0
            and dataset['reveal_time'][inst_idx, i] <= clock + 1e-6]


def replan_ids_of(a, s, K):
    return {k for k in range(K) if a['replan_mask'][s, k]}


def route_cost(env, inst_idx, vehicles, active_ids):
    """OR 解的总 recourse cost = active 车 route 距离和（不含已服务/committed 历史）。"""
    total = 0.0
    for v in vehicles:
        if v.vehicle_id not in active_ids:
            continue
        prev = int(v.current_node)
        for n in v.mutable_suffix:
            total += float(env.dist_mat[inst_idx, prev, int(n)])
            prev = int(n)
    return total


def run_or(env, rp, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids):
    rp.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
    cost = route_cost(env, inst_idx, vehicles, replan_ids)
    # 剩余 mutable 客户是否都被服务（infeasible 检测）
    reserved = env.get_reserved_customers(vehicles)
    remaining = [i for i in visible_ids if not served_mask[i] and i not in reserved]
    served_now = set()
    for v in vehicles:
        for n in v.mutable_suffix:
            if n != 0:
                served_now.add(int(n))
    unserved = [i for i in remaining if i not in served_now]
    return cost, len(unserved)


def commit_customer(env, inst_idx, v, j):
    """把车辆 v 的 anchor 更新为「刚服务完客户 j」（冻结 j→v 这个 ownership 决策）。
    返回 anchor→j 的 travel distance（counterfactual cost 需补这段，否则漏算）。"""
    orig = int(v.current_node)
    travel = float(env.dist_mat[inst_idx, orig, int(j)]) / env.tw_speed
    arrival = max(float(v.ready_time) + travel, float(env.tw_start[inst_idx, int(j)]))
    finish = arrival + float(env.service_time[inst_idx, int(j)])
    v.current_node = int(j)
    v.ready_time = finish
    v.current_load += float(env.demands[inst_idx, int(j)])
    return float(env.dist_mat[inst_idx, orig, int(j)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--time_limit_ms', type=int, default=50)
    parser.add_argument('--max_disagreements', type=int, default=-1, help='-1 = 全部')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    a = dict(np.load(args.states))
    teacher = np.load(args.teacher)['teacher_vehicle']
    dataset = dict(np.load(args.data))
    K = a['vehicle_status'].shape[1]
    N = dataset['coords'].shape[1]
    S = a['instance_id'].shape[0]

    env0 = StrictOnlineEnv(dataset, 50, 1.0, K, replanner=None)
    rp = ORToolsReplanner(50, K, args.time_limit_ms)

    anchor_ids = a['anchor_ids']
    veh_feat = a['veh_feat']
    base_score = a['base_score']
    cand_mask = a['candidate_mask']

    # 等价判断（对称 audit 同款）
    def equiv(s, k1, k2):
        return (anchor_ids[s, k1] == anchor_ids[s, k2]
                and abs(veh_feat[s, k1, 0] - veh_feat[s, k2, 0]) < 1e-4
                and abs(veh_feat[s, k1, 1] - veh_feat[s, k2, 1]) < 1e-4)

    regrets = []
    n_infeasible = 0   # 强制 min-travel 后剩余客户无法全部服务的 case（高杠杆，单独计）
    n_processed = 0
    for s in range(S):
        inst = int(a['instance_id'][s]); clock = float(a['clock'][s])
        min_travel = np.argmax(np.where(cand_mask[s], base_score[s], -1e9), axis=0)  # [N] 每客户 min-travel 车辆 id
        # 找 true disagreement（teacher != min-travel 且非等价）
        disagr = []
        for j in range(1, N):
            kt = int(teacher[s, j])
            if kt == -1:
                continue
            km = int(min_travel[j])
            if kt == km:
                continue
            if equiv(s, kt, km):
                continue
            disagr.append((j, km, kt))
        if not disagr:
            continue

        served = a['served_mask'][s]
        vis = visible_ids_of(dataset, inst, clock)
        rp_ids = replan_ids_of(a, s, K)

        # base OR（J*）
        vehicles = reconstruct_vehicles(a, s, K, N)
        Jstar, _ = run_or(env0, rp, inst, clock, vehicles, served.copy(), vis, rp_ids)

        # 每个 true disagreement 的 counterfactual
        for (j, km, kt) in disagr:
            if args.max_disagreements >= 0 and n_processed >= args.max_disagreements:
                break
            v2 = reconstruct_vehicles(a, s, K, N)
            # 冻结 j -> km（min-travel 车），补 anchor→j 的 travel 到 cost
            commit_travel = commit_customer(env0, inst, v2[km], j)
            s2 = served.copy(); s2[j] = True
            J_km, unserved = run_or(env0, rp, inst, clock, v2, s2, vis, rp_ids)
            J_km = J_km + commit_travel
            if unserved > 0:
                n_infeasible += 1   # 高杠杆：min-travel 破坏剩余可行性，单独计
            else:
                regrets.append(float(J_km - Jstar))
            n_processed += 1
            if args.max_disagreements >= 0 and n_processed >= args.max_disagreements:
                break
        if args.max_disagreements >= 0 and n_processed >= args.max_disagreements:
            break
        if (s + 1) % 500 == 0:
            print(f"  processed {s+1}/{S} states, {n_processed} disagreements", flush=True)

    regrets = np.array(regrets)
    n_feasible = len(regrets)
    n_total = n_feasible + n_infeasible
    if n_total == 0:
        print("no true disagreement found"); return

    pos = regrets[regrets > 1e-6]
    neg = regrets[regrets < -1e-6]
    near0 = regrets[np.abs(regrets) <= 1e-6]
    Rpos = np.maximum(regrets, 0.0)

    def pct(x, q): return float(np.percentile(x, q)) if len(x) else float('nan')
    # Lorenz concentration：top p% 的 positive regret 占总 positive regret 的比例（仅 feasible）
    order = np.argsort(-Rpos)
    cum = np.cumsum(Rpos[order])
    total_pos = cum[-1] if len(cum) else 0.0
    def top_mass(p):
        k = max(1, int(np.ceil(n_feasible * p / 100.0)))
        return float(cum[k - 1] / total_pos) if total_pos > 0 else float('nan')

    print(f"\n=== JF2 Regret Mass Decomposition（{n_total} true disagreements）===")
    print(f"  N infeasible (min-travel 破坏剩余可行性): {n_infeasible} ({n_infeasible/n_total:.1%})")
    print(f"  N feasible  : {n_feasible} ({n_feasible/n_total:.1%})")
    print(f"    ├─ positive-regret : {len(pos)} ({len(pos)/n_feasible:.1%})")
    print(f"    ├─ near-zero       : {len(near0)} ({len(near0)/n_feasible:.1%})")
    print(f"    └─ negative-regret : {len(neg)} ({len(neg)/n_feasible:.1%})")
    print(f"  [feasible] mean signed   : {regrets.mean():+.4f}")
    print(f"  [feasible] median signed : {np.median(regrets):+.4f}")
    print(f"  [feasible] p90/p95/p99   : {pct(regrets,90):+.4f} / {pct(regrets,95):+.4f} / {pct(regrets,99):+.4f}")
    print(f"  [feasible] mean positive : {pos.mean():.4f}" if len(pos) else "  [feasible] mean positive : nan")
    print(f"  [feasible] mean negative : {neg.mean():.4f}" if len(neg) else "  [feasible] mean negative : nan")
    print(f"  [feasible] Top 1%/5%/10%/20% positive mass : "
          f"{top_mass(1):.1%} / {top_mass(5):.1%} / {top_mass(10):.1%} / {top_mass(20):.1%}")
    print(f"  [feasible] Σ R+ / Σ |R| = {Rpos.sum()/np.abs(regrets).sum():.1%}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'regret_mass_summary.json'), 'w') as f:
        json.dump({
            'n_total': n_total, 'n_infeasible': int(n_infeasible), 'n_feasible': int(n_feasible),
            'n_positive': int(len(pos)), 'n_nearzero': int(len(near0)), 'n_negative': int(len(neg)),
            'mean_signed': float(regrets.mean()), 'median_signed': float(np.median(regrets)),
            'p90': pct(regrets, 90), 'p95': pct(regrets, 95), 'p99': pct(regrets, 99),
            'mean_positive': (float(pos.mean()) if len(pos) else None),
            'mean_negative': (float(neg.mean()) if len(neg) else None),
            'top1': top_mass(1), 'top5': top_mass(5), 'top10': top_mass(10), 'top20': top_mass(20),
            'pos_ratio': float(Rpos.sum() / np.abs(regrets).sum()),
        }, f, indent=2)
    np.save(os.path.join(args.out, 'regrets.npy'), regrets)
    print(f"  saved: {args.out}/regret_mass_summary.json + regrets.npy")


if __name__ == '__main__':
    main()
