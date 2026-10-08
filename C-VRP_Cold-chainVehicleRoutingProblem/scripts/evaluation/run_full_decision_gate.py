"""⚠️ 废弃（DEPRECATED，2026-09-23，GPT 审计 7 点成立）。

本脚本不是决策器，是终局离线重排 + 硬约束丢弃的简化打分器，缺陷：
1. evaluate_plan 的 feasible 被丢弃，超预算方案利润照样计均值/CI；
2. t=max(t,deadline) 把截止时间当最早服务时间（应 ASAP 服务、deadline 作上界）；
3. 无因果在线执行（后揭示订单能改第一站，非可部署在线车队可实现）；
4. 品质 dwell 按 NN 访问序返回却按原 idxs 序配 demand；且 dwell 用 service_start
   非 service_finish（多算每单 0.05h 服务时间）；
5. 缺容量 / Q≥0.8 / 车队约束；未调 coldchain_state；
6. "oracle" 是贪心按 revenue/proxy 排序，非最优上界；
7. 只证"能前向"未证"无泄漏"。

正确路径：把真 3b 建在现有 strict_online_env + coldchain_state + jf1h_repair 上
（已正确实现因果执行/品质配对/硬约束），只新加 accept/reject + 绿色预算 + 冻结 encoder。
本脚本保留作负例参考，不再用于任何结论。

"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REV = {0: 10.0, 1: 20.0, 2: 30.0}
DT = {0: 7.0, 1: 21.0, 2: 43.0}
K_EFF = {0: 4.53e-3, 1: 1.3e-3, 2: 2.23e-5}
PVAL = {0: 1.0, 1: 1.3, 2: 2.0}
CENTERS = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
DEPOT = np.array([0.5, 0.5])
UA, E_DOOR, COP = 0.03, 0.056, 1.5
SPEED_KMH, KM_PER_UNIT, SERVICE_H = 30.0, 20.0, 0.05


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def generate_day(rng, z, T=16.0, W=4.0):
    beta = np.array([2.0, -0.5, -0.5])
    base = np.array([0.2, 0.4, 0.4])
    weights = _softmax(np.log(base) + beta * z)
    peak = float(np.clip(11.0 + 4.0 * z, 9.0, 19.0))
    total_per_h = 200.0 * np.exp(0.3 * z) / T
    lam_max = total_per_h * 2.2
    n = int(rng.poisson(lam_max * T))
    times = rng.uniform(0.0, T, n)
    prof = 1.0 + 1.2 * np.exp(-((times - peak) ** 2) / (2.0 * 1.5 ** 2))
    keep = rng.uniform(size=n) < (prof * total_per_h / lam_max)
    times = times[keep]
    spot = rng.choice(3, size=times.size, p=weights)
    locs = CENTERS[spot] + rng.normal(0.0, 0.12, size=(times.size, 2))
    class_probs = np.array([[0.1, 0.3, 0.6], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
    cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
    demand = rng.uniform(1.0, 3.0, times.size)
    deadline = times + rng.uniform(0.5, 1.0) * W
    idx = np.argsort(times)
    return (times[idx], cls[idx], locs[idx], demand[idx], deadline[idx])


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def evaluate_route(idxs, locs, demand, cls, deadline):
    """单类最近邻路由，返回 (feasible, distance_km, route_time_h, dwells_h, n_stops)。

    dwells[i] = 从该单取货到返仓的在途+服务时长（小时）。
    """
    if len(idxs) == 0:
        return True, 0.0, 0.0, np.array([]), 0
    # 最近邻 tour
    remaining = list(idxs)
    cur = DEPOT
    tour = []
    while remaining:
        nxt = min(remaining, key=lambda i: _dist(cur, locs[i]))
        tour.append(nxt)
        cur = locs[nxt]
        remaining.remove(nxt)
    # 距离
    dist_units = _dist(DEPOT, locs[tour[0]])
    for a, b in zip(tour[:-1], tour[1:]):
        dist_units += _dist(locs[a], locs[b])
    dist_units += _dist(locs[tour[-1]], DEPOT)
    distance_km = dist_units * KM_PER_UNIT

    # 时间（旅行 + 服务），从 0 起（返仓为终点）
    t = 0.0
    arrive_times = []
    cur = DEPOT
    for i in tour:
        t += _dist(cur, locs[i]) / SPEED_KMH * KM_PER_UNIT  # 注意单位：dist_units*KM_PER_UNIT / SPEED_KMH
        t = max(t, float(deadline[i]))  # 到站即取（不早于 deadline 无效，这里用 deadline 作软上限）
        arrive_times.append(t)
        t += SERVICE_H
        cur = locs[i]
    t += _dist(cur, DEPOT) / SPEED_KMH * KM_PER_UNIT
    return_time = t

    dwells = np.array([return_time - a for a in arrive_times])
    route_time_h = return_time
    # 时间窗可行性：取货时间 <= deadline（到站即取，已 max 处理；这里检查是否违反）
    feasible = all(a <= deadline[i] + 1e-6 for a, i in zip(arrive_times, tour))
    return feasible, distance_km, route_time_h, dwells, len(tour)


def evaluate_plan(day, accepted, B):
    """完整评价：给定 accepted（bool 数组），返回 (profit, feasible, D_km, Q, E_kwh)。"""
    times, cls, locs, demand, deadline = day
    D_km, Q, E = 0.0, 0.0, 0.0
    feasible = True
    for c in range(3):
        idxs = [i for i in range(len(times)) if accepted[i] and cls[i] == c]
        if not idxs:
            continue
        ok, dist_km, route_time_h, dwells, n_stops = evaluate_route(
            idxs, locs, demand, cls, deadline)
        if not ok:
            feasible = False
        D_km += dist_km
        E += (UA * DT[c] * route_time_h + E_DOOR * n_stops) / COP
        for i, dwell in zip(idxs, dwells):
            Q += PVAL[c] * demand[i] * (1.0 - np.exp(-K_EFF[c] * dwell))
    if E > B + 1e-9:
        feasible = False
    revenue = float(np.sum([REV[int(cls[i])] for i in range(len(times)) if accepted[i]]))
    profit = revenue - Q - D_km  # 距离作燃油成本（约 1:1 无量纲）
    return profit, feasible, D_km, Q, E


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]

    # 预算 B：训练天"全接"能耗均值 × (1−ρ)
    def full_energy(day):
        times, cls, locs, demand, deadline = day
        E = 0.0
        for c in range(3):
            idxs = [i for i in range(len(times)) if cls[i] == c]
            if not idxs:
                continue
            _, _, rt, _, ns = evaluate_route(idxs, locs, demand, cls, deadline)
            E += (UA * DT[c] * rt + E_DOOR * ns) / COP
        return E
    baseline = float(np.mean([full_energy(d) for d in hist_days]))
    B = (1.0 - args.rho) * baseline

    # 每单能量代理（用于策略的机会成本）：UA·ΔT·(2×depot距离)/speed + E_door, /COP
    def energy_proxy(cls, loc):
        d_km = 2.0 * _dist(DEPOT, loc) * KM_PER_UNIT
        return (UA * DT[cls] * (d_km / SPEED_KMH) + E_DOOR) / COP

    # 预计算每历史天的 value/energy 数组（k-NN 内层向量化）
    hist_v = [np.array([REV[int(c)] for c in d[1]]) for d in hist_days]
    hist_e = [np.array([energy_proxy(d[1][j], d[2][j]) for j in range(len(d[0]))])
              for d in hist_days]

    # k-NN 索引：morning 状态（每热点+每温区计数）
    def morning_feat(day, tau):
        times, cls, locs, demand, deadline = day
        m = times <= tau
        feat = np.zeros(6)
        for c in range(3):
            feat[3 + c] = np.sum(cls[m] == c)
        if m.sum() > 0:
            d = np.linalg.norm(locs[m][:, None, :] - CENTERS[None, :, :], axis=2)
            nearest = d.argmin(1)
            for s in range(3):
                feat[s] = np.sum(nearest == s)
        return feat
    Xtr = np.array([morning_feat(d, args.tau) for d in hist_days])
    Xtr_n = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-6)

    def stat(x):
        n = len(x)
        r = np.random.default_rng(args.seed + 777)
        mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
        return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(mm, 2.5)),
                "ci_hi": float(np.percentile(mm, 97.5))}

    P_my = np.zeros(args.test_days); P_kn = np.zeros(args.test_days); P_or = np.zeros(args.test_days)
    for di, day in enumerate(test_days):
        times, cls, locs, demand, deadline = day
        morning = times <= args.tau
        # myopic：来就收（能耗预算内）
        accepted = np.zeros(len(times), dtype=bool)
        accepted[morning] = True
        E_used = 0.0
        # morning 能耗
        for c in range(3):
            idxs = [i for i in range(len(times)) if morning[i] and cls[i] == c]
            if idxs:
                _, _, rt, _, ns = evaluate_route(idxs, locs, demand, cls, deadline)
                E_used += (UA * DT[c] * rt + E_DOOR * ns) / COP
        for i in np.where(~morning)[0]:
            e = energy_proxy(cls[i], locs[i])
            if E_used + e <= B + 1e-9:
                accepted[i] = True
                E_used += e
        P_my[di], _, _, _, _ = evaluate_plan(day, accepted, B)

        # oracle（greedy 全知：按 value/energy 降序收，直到预算满）
        accepted = np.zeros(len(times), dtype=bool)
        accepted[morning] = True
        E_used = 0.0
        for c in range(3):
            idxs = [i for i in range(len(times)) if morning[i] and cls[i] == c]
            if idxs:
                _, _, rt, _, ns = evaluate_route(idxs, locs, demand, cls, deadline)
                E_used += (UA * DT[c] * rt + E_DOOR * ns) / COP
        aft = np.where(~morning)[0]
        ratio = np.array([REV[int(cls[i])] / max(energy_proxy(cls[i], locs[i]), 1e-9) for i in aft])
        order = aft[np.argsort(-ratio)]
        for i in order:
            e = energy_proxy(cls[i], locs[i])
            if E_used + e <= B + 1e-9:
                accepted[i] = True
                E_used += e
        P_or[di], _, _, _, _ = evaluate_plan(day, accepted, B)

        # k-NN bootstrap
        feat = morning_feat(day, args.tau)
        x = (feat - Xtr.mean(0)) / (Xtr.std(0) + 1e-6)
        d = ((Xtr_n - x) ** 2).sum(1)
        kn = np.argsort(d)[:args.k]
        accepted = np.zeros(len(times), dtype=bool)
        accepted[morning] = True
        E_used = 0.0
        for c in range(3):
            idxs = [i for i in range(len(times)) if morning[i] and cls[i] == c]
            if idxs:
                _, _, rt, _, ns = evaluate_route(idxs, locs, demand, cls, deadline)
                E_used += (UA * DT[c] * rt + E_DOOR * ns) / COP
        rem = B - E_used
        for i in np.where(~morning)[0]:
            if rem <= 1e-9:
                break
            v_o, e_o = REV[int(cls[i])], energy_proxy(cls[i], locs[i])
            t = times[i]
            lams = []
            for h in kn:
                ht = hist_days[h][0]
                f = ht > t
                fv = hist_v[h][f]
                fe = hist_e[h][f]
                if len(fv) == 0 or rem <= 1e-12:
                    lams.append(0.0); continue
                order_f = np.argsort(-(fv / np.maximum(fe, 1e-12)))
                rem_f = rem; lam = 0.0
                for j in order_f:
                    if fe[j] <= rem_f + 1e-12:
                        rem_f -= fe[j]
                    else:
                        lam = float(fv[j] / fe[j]); break
                lams.append(lam)
            lam = float(np.mean(lams))
            if e_o <= rem + 1e-9 and v_o > e_o * lam + 1e-9:
                accepted[i] = True
                rem -= e_o
        P_kn[di], _, _, _, _ = evaluate_plan(day, accepted, B)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"baseline": baseline, "B": B},
        "profit": {"myopic": stat(P_my), "knn": stat(P_kn), "oracle": stat(P_or)},
        "knn_minus_myopic": stat(P_kn - P_my),
        "oracle_minus_myopic": stat(P_or - P_my),
        "knn_share_of_future_info": float(np.mean(P_kn - P_my) / max(np.mean(P_or - P_my), 1e-9)),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
