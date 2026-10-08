"""SUPERSEDED (attempt #1, refuted 2026-09-23). The coupled-physics design was
refuted by adversarial review: door-shock coupling inflates door heat 40-70x
(= signal manufacture) and applies an air-temperature trajectory to product
quality, whereas cargo is the real thermal buffer (= the 0.2->0.003 error
class). The research-extension contract fields this script depends on
(door_cross_zone_fraction / quality_trajectory_integral) have been REVERTED
from coldchain_contract.py / coldchain_state.py, so this script no longer runs.
Kept as a record; the honest verdict was NO-HEADROOM (~1.5-2% even at an
unphysical 8 C door spike). See docs/当前规划/研究设计/B_静态耦合MCVRP_协议设计与无训练筛选.md.

Coupled-physics MCVRP screening gate (benchmark-B step 1, no training).

Answers one question with pure NumPy + the C0 transition engine: under the
coupled cold-chain objective (cross-zone door shock + trajectory-integral
decay), does a cold-chain-aware search beat (a) distance-optimal routing and
(b) a declared cold-chain rule, by a *material* and *non-rule-capturable*
margin?

Four solvers share one distance-optimal-ish route structure, differing only in
what they optimize:
  D     distance cheapest-insertion plan                       (distance baseline)
  Dplus D + distance-objective local search (2-opt + relocate) (distance control)
  R     Dplus + best of a declared cold-chain rule family (no-op / class asc / class desc)
  S     Dplus + coupled-objective local search (2-opt + relocate on J)

Decision (relative headroom over the distance-optimal Dplus, point estimate +
paired bootstrap):
  no-headroom      G_S < 0.02
  rule-capturable  G_S >= 0.02  AND  G_R >= 0.80 * G_S
  viable           G_S >= 0.02  AND  G_R <  0.80 * G_S
where G_X = mean_i (J_Dplus(i) - J_X(i)) / J_Dplus(i)  (positive = X better).
G_S is the cold-chain headroom of the search; G_R is the headroom recovered by
the best simple rule.  Rule family = {distance-only (no-op), serve-by-class asc,
serve-by-class desc}, where the class orders preserve distance within a class
(no strawman): reordering only regroups same-temperature stops.

The objective is a screening-level devmean normalization over the D solutions
(distance/quality/energy each divided by their per-component mean, lambda_q =
lambda_e = 1), NOT the frozen production objective.  Physics magnitude is swept
over door-shock strength dT_door so the headroom curve is visible.

Usage (from the extension root):
    python scripts/evaluation/run_coupled_mcvrp_screen.py \
        --num-customers 100 --num-instances 20 --out results/coupled_screen
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
for p in ("coldchain", "evaluation"):
    _p = os.path.join(_SCRIPTS, p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import replace as dc_replace  # noqa: E402

from coldchain_contract import default_pilot_contract  # noqa: E402
from coldchain_evaluator import replay_static_pickup_route  # noqa: E402


# --------------------------------------------------------------------------- #
# Instance generation
# --------------------------------------------------------------------------- #
def generate_instance(rng: np.random.Generator, n: int, capacity: float,
                      demand_lo: float, demand_hi: float) -> dict:
    """One static MCVRP instance: depot at centre, customers in a 100x100 km box."""
    coords = rng.uniform(0.0, 100.0, size=(n + 1, 2))
    coords[0] = [50.0, 50.0]
    demands = rng.uniform(demand_lo, demand_hi, size=n + 1)
    demands[0] = 0.0
    temp_class = rng.integers(0, 3, size=n + 1)
    temp_class[0] = 0
    d = np.asarray(coords, dtype=np.float64)
    dist_mat = np.linalg.norm(d[:, None, :] - d[None, :, :], axis=-1)
    return {
        "coords": coords.astype(np.float64),
        "demands": demands.astype(np.float64),
        "tw_start": np.zeros(n + 1),
        "tw_end": np.full(n + 1, 1e12),
        "service_time": np.full(n + 1, 0.05),
        "temp_class": temp_class.astype(np.int64),
        "initial_quality": np.ones(n + 1),
        "dist_mat": dist_mat,
    }


def num_vehicles_for(inst: dict, capacity: float) -> int:
    total = float(np.sum(inst["demands"][1:]))
    return int(np.ceil(total / capacity)) + 2


# --------------------------------------------------------------------------- #
# Route representation helpers
# --------------------------------------------------------------------------- #
def trips_to_route(trips: list[list[int]]) -> np.ndarray:
    route = []
    for trip in trips:
        route.append(0)
        route.extend(trip)
    route.append(0)
    return np.asarray(route, dtype=np.int64)


def distance(inst: dict, a: int, b: int) -> float:
    return float(inst["dist_mat"][a, b])


# --------------------------------------------------------------------------- #
# Distance cheapest insertion + distance local search (Dplus)
# --------------------------------------------------------------------------- #
def cheapest_insertion(inst: dict, capacity: float, num_vehicles: int) -> list[list[int]]:
    customers = [c for c in range(1, len(inst["demands"])) if inst["demands"][c] > 0]
    trips: list[list[int]] = [[] for _ in range(num_vehicles)]
    loads = [0.0] * num_vehicles
    for c in customers:
        best = None  # (marginal, vid, pos)
        for vid in range(num_vehicles):
            if loads[vid] + inst["demands"][c] > capacity + 1e-9:
                continue
            for pos in range(len(trips[vid]) + 1):
                prev = trips[vid][pos - 1] if pos > 0 else 0
                nxt = trips[vid][pos] if pos < len(trips[vid]) else 0
                marginal = distance(inst, prev, c) + distance(inst, c, nxt) - distance(inst, prev, nxt)
                if best is None or marginal < best[0]:
                    best = (marginal, vid, pos)
        if best is None:
            raise ValueError("capacity infeasible: no vehicle fits customer %d" % c)
        _, vid, pos = best
        trips[vid].insert(pos, c)
        loads[vid] += float(inst["demands"][c])
    return [t for t in trips if t]


def _plan_distance(inst: dict, trips: list[list[int]]) -> float:
    total = 0.0
    for trip in trips:
        prev = 0
        for c in trip:
            total += distance(inst, prev, c)
            prev = c
        total += distance(inst, prev, 0)
    return total


def distance_local_search(inst: dict, trips: list[list[int]], capacity: float, budget: int,
                          rng: np.random.Generator) -> list[list[int]]:
    """First-improvement 2-opt (intra) + relocate (inter) on pure distance."""
    current = [list(t) for t in trips]
    best = _plan_distance(inst, current)
    used = 0
    improved = True
    while improved and used < budget:
        improved = False
        # intra 2-opt
        for vi in range(len(current)):
            trip = current[vi]
            L = len(trip)
            for i in range(L - 1):
                for j in range(i + 2, L + 1):
                    cand = [list(t) for t in current]
                    cand[vi] = trip[:i] + trip[i:j][::-1] + trip[j:]
                    d = _plan_distance(inst, cand)
                    used += 1
                    if d < best - 1e-12:
                        current, best = cand, d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved or used >= budget:
            continue
        # inter relocate
        for vi in range(len(current)):
            for ci, c in enumerate(current[vi]):
                for vj in range(len(current)):
                    if vj == vi:
                        continue
                    if sum(inst["demands"][x] for x in current[vj]) + inst["demands"][c] > capacity + 1e-9:
                        continue
                    for pos in range(len(current[vj]) + 1):
                        cand = [list(t) for t in current]
                        cand[vi] = current[vi][:ci] + current[vi][ci + 1:]
                        cand[vj] = current[vj][:pos] + [c] + current[vj][pos:]
                        if not cand[vi]:
                            cand = [t for t in cand if t]
                        d = _plan_distance(inst, cand)
                        used += 1
                        if d < best - 1e-12:
                            current, best = cand, d
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current


# --------------------------------------------------------------------------- #
# Coupled-objective evaluation
# --------------------------------------------------------------------------- #
def eval_components(inst: dict, trips: list[list[int]], contract, speed: float) -> dict:
    r = replay_static_pickup_route(trips_to_route(trips), inst, contract, speed=speed)
    return {
        "distance": float(r["distance_km"]),
        "quality": float(r["quality_loss"]),
        "energy": float(r["energy_kwh"]),
        "coldchain_cost": float(r["coldchain_cost"]),
    }


def _normalized_J(comp: dict, norm: dict) -> float:
    return (comp["distance"] / norm["distance"]
            + comp["quality"] / norm["quality"]
            + comp["energy"] / norm["energy"])


# --------------------------------------------------------------------------- #
# Rule reorder (R)
# --------------------------------------------------------------------------- #
def reorder_by_class(inst: dict, trips: list[list[int]], descending: bool) -> list[list[int]]:
    """Serve by temp class (desc = frozen first), preserving distance order within a class."""
    tc = inst["temp_class"]
    out = []
    for trip in trips:
        if descending:
            order = sorted(range(len(trip)), key=lambda i: (-int(tc[trip[i]]), i))
        else:
            order = sorted(range(len(trip)), key=lambda i: (int(tc[trip[i]]), i))
        out.append([trip[i] for i in order])
    return out


# --------------------------------------------------------------------------- #
# Coupled-objective local search (S)
# --------------------------------------------------------------------------- #
def coupled_local_search(inst: dict, trips: list[list[int]], capacity: float,
                         contract, speed: float, norm: dict, budget: int,
                         rng: np.random.Generator) -> list[list[int]]:
    def score(trips):
        return _normalized_J(eval_components(inst, trips, contract, speed), norm)

    current = [list(t) for t in trips]
    best = score(current)
    used = 0
    improved = True
    while improved and used < budget:
        improved = False
        for vi in range(len(current)):
            trip = current[vi]
            L = len(trip)
            for i in range(L - 1):
                for j in range(i + 2, L + 1):
                    cand = [list(t) for t in current]
                    cand[vi] = trip[:i] + trip[i:j][::-1] + trip[j:]
                    s = score(cand)
                    used += 1
                    if s < best - 1e-12:
                        current, best = cand, s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved or used >= budget:
            continue
        for vi in range(len(current)):
            for ci, c in enumerate(current[vi]):
                for vj in range(len(current)):
                    if vj == vi:
                        continue
                    if sum(inst["demands"][x] for x in current[vj]) + inst["demands"][c] > capacity + 1e-9:
                        continue
                    for pos in range(len(current[vj]) + 1):
                        cand = [list(t) for t in current]
                        cand[vi] = current[vi][:ci] + current[vi][ci + 1:]
                        cand[vj] = current[vj][:pos] + [c] + current[vj][pos:]
                        if not cand[vi]:
                            cand = [t for t in cand if t]
                        s = score(cand)
                        used += 1
                        if s < best - 1e-12:
                            current, best = cand, s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current


# --------------------------------------------------------------------------- #
# Research contract
# --------------------------------------------------------------------------- #
def research_contract(dt_door: float, cross_fraction: float) -> "default_pilot_contract":
    base = default_pilot_contract()
    th = base.thermal
    # door_heat scaled so each opening raises the opened zone by dt_door degC
    door_heat = tuple(dt_door * cap for cap in th.thermal_capacity_kwh_per_c)
    th = dc_replace(
        th,
        door_heat_kwh=door_heat,
        door_cross_zone_fraction=cross_fraction,
        quality_trajectory_integral=True,
        diurnal_amplitude_c=0.0,  # isolate door coupling; diurnal is a separate closed lever
    )
    return dc_replace(base, thermal=th)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(h: np.ndarray, b: int, rng: np.random.Generator):
    means = np.empty(b)
    n = len(h)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        means[k] = np.mean(h[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def relative_headroom(j_x: np.ndarray, j_s: np.ndarray) -> np.ndarray:
    return (j_x - j_s) / j_x


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-customers", type=int, default=100)
    ap.add_argument("--num-instances", type=int, default=20)
    ap.add_argument("--capacity", type=float, default=45.0)
    ap.add_argument("--speed-kmph", type=float, default=40.0)
    ap.add_argument("--cross-fraction", type=float, default=0.3)
    ap.add_argument("--dt-door-sweep", type=float, nargs="+", default=[0.0, 4.0, 8.0])
    ap.add_argument("--budget", type=int, default=1200, help="local-search eval budget per instance")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--no-headroom-threshold", type=float, default=0.02)
    ap.add_argument("--rule-capture-threshold", type=float, default=0.80,
                    help="fraction of G_S the rule must recover to be 'rule-capturable'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    instances = [generate_instance(rng, args.num_customers, args.capacity, 1.0, 5.0)
                 for _ in range(args.num_instances)]
    num_vehicles = [num_vehicles_for(inst, args.capacity) for inst in instances]

    # D and Dplus are cold-chain-agnostic: compute once.
    d_plans, dplus_plans = [], []
    for inst, nv in zip(instances, num_vehicles):
        d = cheapest_insertion(inst, args.capacity, nv)
        dp = distance_local_search(inst, d, args.capacity, args.budget, rng)
        d_plans.append(d)
        dplus_plans.append(dp)

    report = {"config": vars(args), "per_dt": {}}

    for dt in args.dt_door_sweep:
        contract = research_contract(dt, args.cross_fraction)

        # Normalization from the D solutions under this physics.
        comps_d = [eval_components(inst, p, contract, args.speed_kmph)
                   for inst, p in zip(instances, d_plans)]
        norm = {
            "distance": float(np.mean([c["distance"] for c in comps_d])) or 1.0,
            "quality": float(np.mean([c["quality"] for c in comps_d])) or 1.0,
            "energy": float(np.mean([c["energy"] for c in comps_d])) or 1.0,
        }

        j_d, j_dplus, j_r, j_s = [], [], [], []
        for inst, d_plan, dplus in zip(instances, d_plans, dplus_plans):
            jd = _normalized_J(eval_components(inst, d_plan, contract, args.speed_kmph), norm)
            jdp = _normalized_J(eval_components(inst, dplus, contract, args.speed_kmph), norm)
            r_asc = reorder_by_class(inst, dplus, descending=False)
            r_desc = reorder_by_class(inst, dplus, descending=True)
            j_asc = _normalized_J(eval_components(inst, r_asc, contract, args.speed_kmph), norm)
            j_desc = _normalized_J(eval_components(inst, r_desc, contract, args.speed_kmph), norm)
            j_d.append(jd)
            j_dplus.append(jdp)
            j_r.append(min(jdp, j_asc, j_desc))  # rule family includes the no-op (distance-only)
            s = coupled_local_search(inst, dplus, args.capacity, contract,
                                     args.speed_kmph, norm, args.budget, rng)
            j_s.append(_normalized_J(eval_components(inst, s, contract, args.speed_kmph), norm))

        j_d = np.asarray(j_d)
        j_dplus = np.asarray(j_dplus)
        j_r = np.asarray(j_r)
        j_s = np.asarray(j_s)

        h_ds = relative_headroom(j_d, j_s)
        h_ddplus = relative_headroom(j_d, j_dplus)
        g_s = relative_headroom(j_dplus, j_s)   # G_S: search headroom over distance routing
        g_r = relative_headroom(j_dplus, j_r)   # G_R: rule headroom over distance routing

        def stat(h):
            return {"mean": float(np.mean(h)),
                    "ci_lo": paired_bootstrap(h, 1000, rng)[0],
                    "ci_hi": paired_bootstrap(h, 1000, rng)[1]}

        row = {
            "H_DS_total": stat(h_ds),
            "H_DDplus_distance": stat(h_ddplus),
            "G_S_search_headroom": stat(g_s),
            "G_R_rule_headroom": stat(g_r),
            "rule_capture_fraction": float(np.mean(g_r) / max(np.mean(g_s), 1e-12)),
            "norm": norm,
        }
        report["per_dt"][str(dt)] = row
        print(f"\n=== dT_door={dt} degC (cross={args.cross_fraction}) ===")
        print(f"  norm: dist={norm['distance']:.3f} quality={norm['quality']:.4f} energy={norm['energy']:.3f}")
        print(f"  H_DS(total over D)        = {row['H_DS_total']['mean']*100:.2f}%")
        print(f"  H_DDplus(distance control)= {row['H_DDplus_distance']['mean']*100:.2f}%")
        print(f"  G_S(search headroom)      = {row['G_S_search_headroom']['mean']*100:.2f}%  CI[{row['G_S_search_headroom']['ci_lo']*100:.2f}%,{row['G_S_search_headroom']['ci_hi']*100:.2f}%]")
        print(f"  G_R(rule headroom)        = {row['G_R_rule_headroom']['mean']*100:.2f}%")
        print(f"  rule capture fraction     = {row['rule_capture_fraction']:.2f}")

    # Decision at the strongest swept setting (nominal = last sweep value).
    nominal = report["per_dt"][str(args.dt_door_sweep[-1])]
    g_s = nominal["G_S_search_headroom"]["mean"]
    g_r = nominal["G_R_rule_headroom"]["mean"]
    t_no = args.no_headroom_threshold
    t_cap = args.rule_capture_threshold
    if g_s < t_no:
        verdict = "NO-HEADROOM (strengthen physics magnitude)"
    elif g_r >= t_cap * g_s:
        verdict = "RULE-CAPTURABLE (B dies; change physics first)"
    else:
        verdict = "VIABLE (B proceeds to full benchmark)"
    print("\n=== DECISION (nominal dT_door=%s) ===" % args.dt_door_sweep[-1])
    print(f"  G_S={g_s*100:.2f}% (thresh {t_no*100:.0f}%), "
          f"rule_capture={g_r/max(g_s,1e-12):.2f} (thresh {t_cap})")
    print(f"  -> {verdict}")
    report["decision"] = {
        "nominal_dt_door": float(args.dt_door_sweep[-1]),
        "G_S_search_headroom": g_s,
        "rule_capture_fraction": float(g_r / max(g_s, 1e-12)),
        "verdict": verdict,
    }

    with open(os.path.join(args.out, "screen.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {args.out}/screen.json")


if __name__ == "__main__":
    main()
