"""Independent vectorised recomputation of the fixed TRAIN-only activity counts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg = json.loads((OUT / "PICKUP_REGISTRATION.json").read_text(encoding="utf-8"))
    saved_path = OUT / "PICKUP_TRAIN_RESULT.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(E / "scripts/evaluation"))
    import run_exp_reserve as rr
    data = rr.generate_dataset(200, 200, reg["fixed_train_seed"])
    quantity = np.asarray(data["demands"][:, 1:], dtype=np.float64)
    klass = np.asarray(data["temp_class"][:, 1:], dtype=np.int64)
    reveal = np.asarray(data["reveal_time"][:, 1:], dtype=np.float64)
    energy = quantity * 0.15 * (25.0 - np.asarray([18.0, 4.0, -18.0])[klass])
    revenue = np.asarray([10.0, 20.0, 30.0])[klass]
    valid = quantity > 0
    prices = np.arange(0.25, 8.0 + 1e-9, 0.125)
    clocks = np.arange(0.0, 16.0 + 1e-9, 0.5)
    rows = []
    for fold in (0, 1):
        calibration = np.arange(200) % 2 != fold
        tables = np.empty((len(prices), len(clocks)))
        for li, price in enumerate(prices):
            select = valid[calibration] & (revenue[calibration] - price * energy[calibration] > 0)
            for ci, clock in enumerate(clocks):
                tables[li, ci] = np.mean(np.sum(np.where(select & (reveal[calibration] >= clock),
                                                                energy[calibration], 0.0), axis=1))
        for day in range(fold, 200, 2):
            ids = np.flatnonzero(valid[day])
            ids = sorted(ids, key=lambda k: (float(reveal[day, k]), int(k)))
            spent = 0.0
            price_rejected = budget_rejected = 0
            seen = [0, 0, 0]
            accepted = [0, 0, 0]
            for order in ids:
                c = int(klass[day, order]); e = float(energy[day, order]); r = float(revenue[day, order])
                seen[c] += 1
                left = reg["fixed_budget"] - spent
                if e > left:
                    budget_rejected += 1
                    continue
                ci = int(np.searchsorted(clocks, reveal[day, order], side="right") - 1)
                ci = min(max(ci, 0), len(clocks) - 1)
                feasible = np.flatnonzero(tables[:, ci] <= left)
                lam = float(prices[int(feasible[0])]) if len(feasible) else float(prices[-1])
                if r - lam * e <= 0:
                    price_rejected += 1
                else:
                    accepted[c] += 1
                    spent += e
            rows.append(dict(day=day, orders=len(ids), price_rejects=price_rejected,
                             budget_rejects=budget_rejected, seen_by_class=seen,
                             accepted_by_class=accepted, precool_energy_spent=spent))
    rows.sort(key=lambda r: r["day"])
    checks = []
    for actual, expected in zip(rows, saved["per_day"]):
        checks.append(actual["day"] == expected["day"] and
                      all(actual[k] == expected[k] for k in ("orders", "price_rejects",
                                                              "budget_rejects", "seen_by_class", "accepted_by_class")) and
                      abs(actual["precool_energy_spent"] - expected["precool_energy_spent"]) < 1e-8)
    report = {
        "saved_result_sha256": sha(saved_path),
        "source_sha256": sha(E / "scripts/evaluation/run_exp_reserve.py"),
        "days_checked": len(checks),
        "days_failed": sum(not x for x in checks),
        "all_aggregate_counts_match": sum(r["price_rejects"] for r in rows) == saved["total_price_rejections"]
                                      and sum(r["budget_rejects"] for r in rows) == saved["total_budget_rejections"],
        "online_utility_qualified": False,
    }
    with (OUT / "PICKUP_INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
