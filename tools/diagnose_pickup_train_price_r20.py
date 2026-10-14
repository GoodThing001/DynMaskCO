"""Frozen-TRAIN-only, cross-fit admission activity diagnostic; no online utility."""
import bisect
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
P = ROOT / "research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4"
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    registration = json.loads((OUT / "PICKUP_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == registration["runner_sha256"]
    for path, digest in registration["sources"].items():
        assert sha(ROOT / path) == digest
    sys.path.insert(0, str(E / "scripts/evaluation"))
    sys.path.insert(0, str(P))
    import run_exp_reserve as rr
    from reserving_admission import REV, build_clock_grid, build_lambda_grid, calibrate, e_lower_of

    old = json.loads((P / "REGISTRATION.json").read_text(encoding="utf-8"))
    fixed = old["frozen"]
    train = rr.generate_dataset(fixed["train_instances"], fixed["n_orders"], fixed["train_seed"])
    clock = build_clock_grid(fixed["booking_horizon"])
    prices = build_lambda_grid()
    budget = registration["fixed_budget"]
    fold_rows = []
    for fold in (0, 1):
        # Day is the unit. Evaluation fold never calibrates its own table.
        calibration = {k: v[[i for i in range(200) if i % 2 != fold]] for k, v in train.items()}
        table = calibrate(calibration, old["objective"], fixed["booking_horizon"], prices, clock)
        day_rows = []
        for day in range(fold, 200, 2):
            orders = [i for i in range(1, fixed["n_orders"] + 1) if float(train["demands"][day, i]) > 0]
            orders.sort(key=lambda i: (float(train["reveal_time"][day, i]), i))
            spent = 0.0
            rejected_price = rejected_budget = 0
            accepted_by_class = [0, 0, 0]
            seen_by_class = [0, 0, 0]
            for order in orders:
                klass = int(train["temp_class"][day, order])
                energy = e_lower_of(klass, float(train["demands"][day, order]), old["objective"])
                seen_by_class[klass] += 1
                remaining = budget - spent
                if energy > remaining:
                    rejected_budget += 1
                    continue
                t = float(train["reveal_time"][day, order])
                ti = min(max(bisect.bisect_right(clock, t) - 1, 0), len(clock) - 1)
                lam = next((prices[i] for i in range(len(prices)) if table[i][ti] <= remaining), prices[-1])
                if REV[klass] - lam * energy <= 0:
                    rejected_price += 1
                else:
                    accepted_by_class[klass] += 1
                    spent += energy
            day_rows.append({"day": day, "orders": len(orders), "price_rejects": rejected_price,
                             "budget_rejects": rejected_budget, "accepted_by_class": accepted_by_class,
                             "seen_by_class": seen_by_class, "precool_energy_spent": spent})
        fold_rows.extend(day_rows)
    fold_rows.sort(key=lambda r: r["day"])
    assert [r["day"] for r in fold_rows] == list(range(200))
    result = {
        "train_seed": fixed["train_seed"],
        "train_days": 200,
        "crossfit": "even/odd by frozen day index, calibrate on opposite half",
        "fixed_budget": budget,
        "days_with_price_rejection": sum(r["price_rejects"] > 0 for r in fold_rows),
        "total_price_rejections": sum(r["price_rejects"] for r in fold_rows),
        "total_budget_rejections": sum(r["budget_rejects"] for r in fold_rows),
        "days_with_no_price_rejection": sum(r["price_rejects"] == 0 for r in fold_rows),
        "class_seen": [sum(r["seen_by_class"][c] for r in fold_rows) for c in range(3)],
        "class_accepted": [sum(r["accepted_by_class"][c] for r in fold_rows) for c in range(3)],
        "per_day": fold_rows,
        "routing_and_C0_evaluated": False,
        "online_utility_evaluated": False,
        "no_model_or_science_gain_claim": True,
    }
    with (OUT / "PICKUP_TRAIN_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("days_with_price_rejection", "total_price_rejections",
                                             "total_budget_rejections", "days_with_no_price_rejection")}))


if __name__ == "__main__":
    main()
