"""Bound the cold-chain fleet optimum using an assignment with depot copies."""
import csv
import json
import math
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recent5_20261120"))
from recent_independent import original_model, original_panel, read


def loss_bound(model, customers):
    """Return a lower bound on nominal revenue minus feasible achieved revenue net of costs."""
    rows = model.ins["nodes"]
    distance = model.ins["distance"]
    speed = model.c["routing"]["speed_km_per_min"]
    capacity = model.c["routing"]["capacity_kg"]
    km_cost = model.c["money"]["distance_U_per_km"] + model.cold / speed
    n = len(customers)
    demand = sum(rows[i][3] for i in customers)
    min_vehicles = max(1, math.ceil(demand / (capacity + 1e-7)))
    costs = np.full((2*n, 2*n), np.inf)
    costs[n+min_vehicles:, n:] = 0.
    for a, i in enumerate(customers):
        origin_km = distance[0][i]
        direct_age = origin_km / speed + rows[i][6]
        costs[n:, a] = km_cost * origin_km + model.price * rows[i][3] * model.rate * direct_age
        costs[a, n:] = km_cost * origin_km
        for b, j in enumerate(customers):
            if i == j or rows[i][3] + rows[j][3] > capacity + 1e-7:
                continue
            # Earliest possible time serving j when i is the preceding delivery.
            leg = distance[i][j]
            first_start = max(rows[0][4] + origin_km / speed, rows[i][4])
            j_start = max(rows[j][4], first_start + rows[i][6] + leg / speed)
            if j_start > rows[j][5] + 1e-7:
                continue
            if j_start + rows[j][6] + distance[j][0] / speed > rows[0][5] + 1e-7:
                continue
            age = direct_age + leg / speed + rows[j][6]
            if 1. - model.rate * age < model.minimum - 1e-7:
                continue
            costs[a,b] = km_cost * leg + model.price * rows[j][3] * model.rate * age
    row_index, col_index = linear_sum_assignment(costs)
    matching = costs[row_index, col_index].sum()
    services = sum(rows[i][6] for i in customers)
    loss = float(matching + model.cold * services)
    return dict(n=n, minimum_vehicles=min_vehicles, nominal_U=demand*model.price,
                loss_lower_U=loss, net_upper_U=demand*model.price-loss)


def exact_small(model, customers):
    best = -math.inf
    for ordered in permutations(customers):
        for cuts in range(1 << (len(ordered)-1)):
            start = 0
            profit = 0.
            for index in range(len(ordered)):
                if index+1 != len(ordered) and not (cuts >> index) & 1:
                    continue
                route = model.route(ordered[start:index+1])
                if not route.feasible:
                    break
                profit += route.net_benefit
                start = index+1
            else:
                best = max(best, profit)
    if not math.isfinite(best):
        raise ValueError("Small exact subset has no feasible fleet")
    return best


def main():
    old_inputs, _ = original_panel()
    meta = read(HERE/"metadata.json")["cases"]
    new_meta = read(HERE/"new_panel/heldout_sources.json")
    groups = [("benchmark01", old_inputs, meta),
              ("prospective01", {c:HERE/"new_panel"/v["file"] for c,v in new_meta["derived_instances"].items()}, new_meta["cases"])]
    retail = HERE / "retail_source"
    if (retail/"heldout_admission.json").is_file() and read(retail/"heldout_admission.json")["status"]=="PASS":
        registry = read(retail/"heldout_sources.json")
        groups.append(("retail01", {c:retail/v["file"] for c,v in registry["derived_instances"].items()}, registry["cases"]))
    records = []
    exact = []
    for batch, inputs, sources in groups:
        for case, path in sorted(inputs.items()):
            model, _, initial = original_model(path, case)
            row = loss_bound(model, list(range(1,101)))
            if row["net_upper_U"] < initial - 1e-6:
                raise ValueError((case, row["net_upper_U"], initial))
            records.append(dict(batch=batch, case=case, source=sources[case]["dataset"],
                                geometry=sources[case]["geometry"], **row))
            if batch=="benchmark01" and sources[case]["dataset"] not in [r["source"] for r in exact]:
                customers = [1, 2, 3, 4, 5]
                small = loss_bound(model, customers)
                optimum = exact_small(model, customers)
                if small["net_upper_U"] < optimum - 1e-6:
                    raise ValueError((case, small["net_upper_U"], optimum))
                exact.append(dict(case=case, source=sources[case]["dataset"], customers=customers,
                                  optimum_U=optimum, bound_upper_U=small["net_upper_U"]))
    out = HERE/"bounds"
    out.mkdir(exist_ok=True)
    with (out/"case_bounds.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (out/"exact_small.json").write_text(json.dumps(exact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Computed {len(records)} full-case bounds; verified {len(exact)} subsets by exact enumeration")


if __name__ == "__main__":
    main()
