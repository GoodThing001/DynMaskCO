"""Predeclared exhaustive interior-pair C0 opportunity diagnosis on old public states."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r18_pair_shard"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def teacher(packet, masked, pf):
    original = copy.deepcopy(packet["plan"])
    baseline, _ = pf.certificate(packet, original)
    assert baseline["qualified"]
    original_owner = {order: vehicle for vehicle, route in original.items() for order in route}
    retained = set()
    for route in original.values():
        for a, b in zip([0] + route, route + [0]):
            if a not in masked and b not in masked and a != b:
                retained.add((min(a, b), max(a, b)))
    current = {v: [o for o in route if o not in masked] for v, route in original.items()}
    remaining = list(masked)
    steps = []
    for step in range(2):
        candidates = []
        for order in remaining:
            for vehicle in range(3):
                route = current[str(vehicle)]
                for pos in range(len(route) + 1):
                    cn = int(packet["snapshot"]["committed_next"][vehicle])
                    if packet["snapshot"]["vehicle_status"][vehicle] == "committed" and cn > 0 and pos == 0:
                        continue
                    a = route[pos - 1] if pos else 0
                    b = route[pos] if pos < len(route) else 0
                    if a != b and (min(a, b), max(a, b)) in retained:
                        continue
                    trial = copy.deepcopy(current)
                    trial[str(vehicle)].insert(pos, order)
                    for pending in remaining:
                        if pending != order:
                            trial[str(original_owner[pending])].append(pending)
                    try:
                        cert, _ = pf.certificate(packet, trial)
                    except ValueError:
                        continue
                    if cert["qualified"]:
                        candidates.append({"action": [order, vehicle, pos], "c0_utility": cert["evaluation"]["utility"], "hard_feasible": cert["evaluation"]["hard_feasible"]})
        if not candidates:
            raise RuntimeError("Registered pair has no certified candidate")
        winner = min(candidates, key=lambda c: (-c["c0_utility"], c["action"]))
        order, vehicle, pos = winner["action"]
        current[str(vehicle)].insert(pos, order)
        remaining.remove(order)
        steps.append({"step": step, "candidates": candidates, "selected_action": winner["action"]})
    final, _ = pf.certificate(packet, current)
    assert final["qualified"]
    edges = {(min(a, b), max(a, b)) for route in current.values() for a, b in zip([0] + route, route + [0]) if a != b}
    assert retained.issubset(edges)
    return {"masked": masked, "original_plan": original, "baseline_c0": baseline["evaluation"]["utility"],
            "steps": steps, "final_plan": current, "final_certificate": final,
            "delta_c0_vs_original": final["evaluation"]["utility"] - baseline["evaluation"]["utility"],
            "retained_edges_preserved": True}


def main():
    import psutil
    psutil.Process().cpu_affinity([7])
    reg = json.loads((BASE / "PICKUP_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == reg["script_sha256"]
    sys.path.insert(0, str(ROOT / "tools"))
    import qualify_pickup_joint_policy_transaction_root_v3 as q
    _, decode = q.codecs()
    pf = q.pf
    rows = []
    for name in reg["input_paths"]:
        path = ROOT / name
        assert sha(path) == reg["input_sha256"][name]
        packet = decode(json.loads(path.read_text(encoding="utf-8")))["scoring_packet"]
        pf.validate_public(packet)
        assert packet["plan"]["0"][1:5] == reg["expected_interior"]
        pairs = [teacher(packet, pair, pf) for pair in reg["masks"]]
        rows.append({"current_order": packet["admission_input"]["current_order"], "input_sha256": sha(path), "pairs": pairs})
    assert [r["current_order"] for r in rows] == [69, 77]
    assert all(sha(ROOT / n) == reg["input_sha256"][n] for n in reg["input_paths"])
    out = {"registration_sha256": sha(BASE / "PICKUP_REGISTRATION.json"), "rows": rows,
           "old_correlated_states_only": True, "new_scientific_family_started": False,
           "model_training_or_online_gain": False}
    with (BASE / "PICKUP_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"orders": [x["current_order"] for x in rows],
                      "deltas": [[p["delta_c0_vs_original"] for p in x["pairs"]] for x in rows]}))


if __name__ == "__main__":
    main()
