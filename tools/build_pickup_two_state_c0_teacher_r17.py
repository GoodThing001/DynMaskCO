"""C0-certified insertable two-step teacher on two frozen old public states.

This is an engineering teacher-input closure, not a scientific training set.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r17_teacher_dualcore"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import psutil
    psutil.Process().cpu_affinity([7])
    reg = json.loads((BASE / "PICKUP_TEACHER_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == reg["script_sha256"]
    sys.path.insert(0, str(ROOT / "tools"))
    import qualify_pickup_joint_policy_transaction_root_v3 as q
    _, decode = q.codecs()
    pf = q.pf
    output = []
    for name in reg["input_paths"]:
        path = ROOT / name
        assert sha(path) == reg["input_sha256"][name]
        packet = decode(json.loads(path.read_text(encoding="utf-8")))["scoring_packet"]
        pf.validate_public(packet)
        assert packet["admission_input"]["current_order"] in (69, 77)
        original = copy.deepcopy(packet["plan"])
        masked = set(reg["masked_orders"])
        assert masked.issubset(set(pf.family(packet)[1]))
        retained = set()
        for route in original.values():
            for a, b in zip([0] + route, route + [0]):
                if a not in masked and b not in masked and a != b:
                    retained.add((min(a, b), max(a, b)))
        current = {v: [o for o in route if o not in masked] for v, route in original.items()}
        remaining = list(reg["masked_orders"])
        steps = []
        for step in range(len(remaining)):
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
                                trial[str(packet["ownership"][str(pending)])].append(pending)
                        try:
                            cert, _ = pf.certificate(packet, trial)
                        except ValueError:
                            continue
                        if not cert["qualified"]:
                            continue
                        candidates.append({"action": [order, vehicle, pos],
                                           "c0_utility": cert["evaluation"]["utility"],
                                           "certificate": cert, "temporary_full_plan": trial})
            assert candidates
            selected = min(candidates, key=lambda x: (-x["c0_utility"], x["action"]))
            order, vehicle, pos = selected["action"]
            remaining_before = remaining.copy()
            current[str(vehicle)].insert(pos, order)
            remaining.remove(order)
            steps.append({"step": step, "remaining_before": remaining_before, "candidates": candidates,
                          "selected_action": selected["action"], "selected_c0_utility": selected["c0_utility"]})
        final_cert, _ = pf.certificate(packet, current)
        assert final_cert["qualified"]
        edges = {(min(a, b), max(a, b)) for route in current.values() for a, b in zip([0] + route, route + [0]) if a != b}
        assert retained.issubset(edges)
        output.append({"input_path": name, "input_sha256": sha(path),
                       "current_order": packet["admission_input"]["current_order"],
                       "original_plan": original, "masked_orders": reg["masked_orders"],
                       "retained_edges": sorted(retained), "steps": steps,
                       "final_plan": current, "final_certificate": final_cert,
                       "retained_edges_preserved": True})
    assert [r["current_order"] for r in output] == [69, 77]
    for name in reg["input_paths"]:
        assert sha(ROOT / name) == reg["input_sha256"][name]
    result = {"registration_sha256": sha(BASE / "PICKUP_TEACHER_REGISTRATION.json"),
              "rows": output, "actual_c0_certification": True,
              "old_correlated_public_states": True, "new_scientific_training_data": False,
              "model_trained": False, "stable_online_gain": False}
    with (BASE / "PICKUP_TEACHER_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"states": len(output), "candidates": [list(map(lambda s: len(s["candidates"]), x["steps"])) for x in output],
                      "selected": [[s["selected_action"] for s in x["steps"]] for x in output],
                      "final_utility": [x["final_certificate"]["evaluation"]["utility"] for x in output]}))


if __name__ == "__main__":
    main()
