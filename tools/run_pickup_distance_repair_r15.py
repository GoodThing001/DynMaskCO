"""Fixed old public state: actual C0-certified geometry-only masked repair.

Engineering mechanism comparator only. It is not a fresh day, trained control,
or equal-time MaskCO experiment. The registered order77 fixture is immutable.
"""
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
PICKUP = ROOT / "research/review_3080_20261010/pickup"
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r15_component"
OWNER = PICKUP / "strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"
MODEL = PICKUP / "learned_after_strong_recovery_execution_root_v2/RESULT.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(x):
    if isinstance(x, dict) and "__mapping__" in x:
        return {unpack(a): unpack(b) for a, b in x["__mapping__"]}
    if isinstance(x, dict) and "__array__" in x:
        return unpack(x["values"])
    if isinstance(x, dict) and "__sequence__" in x:
        return [unpack(y) for y in x["values"]]
    if isinstance(x, list):
        return [unpack(y) for y in x]
    return x


def main():
    import psutil
    psutil.Process().cpu_affinity([7])
    reg = json.loads((BASE / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == reg["script_sha256"] and sha(OWNER) == reg["owner_sha256"]
    assert sha(MODEL) == reg["model_result_sha256"] and reg["masked_orders"] == [25, 30]
    sys.path.insert(0, str(ROOT / "tools"))
    import qualify_pickup_joint_policy_transaction_root_v3 as decode_source
    _, decode = decode_source.codecs()
    pf = decode_source.pf
    packet = decode(json.loads(OWNER.read_text(encoding="utf-8")))["scoring_packet"]
    pf.validate_public(packet)
    original = copy.deepcopy(packet["plan"])
    model = unpack(json.loads(MODEL.read_text(encoding="utf-8")))
    assert packet["admission_input"]["current_order"] == 77
    assert [x["mode"] for x in model["records"]] == ["pretrained", "random"]
    assert all(x["actual"]["proposal"]["masked_orders"] == [25, 30] for x in model["records"])
    assert model["records"][0]["actual"]["raw_candidate"] == model["records"][1]["actual"]["raw_candidate"]
    model_candidate = model["records"][0]["actual"]["raw_candidate"]
    original_cert, _ = pf.certificate(packet, original)
    model_cert, _ = pf.certificate(packet, model_candidate)
    assert original_cert["qualified"] and model_cert["qualified"]
    start_source = {name: sha(ROOT / name) for name in reg["source_files"]}
    assert start_source == reg["source_sha256"]
    ids = sorted({o for route in original.values() for o in route})
    ds = packet["dataset"]
    points = [ds["coords"][0, 0]] + [ds["coords"][0, o] for o in ids]
    loc = {o: i + 1 for i, o in enumerate(ids)}
    def distance(a, b):
        return float(math.hypot(float(points[a][0]) - float(points[b][0]), float(points[a][1]) - float(points[b][1])))
    geom_scale = sum(distance(i, j) for i in range(len(points)) for j in range(len(points)) if i != j) / (len(points) * (len(points) - 1))
    assert geom_scale > 0
    required = set()
    for route in original.values():
        for a, b in zip([0] + route, route + [0]):
            if a not in (25, 30) and b not in (25, 30) and a != b:
                required.add((min(a, b), max(a, b)))
    current = {v: [o for o in route if o not in (25, 30)] for v, route in original.items()}
    remaining = [25, 30]
    steps = []
    started = time.perf_counter()
    for step in range(2):
        choices = []
        for o in remaining:
            for v in [0, 1, 2]:
                route = current[str(v)]
                for pos in range(len(route) + 1):
                    cn = int(packet["snapshot"]["committed_next"][v])
                    if packet["snapshot"]["vehicle_status"][v] == "committed" and cn > 0 and pos == 0:
                        continue
                    a = route[pos - 1] if pos else 0
                    b = route[pos] if pos < len(route) else 0
                    if a != b and (min(a, b), max(a, b)) in required:
                        continue
                    trial = copy.deepcopy(current)
                    trial[str(v)].insert(pos, o)
                    for pending in remaining:
                        if pending != o:
                            trial[str(packet["ownership"][str(pending)])].append(pending)
                    try:
                        cert, _ = pf.certificate(packet, trial)
                    except ValueError:
                        continue
                    if not cert["qualified"]:
                        continue
                    delta = (distance(loc[a] if a else 0, loc[o]) + distance(loc[o], loc[b] if b else 0)
                             - distance(loc[a] if a else 0, loc[b] if b else 0)) / geom_scale
                    choices.append((delta, o, v, pos))
        assert choices
        score, o, v, pos = min(choices)
        current[str(v)].insert(pos, o)
        remaining.remove(o)
        steps.append({"step": step, "order": o, "vehicle": v, "position": pos,
                      "normalized_distance_delta": score, "qualified_action_count": len(choices)})
    certified, _ = pf.certificate(packet, current)
    assert certified["qualified"] and all(r["evaluation"]["utility"] == model_cert["evaluation"]["utility"] for r in [model_cert])
    final_edges = {(min(a, b), max(a, b)) for route in current.values() for a, b in zip([0] + route, route + [0]) if a != b}
    assert required <= final_edges
    end_source = {name: sha(ROOT / name) for name in reg["source_files"]}
    assert end_source == start_source and sha(OWNER) == reg["owner_sha256"] and sha(MODEL) == reg["model_result_sha256"]
    out = {"registration_sha256": sha(BASE / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json"),
           "steps": steps, "pure_distance_candidate": current, "pure_distance_certificate": certified,
           "model_candidate": model_candidate, "model_certificate": model_cert,
           "original_certificate": original_cert, "retained_original_edges": sorted(required),
           "retained_edges_preserved": True, "source_start_end_equal": True,
           "wall_seconds": time.perf_counter() - started, "public_old_state_only": True,
           "formal_equal_time_control": False, "learned_attribution_qualified": False,
           "realized_online_gain": False}
    with (BASE / "PICKUP_DISTANCE_REPAIR_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({"steps": steps, "distance_utility": certified["evaluation"]["utility"],
                      "model_utility": model_cert["evaluation"]["utility"], "wall_seconds": out["wall_seconds"]}))


if __name__ == "__main__":
    main()
