"""Saved-only geometry/model ranking decomposition for the fixed order-77 mask."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"
PICKUP = ROOT / "research/review_3080_20261010/pickup"
MODEL = PICKUP / "learned_after_strong_recovery_execution_root_v2/RESULT.json"
OWNER = PICKUP / "strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(x):
    if isinstance(x, dict) and "__array__" in x:
        return unpack(x["values"])
    if isinstance(x, dict) and "__mapping__" in x:
        return {unpack(k): unpack(v) for k, v in x["__mapping__"]}
    if isinstance(x, dict) and "__sequence__" in x:
        return [unpack(y) for y in x["values"]]
    if isinstance(x, list):
        return [unpack(y) for y in x]
    return x


def main():
    reg = json.loads((BASE / "PICKUP_RANK_RECOVERY_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == reg["script_sha256"] and sha(MODEL) == reg["model_result_sha256"]
    assert sha(OWNER) == reg["owner_response_sha256"] and reg["masked_orders"] == [25, 30]
    saved = unpack(json.loads(MODEL.read_text(encoding="utf-8")))
    packet = unpack(json.loads(OWNER.read_text(encoding="utf-8")))["scoring_packet"]
    routes = packet["plan"]
    ids = sorted({int(o) for route in routes.values() for o in route})
    coords = packet["dataset"]["coords"]["values"][0]
    points = [coords[0]] + [coords[o] for o in ids]
    loc = {order: i + 1 for i, order in enumerate(ids)}
    def distance(a, b):
        p, q = points[a], points[b]
        return math.hypot(p[0] - q[0], p[1] - q[1])
    scale = sum(distance(i, j) for i in range(len(points)) for j in range(len(points)) if i != j) / (len(points) * (len(points) - 1))
    assert scale > 0
    rows = []
    for record in saved["records"]:
        mode = record["mode"]
        proposal = record["actual"]["proposal"]
        audit = proposal["audit"]
        assert len(audit) == 10 and proposal["masked_orders"] == [25, 30]
        now = {v: [o for o in route if o not in (25, 30)] for v, route in routes.items()}
        phases = []
        for phase, entries in enumerate((audit[:6], audit[6:])):
            candidates = []
            for item in entries:
                vehicle, pos, order = item["vehicle"], item["position"], item["order"]
                route = now[str(vehicle)]
                a = loc[route[pos - 1]] if pos else 0
                b = loc[route[pos]] if pos < len(route) else 0
                k = loc[order]
                geometry = (distance(a, k) + distance(k, b) - distance(a, b)) / scale
                candidates.append({"action": [order, vehicle, pos], "geometry": geometry,
                                   "score": item["proposal_score"], "inferred_normalized_model_gain": geometry - item["proposal_score"]})
            selected = min(candidates, key=lambda x: (x["score"], *x["action"]))
            distance_only = min(candidates, key=lambda x: (x["geometry"], *x["action"]))
            now[str(selected["action"][1])].insert(selected["action"][2], selected["action"][0])
            phases.append({"step": phase, "selected": selected["action"], "distance_only_choice": distance_only["action"],
                           "model_changes_distance_choice": selected["action"] != distance_only["action"],
                           "candidate_count": len(candidates), "candidates": candidates})
        assert now == record["actual"]["raw_candidate"]
        rows.append({"mode": mode, "phases": phases})
    assert [x["mode"] for x in rows] == ["pretrained", "random"]
    comparisons = []
    for phase in range(2):
        left = rows[0]["phases"][phase]["candidates"]
        right = rows[1]["phases"][phase]["candidates"]
        assert [x["action"] for x in left] == [x["action"] for x in right]
        assert all(abs(a["geometry"] - b["geometry"]) < 1e-12 for a, b in zip(left, right))
        pairs = 0
        discordant = 0
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                x = left[i]["score"] - left[j]["score"]
                y = right[i]["score"] - right[j]["score"]
                if x and y:
                    pairs += 1
                    discordant += (x < 0) != (y < 0)
        comparisons.append({"step": phase, "comparable_pairs": pairs, "discordant_pairs": discordant})
    out = {"registration_sha256": sha(BASE / "PICKUP_RANK_RECOVERY_REGISTRATION.json"), "script_sha256": sha(__file__),
           "input_sha256": {"model_result": sha(MODEL), "owner_response": sha(OWNER)},
           "distance_scale": scale, "rows": rows, "between_model_rankings": comparisons,
           "saved_only": True, "model_or_C0_rerun": False, "online_gain_qualified": False}
    with (BASE / "PICKUP_RANK_DIAGNOSIS.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({"distance_only_choices": [[p["distance_only_choice"] for p in r["phases"]] for r in rows],
                      "model_choices": [[p["selected"] for p in r["phases"]] for r in rows],
                      "discordant_pairs": [c["discordant_pairs"] for c in comparisons]}))


if __name__ == "__main__":
    main()
