"""Independent minimal replay of registered pickup geometry and delivery clock bounds."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"
PICKUP = ROOT / "research/review_3080_20261010/pickup"
DELIVERY = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1/downloaded"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten(value):
    if isinstance(value, dict) and "__mapping__" in value:
        return {flatten(a): flatten(b) for a, b in value["__mapping__"]}
    if isinstance(value, dict) and "__array__" in value:
        return flatten(value["values"])
    if isinstance(value, dict) and "__sequence__" in value:
        return [flatten(y) for y in value["values"]]
    if isinstance(value, list):
        return [flatten(y) for y in value]
    return value


def main():
    reg = read(BASE / "INDEPENDENT_BOUNDARY_REGISTRATION.json")
    assert sha(__file__) == reg["script_sha256"]
    pickup_out = BASE / "PICKUP_RANK_DIAGNOSIS.json"
    delivery_out = BASE / "DELIVERY_POST_SELECTION_TIMING.json"
    assert sha(pickup_out) == reg["pickup_diagnosis_sha256"]
    assert sha(delivery_out) == reg["delivery_diagnosis_sha256"]
    owner_path = PICKUP / "strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"
    model_path = PICKUP / "learned_after_strong_recovery_execution_root_v2/RESULT.json"
    assert sha(owner_path) == reg["owner_response_sha256"] and sha(model_path) == reg["model_result_sha256"]
    owner = flatten(read(owner_path))["scoring_packet"]
    saved = flatten(read(model_path))
    reported = read(pickup_out)
    points = owner["dataset"]["coords"][0]
    ids = sorted({o for route in owner["plan"].values() for o in route})
    def dist(a, b):
        x, y = points[a], points[b]
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    norm = sum(dist(a, b) for a in [0] + ids for b in [0] + ids if a != b) / ((len(ids) + 1) * len(ids))
    checks = []
    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
    check("distance_norm", abs(norm - reported["distance_scale"]) < 1e-12)
    for index, record in enumerate(saved["records"]):
        trial = {v: [o for o in route if o not in (25, 30)] for v, route in owner["plan"].items()}
        for step, entries in enumerate((record["actual"]["proposal"]["audit"][:6], record["actual"]["proposal"]["audit"][6:])):
            vals = []
            for item in entries:
                route = trial[str(item["vehicle"])]
                pos = item["position"]
                a = route[pos - 1] if pos else 0
                b = route[pos] if pos < len(route) else 0
                o = item["order"]
                vals.append(((dist(a, o) + dist(o, b) - dist(a, b)) / norm,
                             item["proposal_score"], [o, item["vehicle"], pos]))
            geo = min(vals, key=lambda z: (z[0], *z[2]))[2]
            model = min(vals, key=lambda z: (z[1], *z[2]))[2]
            claim = reported["rows"][index]["phases"][step]
            check(f"{index}:{step}:distance_choice", geo == claim["distance_only_choice"])
            check(f"{index}:{step}:model_choice", model == claim["selected"])
            trial[str(model[1])].insert(model[2], model[0])
    clock_claims = read(delivery_out)["rows"]
    result = read(DELIVERY / "RESULT.json")
    for i, arm in enumerate(reg["arms"]):
        marker = read(DELIVERY / f"{arm}_fault0_worker/STAGE_6.json")
        receipt = result["receipts"][i]
        actual = Fraction.from_float(receipt["child_deadline"]) - Fraction.from_float(marker["clock"])
        claim = clock_claims[i]["stage6_to_deadline"]
        check(f"{arm}:deadline_fraction", (actual.numerator, actual.denominator) == (claim["numerator"], claim["denominator"]))
        check(f"{arm}:no_source_end", not (DELIVERY / f"{arm}_fault0_worker/request/SOURCE_END.json").exists())
        check(f"{arm}:reaped_unselected", receipt["worker_reaped"] and not receipt["selected"])
    out = {"checks": len(checks), "failures": sum(not x["passed"] for x in checks), "items": checks,
           "saved_only": True, "no_model_or_C0_or_request_rerun": True, "stable_online_gain": False}
    with (BASE / "INDEPENDENT_BOUNDARY_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"checks": out["checks"], "failures": out["failures"]}))
    if out["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
