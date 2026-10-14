"""Independent saved-evidence audit for round r15; no model, C0 or request run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
PICK = TREE / "goal_round_20261014_r15_component"
DEL = TREE / "delivery/source_manifest_timing_root_v1"
OUT = PICK / "INDEPENDENT_R15_AUDIT.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def unpack(x):
    if isinstance(x, dict) and "__mapping__" in x:
        return {unpack(a): unpack(b) for a, b in x["__mapping__"]}
    if isinstance(x, dict) and ("__array__" in x or "__sequence__" in x):
        return unpack(x["values"])
    if isinstance(x, list):
        return [unpack(y) for y in x]
    return x


def main():
    pr = read(PICK / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json")
    p = read(PICK / "PICKUP_DISTANCE_REPAIR_RESULT.json")
    dr = read(DEL / "REGISTRATION.json")
    d = read(DEL / "RESULT.json")
    receipt = read(DEL / "COLLECT_RECEIPT.json")
    run = read(DEL / "RUN_RESPONSE.json")
    timing = read(TREE / "goal_round_20261014_r14_boundary/DELIVERY_POST_SELECTION_TIMING.json")
    owner_path = TREE / "pickup/strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"
    model_path = TREE / "pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json"
    owner = unpack(read(owner_path))["scoring_packet"]
    model = unpack(read(model_path))
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    check("pickup_registration_binding", p["registration_sha256"] == sha(PICK / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json"))
    check("pickup_producer_binding", sha(ROOT / "tools/run_pickup_distance_repair_r15.py") == pr["script_sha256"])
    check("pickup_old_input_bindings", sha(owner_path) == pr["owner_sha256"] and sha(model_path) == pr["model_result_sha256"])
    check("pickup_sources_unchanged", all(sha(ROOT / key) == value for key, value in pr["source_sha256"].items()))
    check("pickup_public_order", owner["admission_input"]["current_order"] == 77)
    check("pickup_mask_fixed", pr["masked_orders"] == [25, 30])
    original = owner["plan"]
    geometry = p["pure_distance_candidate"]
    chosen = p["model_candidate"]
    check("pickup_geometry_returns_original", geometry == original)
    check("pickup_model_matches_saved_pretrained", chosen == model["records"][0]["actual"]["raw_candidate"])
    check("pickup_pretrained_random_same_candidate", chosen == model["records"][1]["actual"]["raw_candidate"])
    check("pickup_both_orders_once", sorted(y for route in geometry.values() for y in route if y in [25, 30]) == [25, 30])
    edges = {(min(a, b), max(a, b)) for route in geometry.values() for a, b in zip([0] + route, route + [0]) if a != b}
    check("pickup_retained_edges", {tuple(x) for x in p["retained_original_edges"]}.issubset(edges))
    check("pickup_geometry_qualified", p["pure_distance_certificate"]["qualified"] and p["pure_distance_certificate"]["evaluation"]["hard_feasible"])
    check("pickup_model_qualified", p["model_certificate"]["qualified"] and p["model_certificate"]["evaluation"]["hard_feasible"])
    check("pickup_original_certificate_identity", p["pure_distance_certificate"] == p["original_certificate"])
    delta = p["model_certificate"]["evaluation"]["utility"] - p["pure_distance_certificate"]["evaluation"]["utility"]
    check("pickup_known_delta", abs(delta - model["records"][0]["actual"]["known_utility_delta"]) < 1e-10)
    check("pickup_ordered_steps", [(x["order"], x["vehicle"], x["position"], x["qualified_action_count"]) for x in p["steps"]] == [(25, 0, 2, 8), (30, 0, 4, 4)])
    check("pickup_scope_not_formal", p["public_old_state_only"] and not p["formal_equal_time_control"] and not p["learned_attribution_qualified"] and not p["realized_online_gain"])

    check("delivery_registration_binding", d["registered_source_start_sha256"] == dr["source_start_sha256"])
    check("delivery_script_binding", sha(DEL / "measure_manifest_hash_time_v1.py") == dr["script_sha256"])
    check("delivery_readonly_collection", receipt["readonly"] and receipt["sha256"] == sha(DEL / "RESULT.json") and receipt["bytes"] == (DEL / "RESULT.json").stat().st_size)
    check("delivery_process_success", run["exit_code"] == 0 and not run["stderr"])
    check("delivery_fixed_file_count", d["file_count"] == dr["expected_file_count"] == 8903)
    check("delivery_bytes_mismatch", d["bytes_hashed"] == 892655800 and d["mismatch_count"] == 0 and not d["mismatches"])
    check("delivery_resource_and_scope", d["cpu_affinity"] == [15] and d["available_ram_gib_before"] >= dr["ram_gate_gib"] and d["no_model_or_request_or_C0"])
    check("delivery_isolated_hash_only", d["single_measurement"] and d["source_close_gate_duration"] is None and not d["full_8s_request_qualified"])
    slack = [row["stage6_to_deadline"]["seconds"] for row in timing["rows"]]
    check("delivery_prior_two_slacks", len(slack) == 2 and abs(slack[0]-1.2212045001797378) < 1e-12 and abs(slack[1]-1.164393899962306) < 1e-12)
    check("delivery_hash_exceeds_both_prior_slacks", d["hash_elapsed_seconds"] > max(slack) > 0)
    check("delivery_no_science_upgrade", not d["scientific_family3_started"])

    output = {"checks": len(checks), "failures": sum(not x["passed"] for x in checks),
              "items": checks, "pickup_model_minus_distance_utility": delta,
              "delivery_hash_elapsed_seconds": d["hash_elapsed_seconds"],
              "prior_stage6_remaining_seconds": slack,
              "saved_only": True, "full_source_close_gate_measured": False,
              "stable_online_gain": False}
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checks": output["checks"], "failures": output["failures"], "delta": delta}))
    if output["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
