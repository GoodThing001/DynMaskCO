"""Independent r17 source/result audit with only two final C0 recomputations."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r17_teacher_dualcore"
DEL = TREE / "delivery/source_overlap_dualcore_root_v1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    sys.path.insert(0, str(ROOT / "tools"))
    import qualify_pickup_joint_policy_transaction_root_v3 as q
    _, decode = q.codecs()
    preg = read(BASE / "PICKUP_TEACHER_REGISTRATION.json")
    p = read(BASE / "PICKUP_TEACHER_RESULT.json")
    dreg = read(DEL / "REGISTRATION.json")
    d = read(DEL / "RESULT.json")
    checks = []

    def ck(name, good):
        checks.append({"name": name, "passed": bool(good)})

    ck("pickup_registration", p["registration_sha256"] == sha(BASE / "PICKUP_TEACHER_REGISTRATION.json"))
    ck("pickup_program", preg["script_sha256"] == sha(ROOT / "tools/build_pickup_two_state_c0_teacher_r17.py"))
    ck("pickup_inputs", all(sha(ROOT / name) == preg["input_sha256"][name] for name in preg["input_paths"]))
    ck("pickup_order", [x["current_order"] for x in p["rows"]] == [69, 77])
    ck("pickup_mask", all(x["masked_orders"] == [25, 30] for x in p["rows"]))
    for idx, row in enumerate(p["rows"]):
        packet = decode(read(ROOT / preg["input_paths"][idx]))["scoring_packet"]
        ck(f"pickup_{idx}_original", row["original_plan"] == packet["plan"])
        ck(f"pickup_{idx}_candidate_counts", [len(x["candidates"]) for x in row["steps"]] == [8, 5])
        ck(f"pickup_{idx}_step_labels", [x["selected_action"] for x in row["steps"]] == [[30, 0, 3], [25, 0, 2]])
        for step in row["steps"]:
            choices = step["candidates"]
            best = min(choices, key=lambda x: (-x["c0_utility"], x["action"]))
            ck(f"pickup_{idx}_{step['step']}_argmax", step["selected_action"] == best["action"] and step["selected_c0_utility"] == best["c0_utility"])
            ck(f"pickup_{idx}_{step['step']}_all_c0", all(x["certificate"]["qualified"] and x["certificate"]["evaluation"]["hard_feasible"] and x["c0_utility"] == x["certificate"]["evaluation"]["utility"] for x in choices))
        actual, _ = q.pf.certificate(packet, row["final_plan"])
        ck(f"pickup_{idx}_final_actual_c0", actual == row["final_certificate"] and actual["qualified"])
        edges = {(min(a, b), max(a, b)) for route in row["final_plan"].values() for a, b in zip([0] + route, route + [0]) if a != b}
        ck(f"pickup_{idx}_edges", {tuple(x) for x in row["retained_edges"]}.issubset(edges))
        ck(f"pickup_{idx}_return_original", row["final_plan"] == row["original_plan"])
    ck("pickup_scope", p["actual_c0_certification"] and p["old_correlated_public_states"] and not p["model_trained"] and not p["new_scientific_training_data"])

    ck("delivery_registration", d["registration_sha256"] == sha(DEL / "REGISTRATION.json"))
    ck("delivery_program", dreg["script_sha256"] == sha(DEL / "measure_overlap_dualcore_v1.py"))
    ck("delivery_preflight", (lambda x: x["ram_gib"] >= 8 and x["cpu14"] <= 20 and x["cpu15"] <= 20 and not x["owned"])(json.loads(read(DEL / "RESOURCE_PREFLIGHT.json")["stdout"])))
    ck("delivery_remote_exit", read(DEL / "RUN_RESPONSE.json")["exit_code"] == 0)
    ck("delivery_collected_hashes", all(read(DEL / (name + ".receipt.json"))["sha256"] == sha(DEL / name) for name in ["RESULT.json", "CPU_CHILD_RESULT.json"]))
    ck("delivery_hash_no_completion", not d["receipts"]["hash"]["completed_by_8s"] and d["hash_result"] is None and read(DEL / "HASH_CHILD_RESULT.json.missing.json")["remote_missing"])
    ck("delivery_cpu_completed", d["receipts"]["cpu"]["completed_by_8s"] and d["cpu_result"]["cpu_seconds"] >= dreg["synthetic_cpu_seconds"])
    ck("delivery_reaped", all(x["reaped"] for x in d["receipts"].values()))
    ck("delivery_parent_time", d["within_8_plus_2"] and 8 <= d["parent_elapsed_seconds"] <= 10)
    ck("delivery_only_engineering", d["synthetic_stress_only"] and not d["actual_model_or_request"] and not d["full_source_or_parameter_close_gate_qualified"])
    ck("no_science_upgrade", not d["scientific_family3_started"] and not p["stable_online_gain"])
    out = {"checks": len(checks), "failures": sum(not x["passed"] for x in checks),
           "items": checks, "final_c0_recomputations": 2,
           "full_teacher_training_qualified": False, "full_delivery_request_qualified": False,
           "stable_online_gain": False}
    with (BASE / "INDEPENDENT_R17_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"checks": out["checks"], "failures": out["failures"]}))
    if out["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
