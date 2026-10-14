"""Independent r18 saved identity, C0 final plan, and remote receipt audit."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r18_pair_shard"
DEL = ROOT / "research/review_3080_20261010/delivery/source_hash_sharded_root_v1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    reg = read(BASE / "INDEPENDENT_REGISTRATION.json")
    assert sha(__file__) == reg["audit_script_sha256"]
    assert all(sha(ROOT / rel) == hash_ for rel, hash_ in reg["input_sha256"].items())
    sys.path.insert(0, str(ROOT / "tools"))
    import qualify_pickup_joint_policy_transaction_root_v3 as q
    _, decode = q.codecs()
    preg, pres = read(BASE / "PICKUP_REGISTRATION.json"), read(BASE / "PICKUP_RESULT.json")
    dreg, dres = read(DEL / "REGISTRATION.json"), read(DEL / "RESULT.json")
    checks = []

    def ck(name, good):
        checks.append({"name": name, "passed": bool(good)})

    ck("pickup_registration_identity", pres["registration_sha256"] == sha(BASE / "PICKUP_REGISTRATION.json"))
    ck("pickup_program_identity", preg["script_sha256"] == sha(ROOT / "tools/build_pickup_interior_pair_c0_sweep_r18.py"))
    ck("pickup_mask_exhaustive_fixed", preg["masks"] == [[40, 25], [40, 32], [40, 30], [25, 32], [32, 30]])
    ck("pickup_two_old_states", [r["current_order"] for r in pres["rows"]] == [69, 77] and pres["old_correlated_states_only"])
    for i, row in enumerate(pres["rows"]):
        packet = decode(read(ROOT / preg["input_paths"][i]))["scoring_packet"]
        ck(f"pickup_{i}_input", row["input_sha256"] == sha(ROOT / preg["input_paths"][i]))
        ck(f"pickup_{i}_pair_count", len(row["pairs"]) == 5)
        for j, pair in enumerate(row["pairs"]):
            cert, _ = q.pf.certificate(packet, pair["final_plan"])
            old, _ = q.pf.certificate(packet, pair["original_plan"])
            ck(f"pickup_{i}_{j}_mask", pair["masked"] == preg["masks"][j])
            ck(f"pickup_{i}_{j}_actual_final_c0", cert == pair["final_certificate"] and cert["qualified"])
            ck(f"pickup_{i}_{j}_delta", pair["delta_c0_vs_original"] == cert["evaluation"]["utility"] - old["evaluation"]["utility"])
            ck(f"pickup_{i}_{j}_step_argmax", len(pair["steps"]) == 2 and all(
                step["selected_action"] == min(step["candidates"], key=lambda x: (-x["c0_utility"], x["action"]))["action"]
                for step in pair["steps"]))
            ck(f"pickup_{i}_{j}_all_hard", all(c["hard_feasible"] for step in pair["steps"] for c in step["candidates"]))
            ck(f"pickup_{i}_{j}_retained", pair["retained_edges_preserved"])
    ck("pickup_not_science", not pres["new_scientific_family_started"] and not pres["model_training_or_online_gain"])

    ck("delivery_registration_identity", dres["registration_sha256"] == sha(DEL / "REGISTRATION.json"))
    ck("delivery_script_identity", dreg["script_sha256"] == sha(DEL / "measure_source_shards_v1.py"))
    ck("delivery_upload", read(DEL / "UPLOAD_ACCEPTANCE.json")["verified"])
    ck("delivery_exit", read(DEL / "RUN_RESPONSE.json")["exit_code"] == 0)
    pre = json.loads(read(DEL / "RESOURCE_PREFLIGHT.json")["stdout"])
    ck("delivery_preflight", pre["ram_gib"] >= 8 and not pre["owned"] and all(pre["cpu"][str(i)] <= 20 and i in pre["allowed"] for i in [10, 11, 12, 13, 15]))
    ck("delivery_receipts", all(read(DEL / (name + ".receipt.json"))["sha256"] == sha(DEL / name)
                                  for name in ["RESULT.json", "CPU_CHILD_RESULT.json", "SHARD_0.json", "SHARD_3.json"]))
    ck("delivery_missing_shards", all(read(DEL / f"SHARD_{i}.json.missing.json")["remote_missing"] for i in [1, 2]))
    ck("delivery_shards", [s is not None for s in dres["shards"]] == [True, False, False, True]
                            and all(s["mismatch_count"] == 0 for s in dres["shards"] if s))
    ck("delivery_deadline", not dres["all_source_files_verified"] and not dres["receipts"]["hash_1"]["completed_by_8s"]
                             and not dres["receipts"]["hash_2"]["completed_by_8s"])
    ck("delivery_reaped", all(r["reaped"] for r in dres["receipts"].values()) and dres["within_8_plus_2"])
    ck("delivery_cpu", dres["receipts"]["cpu"]["completed_by_8s"] and dres["cpu_result"]["cpu_seconds"] >= 4.9)
    ck("delivery_not_actual_request", dres["synthetic_stress_only"] and not dres["actual_model_or_request"]
                                     and not dres["full_source_or_parameter_close_gate_qualified"])
    out = {"checks": len(checks), "failures": sum(not c["passed"] for c in checks), "items": checks,
           "independent_final_c0_recomputations": 10, "stable_online_gain": False,
           "learning_attribution": False, "equal_time_iteration": False,
           "genuine_independent_strong_dynamic": False, "delivery_full_request_qualified": False}
    with (BASE / "INDEPENDENT_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"checks": out["checks"], "failures": out["failures"]}))
    if out["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
