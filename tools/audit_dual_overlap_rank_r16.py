"""Independent saved-only r16 audit; no model, C0, source scan or request rerun."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r16_overlap_rank"
DEL = TREE / "delivery/source_overlap_cpu15_root_v1"
MODEL = TREE / "pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def unpack(x):
    if isinstance(x, dict) and "__mapping__" in x:
        return {unpack(a): unpack(b) for a, b in x["__mapping__"]}
    if isinstance(x, dict) and ("__array__" in x or "__sequence__" in x):
        return unpack(x["values"])
    if isinstance(x, list):
        return [unpack(y) for y in x]
    return x


def main():
    preg = read(BASE / "PICKUP_C0_RANK_REGISTRATION.json")
    pres = read(BASE / "PICKUP_C0_RANK_RESULT.json")
    dreg = read(DEL / "REGISTRATION.json")
    dres = read(DEL / "RESULT.json")
    run = read(DEL / "RUN_RESPONSE.json")
    old = unpack(read(MODEL))
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    check("pickup_registration", pres["registration_sha256"] == sha(BASE / "PICKUP_C0_RANK_REGISTRATION.json"))
    check("pickup_script", preg["script_sha256"] == sha(ROOT / "tools/diagnose_pickup_saved_c0_rank_r16.py"))
    check("pickup_inputs", preg["model_sha256"] == sha(MODEL) and preg["prior_rank_sha256"] == pres["prior_rank_sha256"])
    check("pickup_phase_counts", preg["phase_counts"] == [6, 4])
    check("pickup_modes", [x["mode"] for x in pres["rows"]] == [x["mode"] for x in old["records"]] == ["pretrained", "random"])
    for i, row in enumerate(pres["rows"]):
        audit = old["records"][i]["actual"]["proposal"]["audit"]
        for phase in row["phases"]:
            step = phase["step"]
            subset = audit[0:6] if step == 0 else audit[6:10]
            keys = lambda a: (a["order"], a["vehicle"], a["position"])
            by_score = min(subset, key=lambda a: (a["proposal_score"], *keys(a)))
            by_utility = max(subset, key=lambda a: (a["certificate"]["evaluation"]["utility"], *[-n for n in keys(a)]))
            check(f"pickup_{i}_{step}_model_choice", tuple(phase["model_choice"]) == keys(by_score))
            check(f"pickup_{i}_{step}_c0_best", tuple(phase["c0_best_action"]) == keys(by_utility))
            check(f"pickup_{i}_{step}_regret", abs(phase["one_step_c0_regret"] - (by_utility["certificate"]["evaluation"]["utility"] - by_score["certificate"]["evaluation"]["utility"])) < 1e-12)
            opposite = total = 0
            for j, a in enumerate(subset):
                for b in subset[j+1:]:
                    ds = a["proposal_score"] - b["proposal_score"]
                    du = a["certificate"]["evaluation"]["utility"] - b["certificate"]["evaluation"]["utility"]
                    if ds and du:
                        total += 1
                        opposite += ds * du > 0
            check(f"pickup_{i}_{step}_pairs", (phase["discordant_model_c0_pairs"], phase["comparable_pairs"]) == (opposite, total))
    check("pickup_scope", pres["saved_only"] and pres["no_model_or_C0_rerun"] and not pres["learned_attribution_qualified"])

    check("delivery_registration", dres["registration_sha256"] == sha(DEL / "REGISTRATION.json"))
    check("delivery_script", dreg["script_sha256"] == sha(DEL / "measure_overlap_cpu15_v1.py"))
    check("delivery_upload", read(DEL / "UPLOAD_ACCEPTANCE.json")["verified"])
    check("delivery_run", run["exit_code"] == 0 and not run["timed_out"] if "timed_out" in run else run["exit_code"] == 0)
    check("delivery_collection", all(read(DEL / (name + ".receipt.json"))["sha256"] == sha(DEL / name) for name in ("RESULT.json", "CPU_CHILD_RESULT.json")))
    check("delivery_hash_incomplete", not dres["receipts"]["hash"]["completed_by_8s"] and dres["hash_result"] is None and read(DEL / "HASH_CHILD_RESULT.json.missing.json")["remote_missing"])
    check("delivery_cpu_complete", dres["receipts"]["cpu"]["completed_by_8s"] and dres["cpu_result"]["cpu_seconds"] >= dreg["synthetic_cpu_seconds"])
    check("delivery_owned_reaped", all(v["reaped"] for v in dres["receipts"].values()))
    check("delivery_budget", dres["within_8_plus_2"] and dres["parent_elapsed_seconds"] <= 10 and dres["parent_elapsed_seconds"] >= 8)
    check("delivery_resource", dres["available_ram_gib_before"] >= 8 and dres["cpu_affinity_parent"] == [15])
    check("delivery_scope", dres["synthetic_stress_only"] and not dres["actual_model_or_request"] and not dres["full_source_or_parameter_close_gate_qualified"])
    check("no_science", not dres["scientific_family3_started"])
    out = {"checks": len(checks), "failures": sum(not x["passed"] for x in checks),
           "items": checks, "saved_only": True, "stable_online_gain": False}
    with (BASE / "INDEPENDENT_R16_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"checks": out["checks"], "failures": out["failures"]}))
    if out["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
