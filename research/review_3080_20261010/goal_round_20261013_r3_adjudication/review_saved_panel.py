"""Independent arithmetic and registration check over already saved JSON only."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "research" / "review_3080_20261010"
HERE = Path(__file__).resolve().parent


def read(relative: str):
    path = TREE / relative
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check(name: str, condition: bool, checks: list[dict]):
    checks.append({"name": name, "pass": bool(condition)})


def main():
    if (HERE / "SAVED_ONLY_ADJUDICATION.json").exists():
        raise FileExistsError("prior adjudication is immutable")
    scope, scope_hash = read("goal_round_20261013_r3_adjudication/REVIEW_SCOPE.json")
    prereg, prereg_hash = read("pickup/A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json")
    reg, reg_hash = read("pickup/a1_budget_reserving_admission_root_v4/REGISTRATION.json")
    result, result_hash = read("pickup/a1_budget_reserving_admission_root_v4/RESULT.json")
    audit, audit_hash = read("pickup/a1_budget_reserving_admission_root_v4/INDEPENDENT_AUDIT.json")
    addendum, addendum_hash = read("pickup/a1_budget_reserving_admission_root_v4/AUDIT_FAILURE_ADDENDUM.json")
    upper, upper_hash = read("pickup/a1_budget_reachability_bound_root_v1/RELAXATION_CAVEAT_ADDENDUM.json")
    evidence, _ = read("goal_round_20261013_r2/PICKUP_BUDGET_RESERVING_PANEL_EVIDENCE.json")
    checks: list[dict] = []
    check("scope_saved_only", not scope["scientific_experiment"] and scope["original_artifacts_immutable"], checks)
    check("registered_source_hash", reg_hash == evidence["registration_sha256"], checks)
    check("original_prereg_hash", prereg_hash == reg["preregistration_sha256"], checks)
    check("result_hash", result_hash == evidence["evidence_hashes"]["research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/RESULT.json"], checks)
    check("audit_hash", audit_hash == evidence["evidence_hashes"]["research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/INDEPENDENT_AUDIT.json"], checks)
    check("preserved_original_audit_failures", audit["failed"] == 2 and set(audit["failure_names"]) == set(addendum["audit_failure_names"]), checks)
    check("original_prereg_rule_is_reserve_threshold", "B_remaining - reserve_for_future" in prereg["candidate_rule"]["accept_condition"], checks)
    check("executed_rule_is_lambda_price", "lambda(clock)" in reg["goal"] or "lambda(clock)" in reg["mechanism"], checks)
    original_control = prereg["decision_rule_declared_before_execution"]["primary"]
    executed_control = reg["decision_rule"]
    check("original_comparator_is_recorded_explicit", "explicit_feat" in original_control, checks)
    check("executed_comparator_is_contemporaneous_control", "candidate - control" in executed_control, checks)
    check("five_fixed_days", [p["day"] for p in result["paired"]] == [0, 1, 2, 3, 4], checks)
    diffs = []
    recorded_diffs = []
    for p in result["paired"]:
        d = float(p["candidate_utility"] - p["control_utility"])
        rd = float(p["candidate_utility"] - p["recorded_rows"]["explicit_feat"]["utility"])
        diffs.append(d)
        recorded_diffs.append(rd)
        check(f"same_day_delta_{p['day']}", d == p["delta"], checks)
        check(f"reported_hard_{p['day']}", result["arms"]["candidate_reserving"][p["day"]]["hard_feasible"] is True and result["arms"]["control_frozen_saa"][p["day"]]["hard_feasible"] is True, checks)
        check(f"reported_timeouts_{p['day']}", result["arms"]["candidate_reserving"][p["day"]]["timeouts"] == 0 and result["arms"]["control_frozen_saa"][p["day"]]["timeouts"] == 0, checks)
    mean = math.fsum(diffs) / len(diffs)
    mean_recorded = math.fsum(recorded_diffs) / len(recorded_diffs)
    check("same_day_mean", abs(mean - result["mean_delta"]) < 1e-12, checks)
    # Exhaustive empirical bootstrap support for five observations. Descriptive only.
    boot = sorted(math.fsum(xs) / len(xs) for xs in itertools.product(diffs, repeat=len(diffs)))
    lo = boot[int(0.025 * (len(boot) - 1))]
    hi = boot[int(0.975 * (len(boot) - 1))]
    check("bootstrap_enumeration_size", len(boot) == 3125, checks)
    check("bootstrap_lower_includes_zero", lo == 0.0, checks)
    audit_budget = addendum["classification"]["audit_budget_matches"]
    check("budget_drift_matches", abs(audit_budget["delta"] - result["budget_identity"]["delta"]) < 1e-12, checks)
    check("upper_vehicle_time_did_not_bind", upper["finding"]["tw_and_horizon"]["time_bound_strictly_tighter"] == 0, checks)
    report = {
        "stage": scope["stage"],
        "input_sha256": {"scope": scope_hash, "initial_prereg": prereg_hash, "executed_registration": reg_hash, "producer_result": result_hash, "original_audit": audit_hash, "audit_addendum": addendum_hash, "bound_caveat": upper_hash},
        "checks_count": len(checks), "failed": sum(not c["pass"] for c in checks),
        "failure_names": [c["name"] for c in checks if not c["pass"]], "checks": checks,
        "observed": {"same_day_deltas": diffs, "same_day_mean": mean, "recorded_explicit_deltas": recorded_diffs, "recorded_explicit_mean": mean_recorded, "positive_days": sum(x > 0 for x in diffs), "zero_days": sum(x == 0 for x in diffs), "negative_days": sum(x < 0 for x in diffs), "exhaustive_empirical_bootstrap_95pct": [lo, hi], "bootstrap_zero_mean_probability": boot.count(0.0) / len(boot)},
        "registration_equivalence": {"equivalent": False, "differences": ["Initial preregistration reserves B_remaining - reserve_for_future, while executed registration uses a lambda price threshold.", "Initial primary comparator is candidate minus recorded explicit_feat, while executed primary comparator is candidate minus contemporaneous frozen SAA.", "The initial fixed recorded B differs from the recomputed B used by both executed arms."]},
        "audit_adjudication": {"original_check_count": audit["checks_count"], "original_failed": audit["failed"], "failure_names": audit["failure_names"], "addendum_classifies_but_does_not_erase_failures": True},
        "upper_bound_adjudication": {"vehicle_time_relaxation_invalid_if_binding": True, "vehicle_time_bound_any_day": False, "upper_minus_recorded_control_is_not_demonstrated_attainable_improvement": True},
        "conclusion": "A contemporaneous five-day engineering A/B has a positive descriptive mean, but this is a non-learning admission filter, has no stable positive inference, and is not a literal execution of the initial rule/comparator preregistration. A 40-day continuation must be newly registered as exploratory under the exact frozen executed rule and source identity; fresh independent confirmations and full MaskCO attribution remain separate gates.",
        "stable_online_gain": False,
        "new_science_family": False,
        "old_artifacts_modified": False,
    }
    with (HERE / "SAVED_ONLY_ADJUDICATION.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({k: report[k] for k in ("checks_count", "failed", "failure_names", "observed", "registration_equivalence")}, ensure_ascii=False))
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
