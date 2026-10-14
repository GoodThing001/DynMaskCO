"""Read-only, independent release audit of the saved pickup and delivery stages."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "research/review_3080_20261010/pickup"
D = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r19_science_release"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prereg_path = P / "A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json"
    stage_dir = P / "a1_budget_reserving_admission_root_v4"
    stage_path = stage_dir / "REGISTRATION.json"
    result_path = stage_dir / "RESULT.json"
    audit_path = stage_dir / "INDEPENDENT_AUDIT.json"
    addendum_path = stage_dir / "AUDIT_FAILURE_ADDENDUM.json"
    source_path = stage_dir / "reserving_admission.py"
    delivery_path = D / "downloaded/RESULT.json"
    trace_paths = [D / f"downloaded/{arm}_fault0_worker/STAGE_6.json"
                   for arm in ("pretrained_initial", "random888_initial")]
    prereg, stage, result, audit, addendum = map(read, (prereg_path, stage_path, result_path, audit_path, addendum_path))
    delivery = read(delivery_path)
    source = source_path.read_text(encoding="utf-8")
    assert prereg["status"] == "declared_before_any_execution"
    assert stage["panel"]["day_indices"] == list(range(5))
    assert "e_lower(i) <= B_remaining - reserve_for_future(clock, history)" in prereg["candidate_rule"]["accept_condition"]
    assert "REV[cls] - lam * e" in source and "_lambda_for" in source
    assert "revenue exceeds lambda(clock)*est" in stage["mechanism"]
    assert audit["failed"] == 2 and addendum["audit_failed"] == 2
    paired = result["paired"]
    assert [r["day"] for r in paired] == list(range(5))
    deltas = [r["candidate_utility"] - r["control_utility"] for r in paired]
    assert all(abs(a-b["delta"]) < 1e-9 for a,b in zip(deltas,paired))
    mean = sum(deltas)/len(deltas)
    assert abs(mean-result["mean_delta"]) < 1e-9
    active = [x for x in deltas if abs(x)>1e-9]
    # Conditional randomisation of signs on the nonzero paired observations;
    # descriptive only, because this developer panel was not a randomised day sample.
    signs = [sum(s*x for s,x in zip(bits,active))/len(deltas)
             for bits in itertools.product((-1,1),repeat=len(active))]
    p_one = sum(x >= mean-1e-9 for x in signs)/len(signs)
    assert len(delivery["receipts"]) == 2
    traces = [read(p) for p in trace_paths]
    remaining = [t["deadline"]-t["clock"] for t in traces]
    assert all(t["stage"] == 6 for t in traces)
    assert delivery["selected_count"] == 0 and not delivery["full_request_budget_qualified"]
    report = {
        "stage": "independent_saved_science_release_r19",
        "inputs": [identity(p) for p in (prereg_path,stage_path,result_path,audit_path,addendum_path,source_path,delivery_path,*trace_paths)],
        "pickup": {
            "five_day_deltas": deltas,
            "mean_delta": mean,
            "active_days": len(active),
            "conditional_sign_flip_one_sided_p_descriptive": p_one,
            "original_prereg_rule": prereg["candidate_rule"]["accept_condition"],
            "actual_rule": "smallest clock-dependent grid lambda with expected accepted energy <= remaining budget; accept when REV-lambda*e_lower>0",
            "same_rule_as_original_prereg": False,
            "stage_specific_rule_registered_before_stage": bool(stage["panel"]["fixed_before_execution"]),
            "original_audit_failed": audit["failed"],
            "audit_failure_names": audit["failure_names"],
            "automatic_40_day_confirmation_release": False,
            "reason": "mechanism differs from original preregistration; stage-specific 5-day lead is exploratory and original audit retains two FAIL; fresh independent design/controls and preregistration needed"
        },
        "delivery": {
            "saved_initial_arms": [r["arm"] for r in delivery["receipts"]],
            "stage6_seconds_remaining": remaining,
            "selected_count": delivery["selected_count"],
            "whole_request_budget_qualified": delivery["full_request_budget_qualified"],
            "prior_independent_split_audit_failed": read(D/"INDEPENDENT_SAVED_SPLIT_AUDIT.json")["failed"],
            "source_shard_synthetic_not_release": True,
            "scientific_family3_release": False
        },
        "stable_core_online_gain": False,
        "training_attribution_qualified": False,
        "equal_time_iterative_reconstruction_qualified": False,
        "independent_strong_dynamic_accept_reject_qualified": False,
    }
    OUT.mkdir(parents=True)
    with (OUT/"RELEASE_AUDIT.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"pickup_mean":mean,"pickup_original_rule_match":False,"sign_flip_p":p_one,
                      "delivery_stage6_remaining":remaining,"science_release":False}))


if __name__ == "__main__":
    main()
