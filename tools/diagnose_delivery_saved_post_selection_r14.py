"""Bounded saved-clock diagnosis after C0 selection; never reruns a request."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"
OLD = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
ARCHIVE = OLD / "downloaded"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_sub(a, b):
    q = Fraction.from_float(a) - Fraction.from_float(b)
    return {"numerator": q.numerator, "denominator": q.denominator, "seconds": float(q)}


def main():
    reg = read(BASE / "DELIVERY_TIMING_REGISTRATION.json")
    assert sha(__file__) == reg["script_sha256"]
    assert sha(ARCHIVE / "RESULT.json") == reg["result_sha256"]
    assert sha(OLD / "DOWNLOAD_MANIFEST.json") == reg["manifest_sha256"]
    assert sha(ARCHIVE / "INDEPENDENT_TIMED_REQUEST_AUDIT.json") == reg["original_audit_sha256"]
    result = read(ARCHIVE / "RESULT.json")
    assert result["count"] == 2 and result["selected_count"] == 0
    rows = []
    for arm, receipt in zip(reg["arms"], result["receipts"]):
        root = ARCHIVE / f"{arm}_fault0_worker"
        stages = [read(root / f"STAGE_{i}.json") for i in range(1, 7)]
        assert [s["stage"] for s in stages] == list(range(1, 7))
        assert all(s["deadline"] == receipt["child_deadline"] for s in stages)
        assert receipt["actual_stage"] == 6 and receipt["worker_reaped"] and not receipt["selected"]
        assert receipt["child_deadline"] == receipt["start_clock"] + 8
        assert stages[5]["clock"] < receipt["child_deadline"] <= receipt["kill_clock"]
        for stage in stages:
            for relative, expected in stage["evidence_files"].items():
                assert sha(root / relative) == expected
        selection = read(root / "request/C0_SELECTION.json")
        stop = read(root / "request/CHAIN_STOP.json")
        assert stop["remaining"] == [] and stop["status"] == "complete"
        assert not selection["candidate_selected"] and selection["candidate"]["cost"] > selection["incumbent"]["cost"]
        assert not (root / "request/SOURCE_END.json").exists()
        rows.append({"arm": arm, "stage_clocks": {str(s["stage"]): s["clock"] for s in stages},
                     "stage4_to_5": exact_sub(stages[4]["clock"], stages[3]["clock"]),
                     "stage5_to_6": exact_sub(stages[5]["clock"], stages[4]["clock"]),
                     "stage6_to_deadline": exact_sub(receipt["child_deadline"], stages[5]["clock"]),
                     "stage6_to_actual_kill": exact_sub(receipt["kill_clock"], stages[5]["clock"]),
                     "candidate_cost": selection["candidate"]["cost"],
                     "incumbent_cost": selection["incumbent"]["cost"],
                     "source_end_complete": False, "published": False})
    out = {"registration_sha256": sha(BASE / "DELIVERY_TIMING_REGISTRATION.json"),
           "rows": rows, "saved_only": True, "request_rerun": False,
           "post_selection_source_and_parameter_duration_measured": False,
           "full_request_budget_qualified": False, "stable_online_gain": False}
    with (BASE / "DELIVERY_POST_SELECTION_TIMING.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({"stage6_remaining_seconds": [r["stage6_to_deadline"]["seconds"] for r in rows],
                      "stage4_to_5_seconds": [r["stage4_to_5"]["seconds"] for r in rows]}))


if __name__ == "__main__":
    main()
