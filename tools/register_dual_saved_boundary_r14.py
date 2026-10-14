"""Register both bounded saved-only diagnostics before reading their outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"
PICKUP = ROOT / "research/review_3080_20261010/pickup"
DELIVERY = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def main():
    BASE.mkdir(exist_ok=False)
    model = PICKUP / "learned_after_strong_recovery_execution_root_v2/RESULT.json"
    owner = PICKUP / "strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"
    new_json(BASE / "PICKUP_RANK_REGISTRATION.json", {
        "script_sha256": sha(ROOT / "tools/diagnose_pickup_saved_ranking_r14.py"),
        "model_result_sha256": sha(model), "owner_response_sha256": sha(owner),
        "masked_orders": [25, 30], "fixed_two_arms": ["pretrained", "random"],
        "scope": "Saved-only decomposition of the prior naturally reached order77; no model or C0 rerun",
        "hypothesis": "Distinct logits may alter lower rankings while top certified actions remain equal",
        "model_attribution_gain_qualified": False,
    })
    archive = DELIVERY / "downloaded"
    new_json(BASE / "DELIVERY_TIMING_REGISTRATION.json", {
        "script_sha256": sha(ROOT / "tools/diagnose_delivery_saved_post_selection_r14.py"),
        "result_sha256": sha(archive / "RESULT.json"),
        "manifest_sha256": sha(DELIVERY / "DOWNLOAD_MANIFEST.json"),
        "original_audit_sha256": sha(archive / "INDEPENDENT_TIMED_REQUEST_AUDIT.json"),
        "arms": ["pretrained_initial", "random888_initial"],
        "scope": "Saved stage clocks and C0 choice, exact rational differences; no request replay",
        "hypothesis": "Stage6 leaves positive but unqualified time for complete source/parameter/publication gates",
        "full_budget_qualified": False,
    })
    print(json.dumps({"pickup_registration": sha(BASE / "PICKUP_RANK_REGISTRATION.json"),
                      "delivery_registration": sha(BASE / "DELIVERY_TIMING_REGISTRATION.json")}))


if __name__ == "__main__":
    main()
