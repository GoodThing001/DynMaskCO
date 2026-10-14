"""Seal independent replay source and the frozen outputs it checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg = {
        "script_sha256": sha(ROOT / "tools/audit_dual_saved_boundary_r14.py"),
        "pickup_diagnosis_sha256": sha(BASE / "PICKUP_RANK_DIAGNOSIS.json"),
        "delivery_diagnosis_sha256": sha(BASE / "DELIVERY_POST_SELECTION_TIMING.json"),
        "owner_response_sha256": sha(ROOT / "research/review_3080_20261010/pickup/strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"),
        "model_result_sha256": sha(ROOT / "research/review_3080_20261010/pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json"),
        "arms": ["pretrained_initial", "random888_initial"],
        "scope": "Independent saved-only geometry and binary64 clock claims; no model/C0/request",
        "scientific_family3_started": False,
    }
    path = BASE / "INDEPENDENT_BOUNDARY_REGISTRATION.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(sha(path))


if __name__ == "__main__":
    main()
