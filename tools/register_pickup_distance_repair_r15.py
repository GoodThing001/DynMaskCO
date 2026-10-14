"""Freeze the geometry-only comparator before its first actual C0 call."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r15_component"
PICKUP = ROOT / "research/review_3080_20261010/pickup"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    BASE.mkdir(exist_ok=False)
    sources = [
        "research/review_3080_20261010/pickup/public_fleet.py",
        "C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/coldchain_evaluator_a1.py",
        "C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py",
        "tools/qualify_pickup_joint_policy_transaction_root_v3.py",
    ]
    reg = {
        "script_sha256": sha(ROOT / "tools/run_pickup_distance_repair_r15.py"),
        "owner_sha256": sha(PICKUP / "strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json"),
        "model_result_sha256": sha(PICKUP / "learned_after_strong_recovery_execution_root_v2/RESULT.json"),
        "source_files": sources,
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "masked_orders": [25, 30], "fixed_public_order": 77,
        "fixed_rule": "at each step choose C0-certified insertion with least normalized Euclidean distance; preserve unmasked original edges; lexicographic tie-break",
        "arms": ["pretrained", "random", "pure_distance"],
        "scope": "One old public engineering state, actual C0 certificate; no new scientific day, no live installation",
        "scientific_family3_started": False, "equal_time_formal_control": False,
    }
    with (BASE / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(sha(BASE / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json"))


if __name__ == "__main__":
    main()
