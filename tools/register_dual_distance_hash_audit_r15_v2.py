"""Record v1 audit harness failure and freeze narrow v2 recovery."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tree = root / "research/review_3080_20261010"
pick = tree / "goal_round_20261014_r15_component"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
failure = {
    "registration_sha256": sha(pick / "INDEPENDENT_R15_REGISTRATION.json"),
    "source_sha256": sha(root / "tools/audit_dual_distance_hash_r15.py"),
    "exit_code": 1,
    "error": "KeyError: 'known_utility_delta'",
    "cause": "Audit harness addressed a field not present in the old model result; old mechanism record has it",
    "result_written": False,
    "original_experiment_unchanged": True,
}
with (pick / "INDEPENDENT_R15_V1_FAILURE.json").open("x", encoding="utf-8") as stream:
    json.dump(failure, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
paths = [
    root / "tools/audit_dual_distance_hash_r15_v2.py",
    pick / "INDEPENDENT_R15_REGISTRATION.json",
    pick / "INDEPENDENT_R15_V1_FAILURE.json",
    tree / "goal_round_20261013_r13_model_action_resource/PICKUP_SAVED_ACTION_MECHANISM.json",
    pick / "PICKUP_DISTANCE_REPAIR_RESULT.json",
    tree / "delivery/source_manifest_timing_root_v1/RESULT.json",
]
reg = {"inputs_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in paths},
       "recovery_scope": "Only v1 missing-field lookup replaced by existing saved mechanism row; no experiment rerun",
       "v1_failure_preserved": True}
with (pick / "INDEPENDENT_R15_V2_REGISTRATION.json").open("x", encoding="utf-8") as stream:
    json.dump(reg, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
