"""Freeze independent saved-only audit inputs before its execution."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tree = root / "research/review_3080_20261010"
pick = tree / "goal_round_20261014_r15_component"
paths = [
    root / "tools/audit_dual_distance_hash_r15.py",
    pick / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json",
    pick / "PICKUP_DISTANCE_REPAIR_RESULT.json",
    tree / "delivery/source_manifest_timing_root_v1/REGISTRATION.json",
    tree / "delivery/source_manifest_timing_root_v1/RESULT.json",
    tree / "delivery/source_manifest_timing_root_v1/COLLECT_RECEIPT.json",
    tree / "delivery/source_manifest_timing_root_v1/RUN_RESPONSE.json",
    tree / "goal_round_20261014_r14_boundary/DELIVERY_POST_SELECTION_TIMING.json",
]
value = {
    "inputs_sha256": {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "scope": "Independent saved-only binding and arithmetic checks; no model/C0/old request or source-close rerun",
    "expected_decision": "A nonzero failure count is retained as FAIL; no scientific gate upgrade",
}
with (pick / "INDEPENDENT_R15_REGISTRATION.json").open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
