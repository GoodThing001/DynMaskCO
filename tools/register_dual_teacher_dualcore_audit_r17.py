"""Freeze inputs for independent r17 audit before two final C0 recomputations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r17_teacher_dualcore"
DEL = TREE / "delivery/source_overlap_dualcore_root_v1"
paths = [ROOT / "tools/audit_dual_teacher_dualcore_r17.py",
         BASE / "PICKUP_TEACHER_REGISTRATION.json", BASE / "PICKUP_TEACHER_RESULT.json",
         DEL / "REGISTRATION.json", DEL / "RESOURCE_PREFLIGHT.json", DEL / "RUN_RESPONSE.json", DEL / "RESULT.json",
         DEL / "RESULT.json.receipt.json", DEL / "CPU_CHILD_RESULT.json", DEL / "CPU_CHILD_RESULT.json.receipt.json",
         DEL / "HASH_CHILD_RESULT.json.missing.json"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
obj = {"inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths},
       "scope": "Independent saved teacher argmax/identity, two final actual C0 certifications, remote receipts; no model/training/hash/request rerun"}
with (BASE / "INDEPENDENT_R17_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")
