"""Freeze r16 independent saved-only evidence inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r16_overlap_rank"
DEL = TREE / "delivery/source_overlap_cpu15_root_v1"
paths = [ROOT / "tools/audit_dual_overlap_rank_r16.py",
         BASE / "PICKUP_C0_RANK_REGISTRATION.json", BASE / "PICKUP_C0_RANK_RESULT.json",
         DEL / "REGISTRATION.json", DEL / "RESULT.json", DEL / "RUN_RESPONSE.json",
         DEL / "RESULT.json.receipt.json", DEL / "CPU_CHILD_RESULT.json",
         DEL / "CPU_CHILD_RESULT.json.receipt.json", DEL / "HASH_CHILD_RESULT.json.missing.json"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
obj = {"inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths},
       "scope": "Independent saved-only action rank and remote child/transport receipt checks; no model/C0/hash/request rerun"}
with (BASE / "INDEPENDENT_R16_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")
