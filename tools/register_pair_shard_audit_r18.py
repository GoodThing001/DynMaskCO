"""Register independent final-C0 and read-only delivery receipt audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r18_pair_shard"
DEL = ROOT / "research/review_3080_20261010/delivery/source_hash_sharded_root_v1"
files = [BASE / "PICKUP_REGISTRATION.json", BASE / "PICKUP_RESULT.json",
         DEL / "REGISTRATION.json", DEL / "RESOURCE_PREFLIGHT.json", DEL / "RUN_RESPONSE.json",
         DEL / "UPLOAD_ACCEPTANCE.json", DEL / "RESULT.json", DEL / "RESULT.json.receipt.json",
         DEL / "CPU_CHILD_RESULT.json", DEL / "CPU_CHILD_RESULT.json.receipt.json",
         DEL / "SHARD_0.json", DEL / "SHARD_0.json.receipt.json",
         DEL / "SHARD_1.json.missing.json", DEL / "SHARD_2.json.missing.json",
         DEL / "SHARD_3.json", DEL / "SHARD_3.json.receipt.json"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = {"audit_script_sha256": sha(ROOT / "tools/audit_pair_shard_r18.py"),
       "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files},
       "scope": "Read-only remote receipts and 10 independent pickup final C0 recomputations; no remote replay, no old QA or full-day repeat"}
with (BASE / "INDEPENDENT_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(reg, f, ensure_ascii=False, indent=2)
    f.write("\n")
