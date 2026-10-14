"""Freeze one 3080 four-shard source-hash pressure condition before execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/source_hash_sharded_root_v1"
OLD = ROOT / "research/review_3080_20261010/delivery/source_overlap_dualcore_root_v1/REGISTRATION.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior = json.loads(OLD.read_text(encoding="utf-8"))
assert prior["expected_file_count"] == 8903
reg = {
    "script_sha256": sha(BASE / "measure_source_shards_v1.py"),
    "transport_sha256": sha(ROOT / "tools/transport_delivery_source_shards_r18.py"),
    "source_start_path": prior["source_start_path"],
    "source_start_sha256": prior["source_start_sha256"],
    "expected_file_count": 8903,
    "hash_cpus": [10, 11, 12, 13], "synthetic_cpu": 15, "parent_cpu": 15,
    "synthetic_cpu_seconds": 4.9, "active_deadline_seconds": 8.0, "reap_deadline_seconds": 2.0,
    "ram_gate_gib": 8, "max_preflight_cpu_percent_each": 20,
    "previous_dualcore_result_sha256": sha(BASE.parent / "source_overlap_dualcore_root_v1/RESULT.json"),
    "scope": "One full old 8903-file source byte hash split into four deterministic sorted-index shards, beside fixed synthetic CPU load; all shards preflight separate idle allowed cores, all owned children reaped. Not actual model request or dynamic loaded closure/parameter publication certificate.",
    "scientific_family3_started": False,
}
with (BASE / "REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(reg, f, ensure_ascii=False, indent=2)
    f.write("\n")
