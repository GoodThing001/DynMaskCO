"""Freeze the one-shot delivery CPU15 overlap stress before remote execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/source_overlap_cpu15_root_v1"
script = BASE / "measure_overlap_cpu15_v1.py"
old = ROOT / "research/review_3080_20261010/delivery/source_manifest_timing_root_v1/REGISTRATION.json"
old_reg = json.loads(old.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
value = {
    "script_sha256": sha(script),
    "source_start_sha256": old_reg["source_start_sha256"],
    "source_start_path": old_reg["source_start_path"],
    "expected_file_count": 8903,
    "old_isolated_hash_result_sha256": sha(ROOT / "research/review_3080_20261010/delivery/source_manifest_timing_root_v1/RESULT.json"),
    "old_stage6_timing_sha256": sha(ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary/DELIVERY_POST_SELECTION_TIMING.json"),
    "cpu_affinity": [15], "ram_gate_gib": 8,
    "synthetic_cpu_seconds": 4.9,
    "active_deadline_seconds": 8.0, "reap_deadline_seconds": 2.0,
    "scope": "One actual 3080 CPU15 contention stress: old 8903-file hash pass plus fixed synthetic CPU demand, no model/C0/request, not runtime-equivalent to real stage4-5",
    "scientific_family3_started": False,
}
with (BASE / "REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(value, f, indent=2, allow_nan=False)
    f.write("\n")
