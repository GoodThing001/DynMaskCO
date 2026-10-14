"""Freeze a single distinct-core scheduling stress; no performance tuning loop."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/source_overlap_dualcore_root_v1"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior = json.loads((ROOT / "research/review_3080_20261010/delivery/source_overlap_cpu15_root_v1/REGISTRATION.json").read_text(encoding="utf-8"))
reg = {
    "script_sha256": sha(BASE / "measure_overlap_dualcore_v1.py"),
    "source_start_path": prior["source_start_path"],
    "source_start_sha256": prior["source_start_sha256"],
    "expected_file_count": 8903,
    "hash_cpu_affinity": [14], "synthetic_cpu_affinity": [15], "parent_cpu_affinity": [15],
    "ram_gate_gib": 8, "max_preflight_cpu_percent_each": 20,
    "synthetic_cpu_seconds": 4.9, "active_deadline_seconds": 8.0, "reap_deadline_seconds": 2.0,
    "prior_same_core_result_sha256": sha(ROOT / "research/review_3080_20261010/delivery/source_overlap_cpu15_root_v1/RESULT.json"),
    "scope": "One distinct-core CPU14 source hash plus CPU15 synthetic 4.9 CPU-second stress, resource gated and all owned children reaped; not actual model request or full source/parameter close",
    "scientific_family3_started": False,
}
with (BASE / "REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(reg, f, indent=2)
    f.write("\n")
