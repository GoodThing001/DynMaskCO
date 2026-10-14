"""Seal one old source-start set for 3080 hash-time measurement."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/source_manifest_timing_root_v1"
OLD = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
REMOTE = r"C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\split_final_timed_execution_root_v1\pretrained_initial_fault0_worker\SOURCE_START.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = OLD / "downloaded/pretrained_initial_fault0_worker/SOURCE_START.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 8903
    reg = {"script_sha256": sha(BASE / "measure_manifest_hash_time_v1.py"),
           "source_start_sha256": sha(source), "source_start_path": REMOTE,
           "expected_file_count": len(manifest["files"]),
           "old_execution_registration_sha256": sha(OLD / "EXECUTION_REGISTRATION.json"),
           "old_original_audit_sha256": sha(OLD / "downloaded/INDEPENDENT_TIMED_REQUEST_AUDIT.json"),
           "scope": "One actual 3080 hash pass of old saved start file set; lower-bound component, not full post-selection gate",
           "cpu": 15, "ram_gate_gib": 8, "no_GPU": True,
           "no_model_C0_request_or_old_QA_rerun": True, "scientific_family3_started": False}
    with (BASE / "REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(sha(BASE / "REGISTRATION.json"))


if __name__ == "__main__":
    main()
