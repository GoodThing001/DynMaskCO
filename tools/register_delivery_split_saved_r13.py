"""Seal inputs for the saved-only split-head replay before it executes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
ARCHIVE = BASE / "downloaded"
OUT = BASE / "SAVED_SPLIT_AUDIT_REGISTRATION.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError("Registration already exists")
    head = json.loads((ARCHIVE / "pretrained_initial_fault0_worker/INITIAL_PARAMETERS.json").read_text(encoding="utf-8"))["head"]
    reg = {
        "scope": "Saved-only fixed-first-layer/sliced-final-layer JAX score replay; no backbone/request/C0 rerun",
        "script_sha256": sha(ROOT / "tools/audit_delivery_split_saved_r13.py"),
        "download_manifest_sha256": sha(BASE / "DOWNLOAD_MANIFEST.json"),
        "original_audit_sha256": sha(ARCHIVE / "INDEPENDENT_TIMED_REQUEST_AUDIT.json"),
        "head_param_sha256": {k: v["sha256"] for k, v in head.items()},
        "arms": ["pretrained_initial", "random888_initial"],
        "expected_records": 50,
        "bitwise_exact_required": True,
        "old_audit_fail_retained": True,
        "full_request_qualified": False,
        "scientific_family3_started": False,
    }
    OUT.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(sha(OUT))


if __name__ == "__main__":
    main()
