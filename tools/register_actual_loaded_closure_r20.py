"""Freeze component scope before the single round-20 fault-injection run."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    candidate = ROOT / "tools/actual_loaded_closure_candidate_r20.py"
    runner = ROOT / "tools/qualify_actual_loaded_closure_r20.py"
    registration = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "component_fault_injection_only_no_delivery_request",
        "candidate_sha256": sha(candidate),
        "runner_sha256": sha(runner),
        "fixed_checks": ["unchanged_same_process", "changed_explicit_fails",
                         "new_lazy_python_dependency_fails", "missing_explicit_fails", "no_tree_scan"],
        "interpretation": "all checks passing proves only local fail-closed component behavior; 3080 real model loaded closure, native lazy imports, source identity and full 8+2 request remain unqualified",
        "old_split_request_rerun": False,
        "old_failed_audits_preserved": True,
        "science_family3_started": False,
        "stable_online_gain": False,
    }
    with (OUT / "REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(registration, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"registered": True, "candidate_sha256": registration["candidate_sha256"]}))


if __name__ == "__main__":
    main()
