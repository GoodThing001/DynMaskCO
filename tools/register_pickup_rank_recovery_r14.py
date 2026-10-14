"""Preserve the registered v1 decoder failure and bind the array-only repair."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = BASE / "PICKUP_RANK_REGISTRATION.json"
    failure = {"registered_original_sha256": sha(old), "original_source_sha256": sha(ROOT / "tools/diagnose_pickup_saved_ranking_r14.py"),
               "exception": "KeyError: 0 at packet['dataset']['coords']['values'][0]",
               "classification": "saved wire array wrapper not decoded; no output written; no model/C0 executed",
               "old_failure_preserved": True}
    with (BASE / "PICKUP_RANK_V1_FAILURE.json").open("x", encoding="utf-8") as f:
        json.dump(failure, f, indent=2)
    reg = json.loads(old.read_text(encoding="utf-8"))
    reg["script_sha256"] = sha(ROOT / "tools/diagnose_pickup_saved_ranking_r14_v2.py")
    reg["recovery_from_sha256"] = sha(BASE / "PICKUP_RANK_V1_FAILURE.json")
    reg["scope"] += "; v2 only unwraps the saved __array__ envelope"
    with (BASE / "PICKUP_RANK_RECOVERY_REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(sha(BASE / "PICKUP_RANK_RECOVERY_REGISTRATION.json"))


if __name__ == "__main__":
    main()
