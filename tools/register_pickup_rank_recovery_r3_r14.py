"""Preserve the second wire-wrapper mistake; bind the final exact list access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r14_boundary"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = BASE / "PICKUP_RANK_RECOVERY_REGISTRATION.json"
    failure = {"registered_v2_sha256": sha(old), "v2_source_sha256": sha(ROOT / "tools/diagnose_pickup_saved_ranking_r14_v2.py"),
               "exception": "TypeError: list indices must be integers or slices, not str",
               "classification": "wire __array__ unwrapped to list but residual ['values'] access; no output written; no model/C0 executed",
               "prior_failures_preserved": True}
    with (BASE / "PICKUP_RANK_V2_FAILURE.json").open("x", encoding="utf-8") as f:
        json.dump(failure, f, indent=2)
    reg = json.loads(old.read_text(encoding="utf-8"))
    reg["script_sha256"] = sha(ROOT / "tools/diagnose_pickup_saved_ranking_r14_v3.py")
    reg["recovery_from_sha256"] = sha(BASE / "PICKUP_RANK_V2_FAILURE.json")
    reg["scope"] += "; v3 uses the decoded coordinate list"
    with (BASE / "PICKUP_RANK_RECOVERY_R3_REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    print(sha(BASE / "PICKUP_RANK_RECOVERY_R3_REGISTRATION.json"))


if __name__ == "__main__":
    main()
