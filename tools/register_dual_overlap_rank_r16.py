"""Data-before-execution fixed engineering registration for both r16 arms."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r16_overlap_rank"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE.mkdir(exist_ok=False)
pickup = {
    "script_sha256": sha(ROOT / "tools/diagnose_pickup_saved_c0_rank_r16.py"),
    "model_sha256": sha(TREE / "pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json"),
    "prior_rank_sha256": sha(TREE / "goal_round_20261014_r14_boundary/PICKUP_RANK_DIAGNOSIS.json"),
    "phase_counts": [6, 4],
    "rule": "Within each frozen phase compare ascending saved proposal score with descending saved actual C0 utility; report every action and all comparable pairs without selecting data",
    "scope": "one old public state, saved-only mechanism, no C0/model rerun, no fresh scientific family or gain",
}
with (BASE / "PICKUP_C0_RANK_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(pickup, f, ensure_ascii=False, indent=2)
    f.write("\n")
