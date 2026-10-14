"""Pre-register all as-yet-unexamined pairs among four interior route orders."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r18_pair_shard"
BASE.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = [
    "research/review_3080_20261010/pickup/strong_independent_owner_root_v3/RESPONSE_ATTEMPT_0.json",
    "research/review_3080_20261010/pickup/strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json",
]
interior = [40, 25, 32, 30]
pairs = [[interior[i], interior[j]] for i in range(4) for j in range(i + 1, 4)]
pairs = [p for p in pairs if set(p) != {25, 30}]
assert len(pairs) == 5
reg = {
    "script_sha256": sha(ROOT / "tools/build_pickup_interior_pair_c0_sweep_r18.py"),
    "input_paths": inputs, "input_sha256": {p: sha(ROOT / p) for p in inputs},
    "expected_interior": interior, "masks": pairs,
    "rule": "All two-order pairs among positions 1..4 of route0 on both old consecutive states, except previously computed [25,30]; enumerate insertable actions, choose each step's highest actual C0 utility with lexicographic tie, certify final, report every delta without choosing a winning mask.",
    "scope": "Old correlated observed states only; engineering availability diagnosis, no new day/seed/model training or scientific family. Disallow cherry-picking a mask for future science from this outcome.",
    "cpu": [7], "final_seeds_used": False,
}
with (BASE / "PICKUP_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(reg, f, ensure_ascii=False, indent=2)
    f.write("\n")
