"""Freeze two naturally consecutive old public inputs and teacher rule."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r17_teacher_dualcore"
BASE.mkdir(exist_ok=False)
inputs = [
    "research/review_3080_20261010/pickup/strong_independent_owner_root_v3/RESPONSE_ATTEMPT_0.json",
    "research/review_3080_20261010/pickup/strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json",
]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg = {
    "script_sha256": sha(ROOT / "tools/build_pickup_two_state_c0_teacher_r17.py"),
    "input_paths": inputs,
    "input_sha256": {name: sha(ROOT / name) for name in inputs},
    "masked_orders": [25, 30],
    "rule": "Preserve unmasked original edges, original vehicles and committed next; enumerate all insertable orders/vehicles/positions. Temporarily place the other masked order on its old owner, certify actual C0, choose highest C0 utility; lexicographic action tie. Repeat once. Record all candidates and final C0.",
    "scope": "Two consecutive observed public engineering states order69/77 from one old day; not an independent train/validation split, no model training or new science family",
    "cpu": [7], "final_seeds_used": False,
}
with (BASE / "PICKUP_TEACHER_REGISTRATION.json").open("x", encoding="utf-8") as f:
    json.dump(reg, f, ensure_ascii=False, indent=2)
    f.write("\n")
