"""Register the fixed TRAIN-only mechanism readout before generation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"
E = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if not (OUT / "REGISTRATION.json").exists():
        raise RuntimeError("delivery engineering registration missing")
    src = [E / "scripts/evaluation/run_exp_reserve.py",
           ROOT / "research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/reserving_admission.py",
           ROOT / "research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/REGISTRATION.json"]
    reg = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha(ROOT / "tools/diagnose_pickup_train_price_r20.py"),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in src},
        "fixed_train_seed": 20260925,
        "fixed_train_days": 200,
        "fixed_budget": 710.6548328611384,
        "folds": "even/odd day indices, opposite-half calibration",
        "readout": ["number of days with price rejection", "price and hard budget rejection counts",
                    "class counts before and after admission"],
        "old_gate_days_used": False,
        "final_seeds_used": False,
        "online_effect_test": False,
        "scientific_family_started": False,
    }
    with (OUT / "PICKUP_REGISTRATION.json").open("x", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"registered": True, "runner_sha256": reg["runner_sha256"]}))


if __name__ == "__main__":
    main()
