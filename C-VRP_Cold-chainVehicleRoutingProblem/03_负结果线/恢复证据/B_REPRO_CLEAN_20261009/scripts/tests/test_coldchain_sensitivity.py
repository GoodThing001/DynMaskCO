"""Small preregistered C0 behavior-sensitivity checks for pilot parameters.

These tests verify directional behavior and provenance bounds. They do not
calibrate the pilot parameters and must not be reported as empirical results.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = TEST_DIR.parent
for path in (SCRIPTS_ROOT / "coldchain", SCRIPTS_ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coldchain_contract import default_pilot_contract  # noqa: E402
from coldchain_evaluator import replay_static_pickup_route  # noqa: E402


def instance():
    return {
        "coords": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.5, 0.0], [4.5, 0.0]],
            dtype=np.float32,
        ),
        "demands": np.asarray([0.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "tw_start": np.zeros(4, dtype=np.float32),
        "tw_end": np.full(4, 50.0, dtype=np.float32),
        "service_time": np.asarray([0.0, 0.25, 0.25, 0.25], dtype=np.float32),
        "temp_class": np.asarray([0, 0, 1, 2], dtype=np.int32),
        "initial_quality": np.ones(4, dtype=np.float32),
    }


def replay(contract):
    return replay_static_pickup_route([0, 1, 2, 3, 0], instance(), contract)


def main() -> int:
    base = default_pilot_contract()
    base_result = replay(base)
    checks = []

    def check(name, passed, low, high, meaning):
        checks.append({
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "low": float(low),
            "high": float(high),
            "meaning": meaning,
        })
        print("  [%s] %s: %.8f -> %.8f" % (
            "PASS" if passed else "FAIL", name, low, high))

    low_rate = replace(
        base,
        quality=replace(
            base.quality,
            reference_rate_per_hour=tuple(
                value * 0.5 for value in base.quality.reference_rate_per_hour),
        ),
    )
    high_rate = replace(
        base,
        quality=replace(
            base.quality,
            reference_rate_per_hour=tuple(
                value * 2.0 for value in base.quality.reference_rate_per_hour),
        ),
    )
    low_rate_result, high_rate_result = replay(low_rate), replay(high_rate)
    check(
        "quality_rate_increases_loss",
        high_rate_result["quality_loss"] > low_rate_result["quality_loss"],
        low_rate_result["quality_loss"],
        high_rate_result["quality_loss"],
        "Higher Arrhenius reference rates must increase pickup-to-return loss.",
    )

    low_door = replace(
        base,
        thermal=replace(base.thermal, door_heat_kwh=(0.0, 0.0, 0.0)),
    )
    high_door = replace(
        base,
        thermal=replace(
            base.thermal,
            door_heat_kwh=tuple(value * 2.0 for value in base.thermal.door_heat_kwh),
        ),
    )
    low_door_result, high_door_result = replay(low_door), replay(high_door)
    check(
        "door_heat_increases_recovery_energy",
        high_door_result["energy_kwh"] > low_door_result["energy_kwh"],
        low_door_result["energy_kwh"],
        high_door_result["energy_kwh"],
        "Higher per-opening heat must increase recovery electricity.",
    )

    low_power = replace(
        base,
        thermal=replace(
            base.thermal,
            cooling_power_kw=tuple(
                value * 0.5 for value in base.thermal.cooling_power_kw),
        ),
    )
    high_power = replace(
        base,
        thermal=replace(
            base.thermal,
            cooling_power_kw=tuple(
                value * 1.5 for value in base.thermal.cooling_power_kw),
        ),
    )
    low_power_result, high_power_result = replay(low_power), replay(high_power)
    check(
        "cooling_power_increases_energy_account",
        high_power_result["energy_kwh"] > low_power_result["energy_kwh"],
        low_power_result["energy_kwh"],
        high_power_result["energy_kwh"],
        "Electrical input power must increase kW-times-hour energy accounting.",
    )

    low_transfer = replace(
        base,
        thermal=replace(base.thermal, heat_transfer_per_hour=(0.05, 0.05, 0.05)),
    )
    high_transfer = replace(
        base,
        thermal=replace(base.thermal, heat_transfer_per_hour=(0.40, 0.40, 0.40)),
    )
    low_transfer_result = replay(low_transfer)
    high_transfer_result = replay(high_transfer)
    check(
        "heat_transfer_does_not_improve_quality",
        high_transfer_result["quality_loss"] >= low_transfer_result["quality_loss"],
        low_transfer_result["quality_loss"],
        high_transfer_result["quality_loss"],
        "More ambient heat leakage must not reduce quality loss.",
    )

    no_quality_weight = replace(
        base,
        objective=replace(base.objective, lambda_quality=0.0),
    )
    high_quality_weight = replace(
        base,
        objective=replace(base.objective, lambda_quality=2.0),
    )
    no_q_result, high_q_result = replay(no_quality_weight), replay(high_quality_weight)
    check(
        "quality_weight_increases_scalar_cost",
        high_q_result["coldchain_cost"] > no_q_result["coldchain_cost"],
        no_q_result["coldchain_cost"],
        high_q_result["coldchain_cost"],
        "A positive quality weight must raise scalar cost when loss is positive.",
    )

    no_energy_weight = replace(
        base,
        objective=replace(base.objective, lambda_energy=0.0),
    )
    high_energy_weight = replace(
        base,
        objective=replace(base.objective, lambda_energy=0.5),
    )
    no_e_result, high_e_result = replay(no_energy_weight), replay(high_energy_weight)
    check(
        "energy_weight_increases_scalar_cost",
        high_e_result["coldchain_cost"] > no_e_result["coldchain_cost"],
        no_e_result["coldchain_cost"],
        high_e_result["coldchain_cost"],
        "A positive energy weight must raise scalar cost when energy is positive.",
    )

    variants = (
        low_rate, high_rate, low_door, high_door, low_power, high_power,
        low_transfer, high_transfer, no_quality_weight, high_quality_weight,
        no_energy_weight, high_energy_weight,
    )
    validation_ok = True
    try:
        for variant in variants:
            variant.validate()
    except ValueError:
        validation_ok = False
    check(
        "all_variants_within_preregistered_bounds",
        validation_ok,
        0.0,
        1.0 if validation_ok else 0.0,
        "Every sensitivity variant must pass the frozen provenance bounds.",
    )

    passed = sum(item["status"] == "PASS" for item in checks)
    artifact = {
        "phase": "C0-sensitivity",
        "scope": "pilot_behavior_checks_not_parameter_calibration",
        "contract_hash": base.contract_hash,
        "base_metrics": {
            key: base_result[key]
            for key in ("distance_km", "quality_loss", "energy_kwh", "coldchain_cost")
        },
        "passed": passed,
        "total": len(checks),
        "gate_complete": passed == len(checks),
        "tests": checks,
    }
    output_dir = SCRIPTS_ROOT.parent / "results" / "c0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "coldchain_sensitivity_tests.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResult: %d/%d PASS" % (passed, len(checks)))
    print("Artifact: %s" % output_path)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
