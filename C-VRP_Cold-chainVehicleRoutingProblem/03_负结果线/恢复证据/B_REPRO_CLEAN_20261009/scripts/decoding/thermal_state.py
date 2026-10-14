"""Retired Phase-3d thermal API bridged to the authoritative C0 model.

The former implementation in this module maintained an independent shared-
cabin thermal state, used delivery-style quality clocks, and contained a
Joule-to-kWh conversion error. It is intentionally no longer executable.
New code must use ``coldchain_state.transition_segment`` for event transitions
or ``coldchain_evaluator.replay_static_pickup_route`` for a static route.

``compute_route_thermal_metrics`` remains as a compatibility wrapper because
external notebooks may import it. The wrapper runs the unified pickup-to-depot
state transition and marks its static (non-online) protocol scope.
"""

from __future__ import annotations

import os
import sys

import numpy as np


_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (
    os.path.join(_SCRIPTS_ROOT, "coldchain"),
    os.path.join(_SCRIPTS_ROOT, "evaluation"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from coldchain_contract import default_pilot_contract  # noqa: E402
from coldchain_evaluator import replay_static_pickup_route  # noqa: E402
from coldchain_state import (  # noqa: E402
    SegmentColdChainMetrics,
    VehicleColdChainState,
    compute_cop as _contract_compute_cop,
)


class LegacyPhysicsRetiredError(RuntimeError):
    """Raised when code tries to execute the removed competing physics model."""


# Import compatibility only. Construction and transitions require the C0 API.
ThermalState = VehicleColdChainState


def compute_cop(delta_t: float, contract=None) -> float:
    """Compatibility COP diagnostic backed by the versioned C0 contract."""

    frozen_contract = contract or default_pilot_contract()
    cabin_temperature = frozen_contract.thermal.ambient_temperature_c - float(delta_t)
    return _contract_compute_cop(cabin_temperature, frozen_contract)


def compute_segment_thermal(*args, **kwargs):
    """Reject the ambiguous legacy segment API.

    Its arguments contain no order-level cargo manifest or pickup/return event,
    so mapping it silently would recreate the semantic bug C0 eliminates.
    """

    raise LegacyPhysicsRetiredError(
        "legacy compute_segment_thermal was retired by C0; use "
        "coldchain_state.transition_segment with a VehicleColdChainState"
    )


def compute_route_thermal_metrics(
    route,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service_time: np.ndarray,
    demands: np.ndarray,
    temp_class: np.ndarray,
    capacity: float = 50.0,
    speed: float = 1.0,
    ambient_temp: float = 25.0,
    quality_salable_threshold: float = 0.1,
    *,
    initial_quality: np.ndarray | None = None,
    dist_mat: np.ndarray | None = None,
    coldchain_contract=None,
):
    """Replay a static route through the unified pickup-to-depot C0 state.

    ``quality_salable_threshold`` is accepted only for source compatibility;
    salability is governed by the per-class thresholds frozen in the contract.
    Ambient temperature must match the selected contract so its hash continues
    to identify the executed physics exactly.
    """

    del quality_salable_threshold
    contract = coldchain_contract or default_pilot_contract()
    if not np.isclose(ambient_temp, contract.thermal.ambient_temperature_c):
        raise ValueError(
            "ambient_temp must be supplied through a versioned ColdChainContract"
        )
    if not np.isclose(capacity, contract.operational.shared_vehicle_capacity):
        raise ValueError(
            "capacity must be supplied through a versioned ColdChainContract"
        )
    instance = {
        "coords": np.asarray(coords),
        "tw_start": np.asarray(tw_start),
        "tw_end": np.asarray(tw_end),
        "service_time": np.asarray(service_time),
        "demands": np.asarray(demands),
        "temp_class": np.asarray(temp_class),
        "initial_quality": (
            np.asarray(initial_quality)
            if initial_quality is not None
            else np.ones_like(np.asarray(demands), dtype=np.float32)
        ),
    }
    if dist_mat is not None:
        instance["dist_mat"] = np.asarray(dist_mat)
    result = replay_static_pickup_route(route, instance, contract, speed=speed)
    # Preserve common historical keys while making their source unambiguous.
    return {
        **result,
        "total_distance": result["distance_km"],
        "total_energy_kwh": result["energy_kwh"],
        "thermal_violations": result["thermal_violation_count"],
        "num_door_opens": sum(
            state.door_open_count for state in result["vehicle_states"]),
    }


__all__ = [
    "LegacyPhysicsRetiredError",
    "SegmentColdChainMetrics",
    "ThermalState",
    "VehicleColdChainState",
    "compute_cop",
    "compute_route_thermal_metrics",
    "compute_segment_thermal",
]
