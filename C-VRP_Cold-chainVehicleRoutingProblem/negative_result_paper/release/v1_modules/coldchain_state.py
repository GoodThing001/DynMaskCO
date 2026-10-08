"""Pure pickup-to-depot cold-chain vehicle-state transitions.

The input state is never mutated.  A vehicle departs the depot empty, an order
enters the manifest only after pickup service finishes, and all cargo is
finalized and unloaded when the vehicle returns to the depot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from coldchain_contract import (
    ColdChainContract,
    distance_to_km,
    power_duration_to_kwh,
    time_to_hours,
)


_EPS = 1e-9


@dataclass(frozen=True)
class CargoLotState:
    order_id: int
    quantity: float
    temp_class: int
    pickup_finish_time: float
    initial_quality: float
    quality_remaining: float


@dataclass(frozen=True)
class DeliveredCargoRecord:
    order_id: int
    quantity: float
    temp_class: int
    pickup_finish_time: float
    return_arrival_time: float
    initial_quality: float
    delivered_quality: float
    quality_loss: float
    salable: bool


@dataclass(frozen=True)
class VehicleColdChainState:
    compartment_temperature_c: tuple[float, ...]
    zone_load: tuple[float, ...]
    cargo_manifest: tuple[CargoLotState, ...] = ()
    delivered_to_depot: tuple[DeliveredCargoRecord, ...] = ()
    cumulative_energy_kwh: float = 0.0
    cumulative_distance_km: float = 0.0
    finalized_quality_loss: float = 0.0
    thermal_violation_count: int = 0
    thermal_violation_duration_h: float = 0.0
    door_open_count: int = 0
    dispatched: bool = False
    closed: bool = False

    @property
    def total_load(self) -> float:
        return float(sum(self.zone_load))


@dataclass(frozen=True)
class SegmentColdChainMetrics:
    duration_h: float
    distance_km: float
    energy_kwh: float
    quality_loss: float
    door_heat_kwh: float
    thermal_violation_count: int
    thermal_violation_duration_h: float
    picked_order_id: int | None
    unloaded_order_ids: tuple[int, ...]
    load_before: float
    load_after: float
    temperature_before_c: tuple[float, ...]
    temperature_after_c: tuple[float, ...]


def create_vehicle_state(contract: ColdChainContract) -> VehicleColdChainState:
    """Create an undispatched, empty vehicle at ambient temperature."""

    contract.validate()
    zones = len(contract.thermal.supported_temp_classes)
    return VehicleColdChainState(
        compartment_temperature_c=(contract.thermal.ambient_temperature_c,) * zones,
        zone_load=(0.0,) * zones,
    )


def dispatch_vehicle(
    state: VehicleColdChainState,
    contract: ColdChainContract,
) -> VehicleColdChainState:
    """Precondition every compartment and dispatch the vehicle empty exactly once."""

    contract.validate()
    _validate_state(state, contract)
    if state.dispatched or state.closed:
        raise ValueError("vehicle cannot be dispatched more than once")
    if state.cargo_manifest or state.total_load > _EPS:
        raise ValueError("pickup-to-depot vehicle must dispatch empty")
    return replace(
        state,
        compartment_temperature_c=tuple(contract.thermal.target_temperature_c),
        cumulative_energy_kwh=(
            state.cumulative_energy_kwh
            + sum(contract.thermal.dispatch_preconditioning_energy_kwh)
        ),
        dispatched=True,
    )


def compute_cop(temperature_c: float, contract: ColdChainContract) -> float:
    """Temperature-lift COP diagnostic defined by the versioned contract."""

    base, slope, minimum = contract.thermal.cop_parameters
    lift = max(0.0, contract.thermal.ambient_temperature_c - float(temperature_c))
    return max(minimum, base - slope * lift)


def quality_rate_per_hour(
    temp_class: int,
    actual_temperature_c: float,
    contract: ColdChainContract,
) -> float:
    """Arrhenius decay rate evaluated at actual compartment temperature."""

    _validate_temp_class(temp_class, contract)
    actual_k = float(actual_temperature_c) + 273.15
    if actual_k <= 0:
        raise ValueError("actual temperature must be above absolute zero")
    quality = contract.quality
    base_rate = quality.reference_rate_per_hour[temp_class]
    activation_over_r = quality.activation_over_r_kelvin[temp_class]
    exponent = activation_over_r * (1.0 / quality.reference_temperature_k - 1.0 / actual_k)
    return float(base_rate * math.exp(exponent))


def active_quality_loss(
    state: VehicleColdChainState,
    contract: ColdChainContract,
) -> float:
    """Recompute quality loss currently represented by the active manifest."""

    total = 0.0
    for lot in state.cargo_manifest:
        value = contract.quality.product_value[lot.temp_class]
        total += value * lot.quantity * (
            1.0 - lot.quality_remaining / lot.initial_quality)
    return float(total)


def total_quality_loss(
    state: VehicleColdChainState,
    contract: ColdChainContract,
) -> float:
    return float(state.finalized_quality_loss + active_quality_loss(state, contract))


def transition_segment(
    state: VehicleColdChainState,
    depart_time: float,
    arrival_time: float,
    service_finish: float,
    served_customer: int | None,
    active_zone_mask: tuple[bool, ...] | list[bool],
    contract: ColdChainContract,
    *,
    order_quantity: float = 0.0,
    order_temp_class: int | None = None,
    initial_quality: float = 1.0,
    return_to_depot: bool = False,
    segment_distance_units: float = 0.0,
) -> tuple[VehicleColdChainState, SegmentColdChainMetrics]:
    """Advance thermal/quality state, then apply pickup or depot-unload event.

    Existing cargo experiences travel, waiting, service and the service-door
    event.  A newly picked order enters only at ``service_finish`` and therefore
    does not accrue loss on the inbound segment.  On a depot-return segment,
    cargo accrues loss until ``arrival_time`` and is then finalized and unloaded.
    """

    contract.validate()
    _validate_state(state, contract)
    if not state.dispatched:
        raise ValueError("vehicle must be dispatched before it can move")
    if state.closed:
        raise ValueError("closed single-trip vehicle cannot transition or reload")
    if not (depart_time <= arrival_time + _EPS
            and arrival_time <= service_finish + _EPS):
        raise ValueError("times must satisfy depart_time <= arrival_time <= service_finish")
    if return_to_depot and served_customer is not None:
        raise ValueError("a depot-return segment cannot also serve a customer")

    zone_mask = tuple(bool(value) for value in active_zone_mask)
    zones = len(contract.thermal.supported_temp_classes)
    if len(zone_mask) != zones:
        raise ValueError("active_zone_mask length must match temperature zones")

    end_time = float(arrival_time if return_to_depot else service_finish)
    duration_h = time_to_hours(end_time - float(depart_time), contract.units)
    distance_km = distance_to_km(segment_distance_units, contract.units)
    temperature_before = tuple(state.compartment_temperature_c)
    temperature_after, cooling_energy = _advance_temperatures(
        temperature_before, duration_h, zone_mask, contract)

    manifest, segment_quality_loss = _advance_manifest_quality(
        state.cargo_manifest,
        temperature_before,
        temperature_after,
        duration_h,
        contract,
    )

    door_heat = 0.0
    door_recovery_energy = 0.0
    door_count = state.door_open_count
    picked_order_id = None
    if served_customer is not None:
        if int(served_customer) <= 0:
            raise ValueError("served_customer must be a positive order id")
        if order_temp_class is None:
            raise ValueError("order_temp_class is required for a pickup")
        _validate_temp_class(order_temp_class, contract)
        temperature_after, door_heat, door_recovery_energy = _apply_door_event(
            temperature_after, zone_mask, int(order_temp_class), contract)
        door_count += 1
        if order_quantity <= 0:
            raise ValueError("order_quantity must be positive for a pickup")
        if not 0 < initial_quality <= 1:
            raise ValueError("initial_quality must lie in (0, 1]")
        known_ids = {lot.order_id for lot in manifest}
        known_ids.update(record.order_id for record in state.delivered_to_depot)
        if int(served_customer) in known_ids:
            raise ValueError("an order can be picked exactly once")
        projected_load = sum(lot.quantity for lot in manifest) + float(order_quantity)
        if projected_load > contract.operational.shared_vehicle_capacity + _EPS:
            raise ValueError("pickup exceeds shared vehicle capacity")
        manifest = manifest + (CargoLotState(
            order_id=int(served_customer),
            quantity=float(order_quantity),
            temp_class=int(order_temp_class),
            pickup_finish_time=float(service_finish),
            initial_quality=float(initial_quality),
            quality_remaining=float(initial_quality),
        ),)
        picked_order_id = int(served_customer)

    zone_load = _zone_load_from_manifest(manifest, zones)
    delivered = state.delivered_to_depot
    finalized_quality_loss = state.finalized_quality_loss
    unloaded_order_ids: tuple[int, ...] = ()
    closed = False
    if return_to_depot:
        records = tuple(_delivery_record(lot, end_time, contract) for lot in manifest)
        delivered = delivered + records
        finalized_quality_loss += sum(record.quality_loss for record in records)
        unloaded_order_ids = tuple(record.order_id for record in records)
        manifest = ()
        zone_load = (0.0,) * zones
        closed = True

    violation_count, violation_duration = _thermal_violations(
        temperature_before, temperature_after, duration_h, contract)
    segment_energy = cooling_energy + door_recovery_energy
    new_state = VehicleColdChainState(
        compartment_temperature_c=temperature_after,
        zone_load=zone_load,
        cargo_manifest=manifest,
        delivered_to_depot=delivered,
        cumulative_energy_kwh=state.cumulative_energy_kwh + segment_energy,
        cumulative_distance_km=state.cumulative_distance_km + distance_km,
        finalized_quality_loss=finalized_quality_loss,
        thermal_violation_count=state.thermal_violation_count + violation_count,
        thermal_violation_duration_h=(
            state.thermal_violation_duration_h + violation_duration),
        door_open_count=door_count,
        dispatched=True,
        closed=closed,
    )
    _validate_state(new_state, contract)
    metrics = SegmentColdChainMetrics(
        duration_h=duration_h,
        distance_km=distance_km,
        energy_kwh=segment_energy,
        quality_loss=segment_quality_loss,
        door_heat_kwh=door_heat,
        thermal_violation_count=violation_count,
        thermal_violation_duration_h=violation_duration,
        picked_order_id=picked_order_id,
        unloaded_order_ids=unloaded_order_ids,
        load_before=state.total_load,
        load_after=new_state.total_load,
        temperature_before_c=temperature_before,
        temperature_after_c=temperature_after,
    )
    return new_state, metrics


def _advance_temperatures(
    temperatures: tuple[float, ...],
    duration_h: float,
    active_zone_mask: tuple[bool, ...],
    contract: ColdChainContract,
) -> tuple[tuple[float, ...], float]:
    if duration_h <= _EPS:
        return tuple(temperatures), 0.0
    th = contract.thermal
    result = []
    energy = 0.0
    for zone, current in enumerate(temperatures):
        coefficient = th.heat_transfer_per_hour[zone]
        if coefficient > _EPS:
            natural = th.ambient_temperature_c + (
                current - th.ambient_temperature_c) * math.exp(-coefficient * duration_h)
        else:
            natural = current
        if active_zone_mask[zone]:
            target = th.target_temperature_c[zone]
            cooled = natural - th.cooling_rate_c_per_hour[zone] * duration_h
            updated = max(target, cooled)
            energy += power_duration_to_kwh(th.cooling_power_kw[zone], duration_h)
        else:
            updated = natural
        result.append(float(updated))
    return tuple(result), float(energy)


def _advance_manifest_quality(
    manifest: tuple[CargoLotState, ...],
    temperature_before: tuple[float, ...],
    temperature_after: tuple[float, ...],
    duration_h: float,
    contract: ColdChainContract,
) -> tuple[tuple[CargoLotState, ...], float]:
    if duration_h <= _EPS or not manifest:
        return manifest, 0.0
    updated_lots = []
    incremental_loss = 0.0
    for lot in manifest:
        average_temperature = 0.5 * (
            temperature_before[lot.temp_class] + temperature_after[lot.temp_class])
        rate = quality_rate_per_hour(lot.temp_class, average_temperature, contract)
        quality_remaining = lot.quality_remaining * math.exp(-rate * duration_h)
        quality_remaining = min(lot.quality_remaining, max(0.0, quality_remaining))
        value = contract.quality.product_value[lot.temp_class]
        incremental_loss += value * lot.quantity * (
            lot.quality_remaining - quality_remaining) / lot.initial_quality
        updated_lots.append(replace(lot, quality_remaining=quality_remaining))
    return tuple(updated_lots), float(incremental_loss)


def _apply_door_event(
    temperatures: tuple[float, ...],
    active_zone_mask: tuple[bool, ...],
    opened_zone: int,
    contract: ColdChainContract,
) -> tuple[tuple[float, ...], float, float]:
    th = contract.thermal
    result = []
    total_heat = 0.0
    recovery_energy = 0.0
    for zone, current in enumerate(temperatures):
        heat = th.door_heat_kwh[zone] if zone == opened_zone else 0.0
        warmed = current + heat / th.thermal_capacity_kwh_per_c[zone]
        result.append(float(warmed))
        total_heat += heat
        if active_zone_mask[zone]:
            recovery_energy += heat / compute_cop(current, contract)
    return tuple(result), float(total_heat), float(recovery_energy)


def _thermal_violations(
    before: tuple[float, ...],
    after: tuple[float, ...],
    duration_h: float,
    contract: ColdChainContract,
) -> tuple[int, float]:
    count = 0
    duration = 0.0
    for zone, bounds in enumerate(contract.thermal.hard_temperature_bounds_c):
        low, high = bounds
        if (before[zone] < low - _EPS or before[zone] > high + _EPS
                or after[zone] < low - _EPS or after[zone] > high + _EPS):
            count += 1
            duration += duration_h
    return count, float(duration)


def _delivery_record(
    lot: CargoLotState,
    return_arrival_time: float,
    contract: ColdChainContract,
) -> DeliveredCargoRecord:
    value = contract.quality.product_value[lot.temp_class]
    loss = value * lot.quantity * (
        1.0 - lot.quality_remaining / lot.initial_quality)
    threshold = contract.quality.unsalable_quality_threshold[lot.temp_class]
    return DeliveredCargoRecord(
        order_id=lot.order_id,
        quantity=lot.quantity,
        temp_class=lot.temp_class,
        pickup_finish_time=lot.pickup_finish_time,
        return_arrival_time=float(return_arrival_time),
        initial_quality=lot.initial_quality,
        delivered_quality=lot.quality_remaining,
        quality_loss=float(loss),
        salable=lot.quality_remaining / lot.initial_quality >= threshold,
    )


def _zone_load_from_manifest(
    manifest: tuple[CargoLotState, ...],
    zones: int,
) -> tuple[float, ...]:
    loads = [0.0] * zones
    for lot in manifest:
        loads[lot.temp_class] += lot.quantity
    return tuple(float(value) for value in loads)


def _validate_temp_class(temp_class: int, contract: ColdChainContract) -> None:
    if int(temp_class) not in contract.thermal.supported_temp_classes:
        raise ValueError("unsupported temperature class: %s" % temp_class)


def _validate_state(state: VehicleColdChainState, contract: ColdChainContract) -> None:
    zones = len(contract.thermal.supported_temp_classes)
    if len(state.compartment_temperature_c) != zones or len(state.zone_load) != zones:
        raise ValueError("state zone dimensions do not match the contract")
    if any(value < -_EPS for value in state.zone_load):
        raise ValueError("zone loads cannot be negative")
    expected = _zone_load_from_manifest(state.cargo_manifest, zones)
    if any(abs(a - b) > 1e-7 for a, b in zip(state.zone_load, expected)):
        raise ValueError("zone_load must equal cargo-manifest quantities")
    if state.total_load > contract.operational.shared_vehicle_capacity + _EPS:
        raise ValueError("state exceeds shared vehicle capacity")
    if state.closed and (state.cargo_manifest or state.total_load > _EPS):
        raise ValueError("closed vehicle must have an empty manifest")
    if state.closed and not state.dispatched:
        raise ValueError("an undispatched vehicle cannot be closed")
    active_ids = [lot.order_id for lot in state.cargo_manifest]
    delivered_ids = [record.order_id for record in state.delivered_to_depot]
    if len(active_ids + delivered_ids) != len(set(active_ids + delivered_ids)):
        raise ValueError("orders must be unique across manifest and delivered records")
    for lot in state.cargo_manifest:
        _validate_temp_class(lot.temp_class, contract)
        if lot.quantity <= 0 or lot.initial_quality <= 0:
            raise ValueError("cargo quantity and initial quality must be positive")
        if not 0 <= lot.quality_remaining <= lot.initial_quality + _EPS:
            raise ValueError("quality_remaining is outside [0, initial_quality]")
