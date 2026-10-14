"""Authoritative terminal evaluator for pickup-to-depot cold-chain traces.

The evaluator consumes cold-chain states already emitted by the event engine;
it never simulates a second thermal trajectory.  This keeps model labels,
online evaluation and replay accounting on one transition implementation.
"""

from __future__ import annotations

from collections import Counter
import os
import sys

import numpy as np


_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLDCHAIN_ROOT = os.path.join(_SCRIPTS_ROOT, 'coldchain')
if _COLDCHAIN_ROOT not in sys.path:
    sys.path.insert(0, _COLDCHAIN_ROOT)

from coldchain_contract import compute_coldchain_cost, default_pilot_contract
from coldchain_state import (
    create_vehicle_state,
    dispatch_vehicle,
    total_quality_loss,
    transition_segment,
)


STATIC_REPLAY_SCHEMA_VERSION = "c0-static-pickup-route-replay-v1"


def _route_trips(route) -> tuple[tuple[int, ...], ...]:
    """Split a zero-delimited route into non-empty single-trip vehicle tours."""

    trips = []
    current = []
    for raw_node in np.asarray(route).reshape(-1):
        node = int(raw_node)
        if node == 0:
            if current:
                trips.append(tuple(current))
                current = []
            continue
        if node < 0:
            raise ValueError("route node ids must be non-negative")
        current.append(node)
    if current:
        trips.append(tuple(current))
    return tuple(trips)


def replay_static_pickup_route(
    route,
    dataset_instance,
    coldchain_contract=None,
    *,
    speed: float = 1.0,
) -> dict:
    """Replay a static route with the same C0 pickup-to-depot transition.

    This is authoritative for cold-chain *physics and accounting*, but it is
    deliberately not a strict-online protocol evaluator: a static route has no
    reveal-event decisions or frozen-prefix history.  Every zero-delimited tour
    dispatches an empty vehicle at time zero, picks cargo after service finishes,
    returns it to the depot, and unloads exactly once.
    """

    contract = coldchain_contract or default_pilot_contract()
    contract.validate()
    if speed <= 0:
        raise ValueError("speed must be positive")

    coords = np.asarray(dataset_instance["coords"], dtype=np.float64)
    demands = np.asarray(dataset_instance["demands"], dtype=np.float64)
    tw_start = np.asarray(dataset_instance["tw_start"], dtype=np.float64)
    tw_end = np.asarray(dataset_instance["tw_end"], dtype=np.float64)
    service_time = np.asarray(
        dataset_instance.get("service_time", np.zeros_like(demands)),
        dtype=np.float64,
    )
    temp_class = np.asarray(
        dataset_instance.get("temp_class", np.zeros_like(demands)),
        dtype=np.int64,
    )
    initial_quality = np.asarray(
        dataset_instance.get("initial_quality", np.ones_like(demands)),
        dtype=np.float64,
    )
    dist_mat = dataset_instance.get("dist_mat")
    if dist_mat is not None:
        dist_mat = np.asarray(dist_mat, dtype=np.float64)

    node_count = len(demands)
    arrays = (coords, tw_start, tw_end, service_time, temp_class, initial_quality)
    if any(len(values) != node_count for values in arrays):
        raise ValueError("all route-replay arrays must share the node dimension")

    def distance(a: int, b: int) -> float:
        if dist_mat is not None:
            return float(dist_mat[a, b])
        return float(np.linalg.norm(coords[a] - coords[b]))

    trips = _route_trips(route)
    states = []
    served = Counter()
    tw_late_count = 0
    depot_return_late_count = 0
    capacity_feasible = True
    active_zones = tuple(True for _ in contract.thermal.supported_temp_classes)

    for trip in trips:
        state = dispatch_vehicle(create_vehicle_state(contract), contract)
        clock = 0.0
        previous = 0
        for node in trip:
            if node <= 0 or node >= node_count:
                raise ValueError("route node id outside dataset: %s" % node)
            if demands[node] <= 0:
                raise ValueError("route contains a non-customer node: %s" % node)
            travel = distance(previous, node) / speed
            arrival = clock + travel
            start = max(arrival, float(tw_start[node]))
            finish = start + float(service_time[node])
            if arrival > float(tw_end[node]) + 1e-6:
                tw_late_count += 1
            try:
                state, _ = transition_segment(
                    state,
                    depart_time=clock,
                    arrival_time=arrival,
                    service_finish=finish,
                    served_customer=node,
                    active_zone_mask=active_zones,
                    contract=contract,
                    order_quantity=float(demands[node]),
                    order_temp_class=int(temp_class[node]),
                    initial_quality=float(initial_quality[node]),
                    segment_distance_units=distance(previous, node),
                )
            except ValueError as exc:
                if "capacity" not in str(exc):
                    raise
                capacity_feasible = False
                raise ValueError(
                    "static pickup route violates shared capacity in trip %s" % (trip,)
                ) from exc
            served[node] += 1
            clock = finish
            previous = node

        return_distance = distance(previous, 0)
        return_arrival = clock + return_distance / speed
        if return_arrival > float(tw_end[0]) + 1e-6:
            depot_return_late_count += 1
        state, _ = transition_segment(
            state,
            depart_time=clock,
            arrival_time=return_arrival,
            service_finish=return_arrival,
            served_customer=None,
            active_zone_mask=active_zones,
            contract=contract,
            return_to_depot=True,
            segment_distance_units=return_distance,
        )
        states.append(state)

    universe = {int(i) for i in range(1, node_count) if demands[i] > 0}
    complete = (
        set(served) == universe
        and all(served[node] == 1 for node in universe)
        and all(state.closed and not state.cargo_manifest for state in states)
    )
    distance_km = sum(state.cumulative_distance_km for state in states)
    energy_kwh = sum(state.cumulative_energy_kwh for state in states)
    quality_loss = sum(total_quality_loss(state, contract) for state in states)
    thermal_count = sum(state.thermal_violation_count for state in states)
    thermal_duration = sum(state.thermal_violation_duration_h for state in states)
    records = [record for state in states for record in state.delivered_to_depot]
    coldchain_cost = compute_coldchain_cost(
        distance_km, quality_loss, energy_kwh, contract.objective)

    return {
        "schema_version": STATIC_REPLAY_SCHEMA_VERSION,
        "operational_scope": "static_pickup_to_depot_route_replay",
        "authoritative_coldchain_physics": True,
        "strict_online_protocol": False,
        "contract_hash": contract.contract_hash,
        "vehicle_count": len(states),
        "complete": bool(complete),
        "capacity_feasible": bool(capacity_feasible),
        "tw_feasible": tw_late_count == 0 and depot_return_late_count == 0,
        "tw_late_count": int(tw_late_count),
        "depot_return_late_count": int(depot_return_late_count),
        "distance_km": float(distance_km),
        "quality_loss": float(quality_loss),
        "energy_kwh": float(energy_kwh),
        "coldchain_cost": float(coldchain_cost),
        "num_unsalable": sum(not record.salable for record in records),
        "thermal_violation_count": int(thermal_count),
        "thermal_violation_duration_h": float(thermal_duration),
        "vehicle_states": tuple(states),
    }


def _base_trace_metrics(traces, instance, capacity, speed=1.0):
    """Small internal fallback used when no distance evaluator result is supplied."""

    coords = np.asarray(instance['coords'])
    tw_end = np.asarray(instance['tw_end'])
    demands = np.asarray(instance['demands'])
    dist_mat = instance.get('dist_mat')

    def dist(i, j):
        if dist_mat is not None:
            return float(dist_mat[i, j])
        return float(np.linalg.norm(coords[i] - coords[j]))

    universe = [i for i in range(1, len(demands)) if demands[i] > 0]
    counts = Counter(sr.node for trace in traces for sr in trace.services)
    duplicate = sum(max(0, counts[node] - 1) for node in counts)
    unserved = sum(counts[node] == 0 for node in universe)
    distance = 0.0
    tw_feasible = True
    return_feasible = True
    capacity_feasible = True
    overload = 0.0
    late = 0
    used = 0
    for trace in traces:
        if not trace.services:
            continue
        used += 1
        load = 0.0
        prev = 0
        for service in trace.services:
            distance += dist(service.prev_node, service.node)
            if service.arrival_time > tw_end[service.node] + 1e-6:
                tw_feasible = False
                late += 1
            load += float(demands[service.node])
            if load > capacity + 1e-6:
                capacity_feasible = False
                overload = max(overload, load - capacity)
            prev = service.node
        distance += dist(prev, 0)
        if trace.return_arrival is None or trace.return_arrival > tw_end[0] + 1e-6:
            return_feasible = False
            tw_feasible = False
    return {
        'complete': unserved == 0 and duplicate == 0,
        'n_visited': sum(counts[node] > 0 for node in universe),
        'n_duplicate': duplicate,
        'n_unserved': unserved,
        'capacity_feasible': capacity_feasible,
        'capacity_overload': float(overload),
        'tw_feasible': tw_feasible,
        'tw_late_count': late,
        'depot_return_feasible': return_feasible,
        'distance_cost': float(distance),
        'quality': 0.0,
        'energy': 0.0,
        'vehicle_count': used,
    }


def evaluate_coldchain_trace(
    traces,
    dataset_instance,
    coldchain_contract,
    objective_contract=None,
    *,
    base_metrics=None,
) -> dict:
    """Evaluate one strict-online execution trace under the frozen C0 contract."""

    coldchain_contract.validate()
    objective = objective_contract or coldchain_contract.objective
    demands = np.asarray(dataset_instance['demands'])
    universe = [int(i) for i in range(1, len(demands)) if demands[i] > 0]
    universe_set = set(universe)
    if base_metrics is None:
        base_metrics = _base_trace_metrics(
            traces,
            dataset_instance,
            coldchain_contract.operational.shared_vehicle_capacity,
            dataset_instance.get('speed', 1.0),
        )
    result = dict(base_metrics)

    picked = Counter()
    delivered = Counter()
    energy_kwh = 0.0
    quality_loss = 0.0
    distance_km = 0.0
    thermal_count = 0
    thermal_duration = 0.0
    num_unsalable = 0
    terminal_manifests_empty = True
    all_enabled_returned = True
    missing_final_state = False
    trace_energy_kwh = 0.0
    trace_quality_loss = 0.0
    trace_distance_km = 0.0
    trace_thermal_count = 0
    trace_thermal_duration = 0.0

    for trace in traces:
        for service in trace.services:
            if service.picked_order_id is not None:
                picked[int(service.picked_order_id)] += 1

        used = bool(trace.services) or trace.final_coldchain_state is not None
        if not used:
            continue
        trace_energy_kwh += float(trace.dispatch_preconditioning_energy_kwh)
        trace_energy_kwh += sum(float(sr.segment_energy_kwh) for sr in trace.services)
        trace_energy_kwh += float(trace.return_segment_energy_kwh)
        trace_quality_loss += sum(float(sr.segment_quality_loss) for sr in trace.services)
        trace_quality_loss += float(trace.return_segment_quality_loss)
        trace_distance_km += sum(float(sr.segment_distance_km) for sr in trace.services)
        trace_distance_km += float(trace.return_segment_distance_km)
        trace_thermal_count += sum(
            int(sr.segment_thermal_violation_count) for sr in trace.services)
        trace_thermal_count += int(trace.return_segment_thermal_violation_count)
        trace_thermal_duration += sum(
            float(sr.segment_thermal_violation_duration_h) for sr in trace.services)
        trace_thermal_duration += float(trace.return_segment_thermal_violation_duration_h)
        final_state = trace.final_coldchain_state
        if final_state is None:
            missing_final_state = True
            terminal_manifests_empty = False
            all_enabled_returned = False
            continue
        energy_kwh += float(final_state.cumulative_energy_kwh)
        distance_km += float(final_state.cumulative_distance_km)
        quality_loss += total_quality_loss(final_state, coldchain_contract)
        thermal_count += int(final_state.thermal_violation_count)
        thermal_duration += float(final_state.thermal_violation_duration_h)
        terminal_manifests_empty &= (
            not final_state.cargo_manifest and final_state.total_load <= 1e-9)
        all_enabled_returned &= (
            bool(final_state.closed) and trace.return_arrival is not None)
        for record in final_state.delivered_to_depot:
            delivered[int(record.order_id)] += 1
            num_unsalable += int(not record.salable)

    all_orders_picked = (
        set(picked) == universe_set
        and all(picked[order_id] == 1 for order_id in universe)
    )
    all_cargo_delivered = (
        set(delivered) == universe_set
        and all(delivered[order_id] == 1 for order_id in universe)
    )
    temperature_hard_feasible = thermal_count == 0 and thermal_duration <= 1e-9
    complete = (
        all_orders_picked
        and all_enabled_returned
        and all_cargo_delivered
        and terminal_manifests_empty
        and not missing_final_state
    )

    expected_distance_km = (
        float(base_metrics['distance_cost'])
        * coldchain_contract.units.distance_km_per_unit)
    distance_accounting_consistent = math_isclose(distance_km, expected_distance_km)
    trace_accounting_consistent = (
        math_isclose(trace_energy_kwh, energy_kwh)
        and math_isclose(trace_quality_loss, quality_loss)
        and math_isclose(trace_distance_km, distance_km)
        and trace_thermal_count == thermal_count
        and math_isclose(trace_thermal_duration, thermal_duration)
    )
    coldchain_cost = compute_coldchain_cost(
        distance_km, quality_loss, energy_kwh, objective)

    result.update({
        'complete': bool(complete),
        'distance_cost': float(base_metrics['distance_cost']),
        'distance_km': float(distance_km),
        'quality_loss': float(quality_loss),
        'energy_kwh': float(energy_kwh),
        'quality': float(quality_loss),
        'energy': float(energy_kwh),
        'num_unsalable': int(num_unsalable),
        'thermal_violation_count': int(thermal_count),
        'thermal_violation_duration_h': float(thermal_duration),
        'coldchain_cost': float(coldchain_cost),
        'all_orders_picked': bool(all_orders_picked),
        'all_cargo_delivered_to_depot': bool(all_cargo_delivered),
        'terminal_manifests_empty': bool(terminal_manifests_empty),
        'all_enabled_vehicles_returned': bool(all_enabled_returned),
        'temperature_hard_feasible': bool(temperature_hard_feasible),
        'distance_accounting_consistent': bool(distance_accounting_consistent),
        'trace_accounting_consistent': bool(trace_accounting_consistent),
        'contract_hash': coldchain_contract.contract_hash,
        'coldchain_schema_version': coldchain_contract.schema_version,
    })
    return result


def math_isclose(a, b, *, atol=1e-7, rtol=1e-7):
    return abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b)))
