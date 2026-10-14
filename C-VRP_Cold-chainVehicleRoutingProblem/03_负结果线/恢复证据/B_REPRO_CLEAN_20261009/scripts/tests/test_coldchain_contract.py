"""C0 contract, transition, trace, compatibility and evaluator gate checks."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys


TEST_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = TEST_DIR.parent
COLDCHAIN_ROOT = SCRIPTS_ROOT / "coldchain"
SIMULATION_ROOT = SCRIPTS_ROOT / "simulation"
EVALUATION_ROOT = SCRIPTS_ROOT / "evaluation"
for path in (SCRIPTS_ROOT, COLDCHAIN_ROOT, SIMULATION_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coldchain_contract import (  # noqa: E402
    compute_coldchain_cost,
    default_pilot_contract,
    joules_to_kwh,
    power_duration_to_kwh,
)
from coldchain_observation import build_visible_coldchain_observation  # noqa: E402
from coldchain_state import (  # noqa: E402
    active_quality_loss,
    create_vehicle_state,
    dispatch_vehicle,
    total_quality_loss,
    transition_segment,
)
from authoritative_evaluator import evaluate_execution_trace  # noqa: E402
from coldchain_evaluator import replay_static_pickup_route  # noqa: E402
from action_contract import certify_route  # noqa: E402
from counterfactual_teacher import enumerate_counterfactuals  # noqa: E402
from recourse_snapshot import (CC_SNAPSHOT_SCHEMA_VERSION, capture_recourse_snapshot,  # noqa: E402
                               snapshot_state_hash)
from strict_online_env import GreedyReplanner, StrictOnlineEnv  # noqa: E402


RESULTS: list[dict[str, object]] = []


def record(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def fresh_dispatched():
    contract = default_pilot_contract()
    return contract, dispatch_vehicle(create_vehicle_state(contract), contract)


def small_dynamic_dataset():
    import numpy as np
    return {
        "coords": np.asarray([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
                             dtype=np.float32),
        "demands": np.asarray([[0.0, 2.0, 3.0, 1.0]], dtype=np.float32),
        "tw_start": np.zeros((1, 4), dtype=np.float32),
        "tw_end": np.full((1, 4), 30.0, dtype=np.float32),
        "service_time": np.asarray([[0.0, 0.5, 0.5, 0.5]], dtype=np.float32),
        "reveal_time": np.asarray([[0.0, 0.0, 0.5, 2.0]], dtype=np.float32),
        "temp_class": np.asarray([[0, 1, 2, 0]], dtype=np.int32),
        "initial_quality": np.ones((1, 4), dtype=np.float32),
    }


def evaluate_env_trace(env, traces, contract):
    instance = {key: value[0] for key, value in env.dataset.items()}
    return evaluate_execution_trace(
        traces,
        instance["coords"],
        instance["tw_start"],
        instance["tw_end"],
        instance["service_time"],
        instance["demands"],
        env.capacity,
        speed=env.tw_speed,
        dist_mat=env.dist_mat[0],
        coldchain_contract=contract,
        dataset_instance=instance,
    )


def pickup(state, contract, order_id, quantity, temp_class, at_time):
    return transition_segment(
        state,
        depart_time=at_time,
        arrival_time=at_time,
        service_finish=at_time,
        served_customer=order_id,
        active_zone_mask=(True, True, True),
        contract=contract,
        order_quantity=quantity,
        order_temp_class=temp_class,
    )


def test_dispatch_empty_and_preconditioned() -> None:
    contract = default_pilot_contract()
    initial = create_vehicle_state(contract)
    dispatched = dispatch_vehicle(initial, contract)
    expected_energy = sum(contract.thermal.dispatch_preconditioning_energy_kwh)
    ok = (
        initial.total_load == 0
        and not initial.dispatched
        and dispatched.dispatched
        and dispatched.total_load == 0
        and dispatched.cargo_manifest == ()
        and dispatched.compartment_temperature_c == contract.thermal.target_temperature_c
        and math.isclose(dispatched.cumulative_energy_kwh, expected_energy)
        and initial.compartment_temperature_c
        == (contract.thermal.ambient_temperature_c,) * 3
    )
    record("dispatch_empty_and_preconditioned", ok, {
        "load": dispatched.total_load,
        "preconditioning_kwh": dispatched.cumulative_energy_kwh,
    })


def test_pickup_enters_at_service_finish() -> None:
    contract, state = fresh_dispatched()
    after, metrics = transition_segment(
        state, 0.0, 1.0, 2.0, 7, (True, True, True), contract,
        order_quantity=4.0, order_temp_class=1, segment_distance_units=1.0)
    lot = after.cargo_manifest[0]
    ok = (
        len(after.cargo_manifest) == 1
        and lot.pickup_finish_time == 2.0
        and lot.quality_remaining == lot.initial_quality
        and after.zone_load == (0.0, 4.0, 0.0)
        and after.total_load == 4.0
        and metrics.quality_loss == 0.0
        and metrics.picked_order_id == 7
        and state.cargo_manifest == ()
    )
    record("pickup_enters_at_service_finish", ok, {
        "pickup_finish": lot.pickup_finish_time,
        "zone_load": after.zone_load,
        "inbound_quality_loss": metrics.quality_loss,
    })


def test_quality_clock_to_return_arrival() -> None:
    contract, state = fresh_dispatched()
    picked, _ = pickup(state, contract, 1, 2.0, 2, 3.0)
    returned, metrics = transition_segment(
        picked, 3.0, 5.0, 5.0, None, (False, False, False), contract,
        return_to_depot=True, segment_distance_units=2.0)
    record_ = returned.delivered_to_depot[0]
    ok = (
        math.isclose(metrics.duration_h, 2.0 * contract.units.hours_per_time_unit)
        and record_.pickup_finish_time == 3.0
        and record_.return_arrival_time == 5.0
        and record_.delivered_quality < record_.initial_quality
        and record_.quality_loss > 0
    )
    record("quality_clock_service_finish_to_return", ok, {
        "exposure_time_units": record_.return_arrival_time - record_.pickup_finish_time,
        "quality_loss": record_.quality_loss,
    })


def test_return_unloads_closes_and_forbids_reload() -> None:
    contract, state = fresh_dispatched()
    picked, _ = pickup(state, contract, 1, 1.0, 0, 0.0)
    returned, _ = transition_segment(
        picked, 0.0, 1.0, 1.0, None, (True, True, True), contract,
        return_to_depot=True, segment_distance_units=1.0)
    rejected = False
    try:
        pickup(returned, contract, 2, 1.0, 0, 1.0)
    except ValueError:
        rejected = True
    ok = (
        returned.closed
        and returned.cargo_manifest == ()
        and returned.zone_load == (0.0, 0.0, 0.0)
        and len(returned.delivered_to_depot) == 1
        and rejected
    )
    record("return_unloads_closes_forbids_reload", ok, {
        "delivered": len(returned.delivered_to_depot),
        "reload_rejected": rejected,
    })


def test_zero_duration_no_event_is_identity() -> None:
    contract, state = fresh_dispatched()
    after, metrics = transition_segment(
        state, 2.0, 2.0, 2.0, None, (True, True, True), contract)
    ok = after == state and metrics.duration_h == 0 and metrics.energy_kwh == 0
    record("zero_duration_no_event_identity", ok, {
        "state_equal": after == state,
        "energy_kwh": metrics.energy_kwh,
    })


def test_energy_unit_conversions() -> None:
    kw_h = power_duration_to_kwh(2.0, 3.5)
    joules = joules_to_kwh(3_600_000.0)
    ok = math.isclose(kw_h, 7.0) and math.isclose(joules, 1.0)
    record("energy_unit_conversions", ok, {"kw_times_h": kw_h, "joules_to_kwh": joules})


def test_quality_monotonicity_and_pickup_order() -> None:
    contract, state = fresh_dispatched()
    picked, _ = pickup(state, contract, 9, 1.0, 2, 0.0)
    one_hour, one_metrics = transition_segment(
        picked, 0.0, 1.0, 1.0, None, (False, False, False), contract)
    two_hours, two_metrics = transition_segment(
        picked, 0.0, 2.0, 2.0, None, (False, False, False), contract)
    hot = replace(picked, compartment_temperature_c=(30.0, 20.0, 10.0))
    _, hot_metrics = transition_segment(
        hot, 0.0, 1.0, 1.0, None, (False, False, False), contract)

    # Same two stops and return time: picking the high-perishability order later
    # shortens its policy-controlled exposure from 3 to 2 time units.
    high_first, _ = pickup(state, contract, 2, 1.0, 2, 0.0)
    high_first, _ = transition_segment(
        high_first, 0.0, 1.0, 1.0, 1, (False, False, False), contract,
        order_quantity=1.0, order_temp_class=0)
    high_first, _ = transition_segment(
        high_first, 1.0, 3.0, 3.0, None, (False, False, False), contract,
        return_to_depot=True)
    low_first, _ = pickup(state, contract, 1, 1.0, 0, 0.0)
    low_first, _ = transition_segment(
        low_first, 0.0, 1.0, 1.0, 2, (False, False, False), contract,
        order_quantity=1.0, order_temp_class=2)
    low_first, _ = transition_segment(
        low_first, 1.0, 3.0, 3.0, None, (False, False, False), contract,
        return_to_depot=True)
    high_loss_early = next(r.quality_loss for r in high_first.delivered_to_depot if r.order_id == 2)
    high_loss_late = next(r.quality_loss for r in low_first.delivered_to_depot if r.order_id == 2)
    ok = (
        two_metrics.quality_loss >= one_metrics.quality_loss > 0
        and hot_metrics.quality_loss > one_metrics.quality_loss
        and high_loss_late < high_loss_early
        and two_hours.cargo_manifest[0].quality_remaining
        <= one_hour.cargo_manifest[0].quality_remaining
    )
    record("quality_monotonicity_and_pickup_order", ok, {
        "loss_1h": one_metrics.quality_loss,
        "loss_2h": two_metrics.quality_loss,
        "loss_hot": hot_metrics.quality_loss,
        "high_loss_early": high_loss_early,
        "high_loss_late": high_loss_late,
    })


def test_door_event_adds_heat() -> None:
    contract, state = fresh_dispatched()
    after, metrics = pickup(state, contract, 1, 1.0, 1, 0.0)
    non_cooling = all(a >= b for a, b in zip(
        after.compartment_temperature_c, state.compartment_temperature_c))
    ok = metrics.door_heat_kwh > 0 and metrics.energy_kwh > 0 and non_cooling
    record("door_event_adds_heat", ok, {
        "door_heat_kwh": metrics.door_heat_kwh,
        "recovery_energy_kwh": metrics.energy_kwh,
    })


def test_return_edge_accounting() -> None:
    contract, state = fresh_dispatched()
    picked, _ = pickup(state, contract, 3, 2.0, 1, 0.0)
    returned, metrics = transition_segment(
        picked, 0.0, 2.0, 2.0, None, (True, True, True), contract,
        return_to_depot=True, segment_distance_units=5.0)
    ok = (
        math.isclose(metrics.distance_km, 5.0 * contract.units.distance_km_per_unit)
        and metrics.energy_kwh > 0
        and metrics.quality_loss > 0
        and returned.delivered_to_depot[0].quality_loss > 0
    )
    record("return_edge_distance_energy_quality", ok, {
        "distance_km": metrics.distance_km,
        "energy_kwh": metrics.energy_kwh,
        "quality_loss": metrics.quality_loss,
    })


def test_capacity_and_zone_load_conservation() -> None:
    contract, state = fresh_dispatched()
    state, _ = pickup(state, contract, 1, 10.0, 0, 0.0)
    state, _ = pickup(state, contract, 2, 15.0, 2, 0.0)
    rejected = False
    try:
        pickup(state, contract, 3, 26.0, 1, 0.0)
    except ValueError:
        rejected = True
    manifest_load = sum(lot.quantity for lot in state.cargo_manifest)
    ok = (
        state.zone_load == (10.0, 0.0, 15.0)
        and state.total_load == manifest_load == 25.0
        and rejected
    )
    record("capacity_and_zone_load_conservation", ok, {
        "zone_load": state.zone_load,
        "capacity_rejected": rejected,
    })


def test_objective_hash_and_provenance() -> None:
    contract, state = fresh_dispatched()
    state, _ = pickup(state, contract, 5, 1.0, 1, 0.0)
    active_loss = active_quality_loss(state, contract)
    state, _ = transition_segment(
        state, 0.0, 1.0, 1.0, None, (True, True, True), contract,
        return_to_depot=True, segment_distance_units=2.0)
    recomputed_loss = sum(record_.quality_loss for record_ in state.delivered_to_depot)
    cost = compute_coldchain_cost(
        state.cumulative_distance_km,
        total_quality_loss(state, contract),
        state.cumulative_energy_kwh,
        contract.objective,
    )
    manual = (
        state.cumulative_distance_km / contract.objective.distance_scale
        + contract.objective.lambda_quality
        * recomputed_loss / contract.objective.quality_scale
        + contract.objective.lambda_energy
        * state.cumulative_energy_kwh / contract.objective.energy_scale
    )
    manifest = contract.to_manifest()
    provenance_ok = all(
        item.source_type in {"literature", "calibrated", "pilot"}
        and item.sensitivity_low <= item.sensitivity_high
        and bool(item.source_reference)
        for item in contract.parameter_provenance
    )
    ok = (
        active_loss == 0.0
        and math.isclose(state.finalized_quality_loss, recomputed_loss)
        and math.isclose(cost, manual)
        and manifest["schema_version"] == "coldchain-contract-v1"
        and manifest["contract_hash"] == contract.contract_hash
        and len(contract.contract_hash) == 64
        and provenance_ok
    )
    record("objective_hash_and_provenance", ok, {
        "coldchain_cost": cost,
        "contract_hash": contract.contract_hash,
        "provenance_entries": len(contract.parameter_provenance),
    })


def test_strict_online_trace_evaluator_parity() -> None:
    contract = default_pilot_contract()
    dataset = small_dynamic_dataset()
    env = StrictOnlineEnv(
        dataset, 50.0, 1.0, 2, GreedyReplanner("edd"),
        coldchain_contract=contract)
    traces, served = env.run(0)
    result = evaluate_env_trace(env, traces, contract)
    enabled = [trace for trace in traces if trace.services]
    ok = (
        bool(served.all())
        and result["complete"]
        and result["all_orders_picked"]
        and result["all_cargo_delivered_to_depot"]
        and result["terminal_manifests_empty"]
        and result["distance_accounting_consistent"]
        and result["trace_accounting_consistent"]
        and result["quality_loss"] > 0
        and result["energy_kwh"] > 0
        and all(trace.final_coldchain_state.closed for trace in enabled)
    )
    record("strict_online_trace_evaluator_parity", ok, {
        "distance_cost": result["distance_cost"],
        "quality_loss": result["quality_loss"],
        "energy_kwh": result["energy_kwh"],
        "enabled_vehicles": len(enabled),
    })


def test_snapshot_v3_resume_parity() -> None:
    contract = default_pilot_contract()
    dataset = small_dynamic_dataset()
    snapshots = []
    env = StrictOnlineEnv(
        dataset, 50.0, 1.0, 2, GreedyReplanner("edd"),
        coldchain_contract=contract)
    env.snapshot_hook = lambda *args: snapshots.append(capture_recourse_snapshot(*args))
    original_traces, original_served = env.run(0)
    # Prefer a mid-leg reveal snapshot so thermal state has advanced before service.
    candidates = [snap for snap in snapshots
                  if snap["schema_version"] == CC_SNAPSHOT_SCHEMA_VERSION
                  and any(status == "committed" for status in snap["vehicle_status"])
                  and snap["clock"] > 0]
    snap = candidates[0]
    resumed_env = StrictOnlineEnv(
        dataset, 50.0, 1.0, 2, GreedyReplanner("edd"),
        coldchain_contract=contract)
    resumed_traces, resumed_served = resumed_env.run_resumed(snap)
    original_result = evaluate_env_trace(env, original_traces, contract)
    resumed_result = evaluate_env_trace(resumed_env, resumed_traces, contract)
    comparable_keys = (
        "complete", "distance_cost", "quality_loss", "energy_kwh",
        "coldchain_cost", "thermal_violation_count",
        "all_cargo_delivered_to_depot", "trace_accounting_consistent",
    )
    same_metrics = all(
        original_result[key] == resumed_result[key] for key in comparable_keys)
    same_states = [trace.final_coldchain_state for trace in original_traces] == [
        trace.final_coldchain_state for trace in resumed_traces]
    ok = (
        snap["coldchain_contract_hash"] == contract.contract_hash
        and snapshot_state_hash(snap) == snap["state_hash"]
        and same_metrics
        and same_states
        and (original_served == resumed_served).all()
    )
    record("snapshot_v3_resume_parity", ok, {
        "snapshot_clock": snap["clock"],
        "schema": snap["schema_version"],
        "same_metrics": same_metrics,
        "same_states": same_states,
    })


def test_future_coldchain_visibility_invariance() -> None:
    import numpy as np
    temp_a = np.asarray([0, 1, 2, 0], dtype=np.int32)
    quality_a = np.asarray([1.0, 0.98, 0.95, 0.90], dtype=np.float32)
    reveal = np.asarray([0.0, 0.0, 2.0, 3.0], dtype=np.float32)
    temp_b = temp_a.copy()
    quality_b = quality_a.copy()
    temp_b[2:] = np.asarray([0, 2])
    quality_b[2:] = np.asarray([0.41, 0.37])
    obs_a = build_visible_coldchain_observation(temp_a, quality_a, reveal, 1.0)
    obs_b = build_visible_coldchain_observation(temp_b, quality_b, reveal, 1.0)
    ok = all(np.array_equal(obs_a[key], obs_b[key]) for key in obs_a)
    ok = ok and np.array_equal(obs_a["temp_class"], np.asarray([0, 1, -1, -1]))
    record("future_coldchain_visibility_invariance", ok, {
        "visible_mask": obs_a["visible_mask"].tolist(),
        "masked_temp_class": obs_a["temp_class"].tolist(),
    })


def test_coldchain_action_certificate_rejects() -> None:
    import numpy as np
    contract = default_pilot_contract()
    dataset = small_dynamic_dataset()
    env = StrictOnlineEnv(dataset, 50.0, 1.0, 1, None, coldchain_contract=contract)
    env.temp_class[0, 1] = 9
    bad_temp = certify_route(env, 0, 0, 0.0, 0.0, [1])

    capacity_dataset = small_dynamic_dataset()
    capacity_dataset["demands"][0, 1] = 51.0
    capacity_env = StrictOnlineEnv(
        capacity_dataset, 50.0, 1.0, 1, None, coldchain_contract=contract)
    bad_capacity = certify_route(capacity_env, 0, 0, 0.0, 0.0, [1])
    ok = (
        not bad_temp["feasible"]
        and bad_temp["reason"] == "temperature"
        and not bad_temp["temperature_compatible"]
        and bad_temp["coldchain_reject_reason"] == "unsupported_temperature_class"
        and not bad_capacity["feasible"]
        and bad_capacity["reason"] == "capacity"
        and bad_capacity["coldchain_reject_reason"]
        == "shared_total_capacity_exceeded"
    )
    record("coldchain_action_certificate_rejects", ok, {
        "temperature_reason": bad_temp["coldchain_reject_reason"],
        "capacity_reason": bad_capacity["coldchain_reject_reason"],
    })


def test_coldchain_counterfactual_teacher() -> None:
    contract = default_pilot_contract()
    dataset = small_dynamic_dataset()
    snapshots = []
    continuation = GreedyReplanner("edd")
    source_env = StrictOnlineEnv(
        dataset, 50.0, 1.0, 2, continuation, coldchain_contract=contract)
    source_env.snapshot_hook = lambda *args: snapshots.append(capture_recourse_snapshot(*args))
    source_env.run(0)
    initial = next(snap for snap in snapshots if snap["event_id"] == 0)
    rollout_env = StrictOnlineEnv(
        dataset, 50.0, 1.0, 2, continuation, coldchain_contract=contract)
    rows, baseline = enumerate_counterfactuals(
        rollout_env, initial, continuation, customer=1, objective="coldchain")
    feasible = [row for row in rows if row["feasible"]]
    incumbents = [row for row in feasible if row["incumbent"]]
    required = {
        "hard_outcome", "distance_cost", "quality_loss", "energy_kwh",
        "num_unsalable", "thermal_violation_count", "coldchain_cost",
        "delta_distance", "delta_quality", "delta_energy",
        "delta_coldchain_cost", "contract_hash",
    }
    ok = (
        bool(feasible)
        and len(incumbents) == 1
        and abs(incumbents[0]["delta_coldchain_cost"]) < 1e-9
        and all(required.issubset(row) for row in feasible)
        and all(row["contract_hash"] == contract.contract_hash for row in feasible)
        and baseline["complete"]
        and baseline["trace_accounting_consistent"]
    )
    record("coldchain_counterfactual_teacher", ok, {
        "feasible_actions": len(feasible),
        "incumbent_delta": (incumbents[0]["delta_coldchain_cost"]
                            if incumbents else None),
        "baseline_cost": baseline["coldchain_cost"],
    })


def test_static_route_replay_uses_pickup_clock() -> None:
    contract = default_pilot_contract()
    instance = {key: value[0] for key, value in small_dynamic_dataset().items()}
    # Each route starts empty. Customer 1 is picked at t=1.5, customer 2 at
    # t=3.0; both accrue quality loss only after their own service finishes.
    result = replay_static_pickup_route(
        [0, 1, 2, 0, 3, 0], instance, contract, speed=1.0)
    states = result["vehicle_states"]
    records = {
        record_.order_id: record_
        for state in states
        for record_ in state.delivered_to_depot
    }
    ok = (
        result["authoritative_coldchain_physics"]
        and not result["strict_online_protocol"]
        and result["complete"]
        and result["vehicle_count"] == 2
        and records[1].pickup_finish_time == 1.5
        and records[2].pickup_finish_time == 3.0
        and records[1].quality_loss > records[2].quality_loss
        and all(state.closed and not state.cargo_manifest for state in states)
    )
    record("static_route_replay_pickup_clock", ok, {
        "schema": result["schema_version"],
        "vehicle_count": result["vehicle_count"],
        "quality_loss": result["quality_loss"],
    })


def test_legacy_segment_physics_is_retired() -> None:
    decoding_root = SCRIPTS_ROOT / "decoding"
    if str(decoding_root) not in sys.path:
        sys.path.insert(0, str(decoding_root))
    from thermal_state import (  # noqa: E402
        LegacyPhysicsRetiredError,
        compute_segment_thermal,
    )
    rejected = False
    try:
        compute_segment_thermal(None, 1.0, 0.1, 1, 1.0)
    except LegacyPhysicsRetiredError:
        rejected = True
    record("legacy_segment_physics_retired", rejected, {
        "old_independent_transition_rejected": rejected,
    })


def test_legacy_route_wrapper_matches_unified_replay() -> None:
    decoding_root = SCRIPTS_ROOT / "decoding"
    if str(decoding_root) not in sys.path:
        sys.path.insert(0, str(decoding_root))
    from thermal_state import compute_route_thermal_metrics  # noqa: E402
    contract = default_pilot_contract()
    instance = {key: value[0] for key, value in small_dynamic_dataset().items()}
    route = [0, 1, 2, 0, 3, 0]
    direct = replay_static_pickup_route(route, instance, contract)
    wrapped = compute_route_thermal_metrics(
        route,
        instance["coords"],
        instance["tw_start"],
        instance["tw_end"],
        instance["service_time"],
        instance["demands"],
        instance["temp_class"],
        initial_quality=instance["initial_quality"],
        coldchain_contract=contract,
    )
    comparable = (
        "distance_km", "quality_loss", "energy_kwh", "coldchain_cost",
        "num_unsalable", "thermal_violation_count", "contract_hash",
    )
    ok = all(direct[key] == wrapped[key] for key in comparable)
    record("legacy_route_wrapper_unified_parity", ok, {
        "schema": wrapped["schema_version"],
        "exact_fields": list(comparable),
    })


def test_legacy_proxy_sources_are_explicitly_isolated() -> None:
    files = {
        "data": SCRIPTS_ROOT / "data" / "generate_coldchain_data.py",
        "decoder": SCRIPTS_ROOT / "decoding" / "cvrptw.py",
        "rolling": SCRIPTS_ROOT / "simulation" / "rolling_horizon.py",
        "dynamic_run": SCRIPTS_ROOT / "simulation" / "run_dynamic_sim.py",
        "model": SCRIPTS_ROOT / "models" / "DynamicColdChainModel.py",
    }
    source = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
    checks = {
        "data_aliases": all(token in source["data"] for token in (
            "legacy_delivery_reference_quality", "legacy_edge_energy_proxy",
            "initial_quality")),
        "decoder_non_authoritative": all(token in source["decoder"] for token in (
            "legacy-delivery-proxy-v1", "authoritative_coldchain_physics': False")),
        "rolling_non_authoritative": all(token in source["rolling"] for token in (
            "legacy-rolling-horizon-proxy-v1", "'coldchain_authoritative': False")),
        "dynamic_run_non_authoritative": all(token in source["dynamic_run"] for token in (
            "legacy-rolling-horizon-proxy-v1", "coldchain_authoritative")),
        "normalized_temp_fixed": "raw_features[..., 5] * 2.0 + 0.5" in source["model"],
    }
    record("legacy_proxy_sources_explicitly_isolated", all(checks.values()), checks)


def main() -> int:
    print("C0-0/C0-1 cold-chain contract/state unit checks")
    tests = (
        test_dispatch_empty_and_preconditioned,
        test_pickup_enters_at_service_finish,
        test_quality_clock_to_return_arrival,
        test_return_unloads_closes_and_forbids_reload,
        test_zero_duration_no_event_is_identity,
        test_energy_unit_conversions,
        test_quality_monotonicity_and_pickup_order,
        test_door_event_adds_heat,
        test_return_edge_accounting,
        test_capacity_and_zone_load_conservation,
        test_objective_hash_and_provenance,
        test_strict_online_trace_evaluator_parity,
        test_snapshot_v3_resume_parity,
        test_future_coldchain_visibility_invariance,
        test_coldchain_action_certificate_rejects,
        test_coldchain_counterfactual_teacher,
        test_static_route_replay_uses_pickup_clock,
        test_legacy_segment_physics_is_retired,
        test_legacy_route_wrapper_matches_unified_replay,
        test_legacy_proxy_sources_are_explicitly_isolated,
    )
    for test in tests:
        try:
            test()
        except Exception as exc:  # preserve a machine-readable failure artifact
            record(test.__name__, False, "%s: %s" % (type(exc).__name__, exc))

    passed = sum(item["status"] == "PASS" for item in RESULTS)
    output = {
        "phase": "C0-0/C0-1",
        "scope": "complete_c0_contract_state_trace_and_legacy_migration_gate",
        "passed": passed,
        "total": len(RESULTS),
        "gate_complete": passed == len(RESULTS),
        "remaining_gate_areas": [],
        "tests": RESULTS,
    }
    output_dir = SCRIPTS_ROOT.parent / "results" / "c0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "coldchain_contract_tests.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResult: %d/%d PASS; C0 functional gate %s" % (
        passed, len(RESULTS), "PASS" if passed == len(RESULTS) else "FAIL"))
    print("Artifact: %s" % output_path)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
