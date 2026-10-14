"""Delivery-only route certification with loaded manifests and customer unloading."""
from dataclasses import dataclass, replace
import math
import numpy as np
from delivery_contract import DeliveryContract
from coldchain_state import (_advance_temperatures, _advance_manifest_quality_v2,
                             _apply_door_event)


@dataclass(frozen=True)
class DeliveryLot:
    order_id: int
    quantity: float
    temp_class: int
    loading_finish_time: float
    initial_quality: float
    quality_remaining: float


def validate_instance(instance, contract):
    contract.validate()
    keys = ('coords', 'demands', 'temp_class', 'reveal_time', 'ready_time',
            'tw_start', 'tw_end', 'service_time')
    if any(k not in instance for k in keys):
        raise ValueError('Missing delivery instance field')
    coords = np.asarray(instance['coords'], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 1:
        raise ValueError('Invalid customer coordinates')
    n = len(coords)
    for k in keys[1:]:
        a = np.asarray(instance[k])
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError('Invalid field shape or nonfinite value: ' + k)
    if not np.all(np.isfinite(coords)):
        raise ValueError('Nonfinite coordinates')
    d = np.asarray(instance['demands'])
    if d[0] != 0 or np.any(d[1:] <= 0):
        raise ValueError('Depot demand or customer quantities invalid')
    c = np.asarray(instance['temp_class'])
    if np.any(c != np.floor(c)) or np.any((c < 0) | (c > 2)):
        raise ValueError('Invalid temperature class')
    if np.any(np.asarray(instance['tw_start']) > instance['tw_end']):
        raise ValueError('Reversed service windows')
    if np.any(np.asarray(instance['ready_time']) < instance['reveal_time']):
        raise ValueError('Goods cannot be ready before request release')
    for k in ('reveal_time', 'ready_time', 'tw_start', 'tw_end', 'service_time'):
        if np.any(np.asarray(instance[k]) < 0):
            raise ValueError('Negative delivery time')


def evaluate(instance, routes, contract=None, wave_h=0.0, deadline=None, clock=None):
    import time
    clock = time.perf_counter if clock is None else clock
    contract = contract or DeliveryContract()
    validate_instance(instance, contract)
    if not math.isfinite(wave_h) or wave_h < 0:
        raise ValueError('Invalid wave time')
    n = len(instance['demands'])
    if any(type(v) not in (int, np.int32, np.int64) or v <= 0 or v >= n for route in routes for v in route):
        raise ValueError('Routes must contain customer IDs only, no depot/PAD/fractional IDs')
    flat = [int(v) for route in routes for v in route]
    errors = []
    if len(flat) != len(set(flat)):
        errors.append('duplicate_ownership')
    if set(flat) != set(range(1, n)):
        errors.append('incomplete_delivery')
    if sum(bool(r) for r in routes) > contract.vehicles:
        errors.append('fleet_limit')
    if errors:
        return {'hard': False, 'complete': False, 'errors': sorted(set(errors)),
            'cost': None, 'distance_km': 0.0, 'energy_kwh': 0.0, 'quality_loss_value': 0.0,
            'delivered': [], 'traces': [], 'contract_hash': contract.identity(),
            'qualification': 'delivery_engineering_snapshot_only', 'online_gain_established': False}
    coords = np.asarray(instance['coords'], float)
    dist = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1))
    p = contract.physics()
    total_d, energy, loss = 0.0, 0.0, 0.0
    traces, delivered = [], []
    for vid, route in enumerate(routes):
        if not route:
            continue
        if deadline is not None and clock() >= deadline:
            errors.append('certification_deadline')
            break
        manifest = tuple(DeliveryLot(int(i), float(instance['demands'][i]),
            int(instance['temp_class'][i]), wave_h + contract.loading_service_h,
            p.quality.initial_value[int(instance['temp_class'][i])],
            p.quality.initial_value[int(instance['temp_class'][i])]) for i in route)
        load = sum(l.quantity for l in manifest)
        if load > contract.capacity + 1e-9:
            errors.append('capacity')
        for i in route:
            if instance['reveal_time'][i] > wave_h + 1e-9 or instance['ready_time'][i] > wave_h + 1e-9:
                errors.append('unreleased_or_unready_loading')
        # Precondition at loading; no pickup/depot-unload transition is reused.
        temperatures = tuple(p.thermal.target_temperature_c)
        energy += sum(p.thermal.dispatch_preconditioning_energy_kwh)
        t, pos = wave_h + contract.loading_service_h, 0
        events = [{'kind': 'load', 'time': t, 'orders': list(route), 'load_after': load,
                   'preconditioning_energy_kwh': sum(p.thermal.dispatch_preconditioning_energy_kwh)}]

        def advance(end, distance, kind):
            nonlocal t, temperatures, manifest, energy, total_d
            active = tuple(any(l.temp_class == z for l in manifest) for z in range(3))
            before = temperatures
            temperatures, e = _advance_temperatures(before, end - t, active, p, t, end)
            manifest, _ = _advance_manifest_quality_v2(manifest, end - t, p)
            for z, bounds in enumerate(p.thermal.hard_temperature_bounds_c):
                if active[z] and (min(before[z], temperatures[z]) < bounds[0] - 1e-9 or
                                  max(before[z], temperatures[z]) > bounds[1] + 1e-9):
                    errors.append('loaded_temperature')
            energy += e
            total_d += distance
            events.append({'kind': kind, 'start': t, 'end': end, 'distance_km': distance,
                'energy_kwh': e, 'load': sum(l.quantity for l in manifest),
                'manifest': [l.order_id for l in manifest]})
            t = end

        for i in route:
            if deadline is not None and clock() >= deadline:
                errors.append('certification_deadline')
                break
            distance = float(dist[pos, i])
            advance(t + distance / contract.speed_kmph, distance, 'travel')
            advance(max(t, float(instance['tw_start'][i])), 0.0, 'wait')
            if t > float(instance['tw_end'][i]) + 1e-9:
                errors.append('time_window')
            active = tuple(any(l.temp_class == z for l in manifest) for z in range(3))
            temperatures, heat, recovery = _apply_door_event(temperatures, active,
                int(instance['temp_class'][i]), p)
            energy += recovery
            advance(t + float(instance['service_time'][i]), 0.0, 'service')
            lot = next(l for l in manifest if l.order_id == i)
            if p.quality.threshold_type[lot.temp_class] == 'absolute' and lot.quality_remaining < p.quality.threshold[lot.temp_class] - 1e-9:
                errors.append('quality_threshold')
            value_loss = contract.product_value[lot.temp_class] * lot.quantity * (1 - lot.quality_remaining / lot.initial_quality)
            loss += value_loss
            delivered.append({'order_id': int(i), 'vehicle': vid, 'loading_finish': lot.loading_finish_time,
                'delivery_finish': t, 'quantity': lot.quantity, 'quality': lot.quality_remaining,
                'quality_loss_value': value_loss})
            before_load = sum(l.quantity for l in manifest)
            manifest = tuple(l for l in manifest if l.order_id != i)
            events.append({'kind': 'deliver', 'order': int(i), 'time': t,
                'load_before': before_load, 'load_after': sum(l.quantity for l in manifest),
                'door_heat_kwh': heat, 'door_recovery_kwh': recovery})
            pos = i
        distance = float(dist[pos, 0])
        advance(t + distance / contract.speed_kmph, distance, 'return')
        if t > contract.horizon_h + 1e-9:
            errors.append('return_horizon')
        if manifest:
            errors.append('terminal_cargo_unresolved')
        traces.append({'vehicle': vid, 'return_time': t, 'events': events,
            'terminal_manifest': [l.order_id for l in manifest]})
    if deadline is not None and clock() >= deadline:
        errors.append('certification_deadline')
    hard = not errors and len(delivered) == n - 1
    cost = contract.distance_weight * total_d + contract.energy_weight * energy + contract.quality_weight * loss
    return {'hard': hard, 'complete': len(delivered) == n - 1,
        'errors': sorted(set(errors)), 'cost': cost if hard else None,
        'distance_km': total_d, 'energy_kwh': energy, 'quality_loss_value': loss,
        'delivered': delivered, 'traces': traces, 'contract_hash': contract.identity(),
        'qualification': 'delivery_engineering_snapshot_only', 'online_gain_established': False}
