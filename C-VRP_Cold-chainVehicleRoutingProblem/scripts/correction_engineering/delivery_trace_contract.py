"""Executable delivery event invariants; engineering verification, not a solver.

Literature mapping: S02/S03 release-ready/finite fleet; S06 frozen decisions;
S08 compartment assumption. Actual energy/quality certification is separate.
"""
import math


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_trace(orders, events, *, vehicles, capacity, horizon,
                   loading_duration, speed, coordinates, compartment_capacity=None):
    require(type(vehicles) is int and vehicles > 0, 'fleet')
    require(all(finite(x) and x > 0 for x in (capacity, horizon, speed)), 'units')
    require(finite(loading_duration) and loading_duration >= 0, 'loading_duration')
    require(0 in coordinates, 'depot_coordinates')
    require(all(len(p) == 2 and all(finite(x) for x in p) for p in coordinates.values()), 'coordinates')
    if compartment_capacity is not None:
        require(len(compartment_capacity) == 3 and all(finite(x) and x >= 0 for x in compartment_capacity), 'compartments')
        require(sum(compartment_capacity) <= capacity + 1e-9, 'compartment_sum')
    for oid, order in orders.items():
        require(type(oid) is int and oid > 0 and oid in coordinates, 'order_id')
        require(all(finite(order[k]) for k in ('quantity','release','ready','tw_start','tw_end','service')), 'order_finite')
        require(order['quantity'] > 0 and min(order['release'],order['ready'],order['tw_start'],order['service']) >= 0, 'order_range')
        require(order['tw_end'] >= order['tw_start'], 'window')
        require(type(order['zone']) is int and 0 <= order['zone'] < 3, 'zone')
    owner = {oid: 'warehouse' for oid in orders}
    fleet = {i: {'node': 0, 'mode': 'idle', 'cargo': set(), 'suffix': [],
                 'target': None, 'until': 0., 'load_ids': set()} for i in range(vehicles)}
    last = 0.
    delivered = set()
    distance = 0.
    for n, event in enumerate(events):
        t, kind, vid = event['time'], event['kind'], event['vehicle']
        require(finite(t) and last <= t <= horizon, 'event_clock')
        require(type(vid) is int and vid in fleet, 'unknown_vehicle')
        last = t
        v = fleet[vid]
        if kind == 'load_start':
            ids = event['orders']
            require(len(ids) == len(set(ids)) and bool(ids), 'duplicate_or_empty_load')
            require(v['node'] == 0 and v['mode'] == 'idle' and not v['cargo'], 'load_requires_idle_depot')
            require(all(oid in orders and owner[oid] == 'warehouse' for oid in ids), 'cargo_ownership')
            require(all(max(orders[oid]['release'], orders[oid]['ready']) <= t for oid in ids), 'unreleased_or_unready')
            require(sum(orders[oid]['quantity'] for oid in ids) <= capacity + 1e-9, 'total_capacity')
            if compartment_capacity is not None:
                require(all(sum(orders[oid]['quantity'] for oid in ids if orders[oid]['zone'] == z) <= compartment_capacity[z] + 1e-9 for z in range(3)), 'zone_capacity')
            v.update(mode='loading', until=t+loading_duration, load_ids=set(ids))
            for oid in ids:
                owner[oid] = ('loading', vid)
        elif kind == 'load_finish':
            require(v['mode'] == 'loading' and t+1e-9 >= v['until'], 'loading_not_finished')
            v.update(mode='idle', cargo=set(v['load_ids']), load_ids=set())
            for oid in v['cargo']:
                owner[oid] = ('vehicle', vid)
        elif kind == 'plan':
            route = event['suffix']
            require(bool(route) and route[-1] == 0 and 0 not in route[:-1], 'route_return')
            customers = route[:-1]
            require(len(customers) == len(set(customers)), 'duplicate_route_order')
            require(set(customers) == v['cargo'], 'route_manifest_mismatch')
            if v['mode'] in ('travel','service') and v['target'] != 0:
                require(route[0] == v['target'], 'committed_leg_changed')
            if v['mode'] == 'travel' and v['target'] == 0:
                require(route == [0], 'committed_return_changed')
            v['suffix'] = list(route)
        elif kind == 'depart':
            require(v['mode'] == 'idle' and v['suffix'], 'busy_or_unplanned_departure')
            target = v['suffix'][0]
            require(target == event['target'], 'departure_differs_from_plan')
            require(target == 0 or owner.get(target) == ('vehicle',vid), 'unloaded_delivery')
            a,b = coordinates[v['node']],coordinates[target]
            d = math.hypot(a[0]-b[0],a[1]-b[1])
            v.update(mode='travel', target=target, until=t+d/speed)
            distance += d
        elif kind == 'arrive':
            require(v['mode'] == 'travel' and event['target'] == v['target'] and t+1e-9 >= v['until'], 'invalid_arrival')
            v.update(node=v['target'], mode='idle')
            if v['node'] == 0:
                require(not v['cargo'], 'return_with_undelivered_cargo')
                v.update(suffix=[], target=None)
        elif kind == 'service_start':
            oid = event['order']
            require(v['mode'] == 'idle' and v['node'] == oid and owner.get(oid) == ('vehicle',vid), 'invalid_service')
            order=orders[oid]
            require(order['tw_start'] <= t <= order['tw_end'], 'service_start_window')
            v.update(mode='service', target=oid, until=t+order['service'])
        elif kind == 'service_finish':
            oid=event['order']
            require(v['mode'] == 'service' and v['target'] == oid and t+1e-9 >= v['until'], 'invalid_service_finish')
            require(v['suffix'] and v['suffix'][0] == oid, 'service_plan_mismatch')
            v['cargo'].remove(oid)
            owner[oid]='delivered'
            delivered.add(oid)
            v.update(mode='idle', target=None, suffix=v['suffix'][1:])
        else:
            raise ValueError('Unknown event kind: '+kind)
        # Every item has exactly one owner, including during loading and travel.
        for oid in orders:
            holders=sum(oid in f['cargo'] or oid in f['load_ids'] for f in fleet.values())
            require(holders == (1 if isinstance(owner[oid],tuple) else 0), 'conservation')
    require(delivered == set(orders), 'incomplete_delivery')
    require(all(v['node']==0 and v['mode']=='idle' and not v['cargo'] for v in fleet.values()), 'unfinished_fleet')
    return {'event_contract_pass': True, 'delivered': sorted(delivered), 'distance_km': distance,
            'quality_energy_certified': False, 'formal_online_ready': False}
