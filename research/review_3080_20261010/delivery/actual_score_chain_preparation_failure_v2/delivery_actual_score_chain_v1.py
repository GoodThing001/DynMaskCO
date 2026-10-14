"""Unexecuted engineering consumer of bound, untrained actual model scores.

Complete temporary skeleton identity is checked before every transition. This
does not grant trained-model, full-C0, online-budget or scientific eligibility.
"""
import math
import numpy as np
from action_head_runtime_v1 import digest
from delivery_feasible_action_options_v1 import options


def enumerate_step(data, routes, remaining, partial, contract):
    flat = [i for route in routes for i in route]
    n = len(data['demands'])
    if len(flat) != len(set(flat)) or len(remaining) != len(set(remaining)):
        raise ValueError('Duplicate current customer')
    if set(flat) & set(remaining) or set(flat) | set(remaining) != set(range(1, n)):
        raise ValueError('Exact current skeleton/remaining coverage')
    if remaining != sorted(remaining) or not remaining:
        raise ValueError('Nonempty ascending remaining customers')
    partial = np.asarray(partial)
    zeros = np.zeros(partial.shape, dtype=float)
    groups = []
    actions = []
    for customer in remaining:
        rows = options(data, routes, customer, partial, zeros, contract)['rows']
        offset = len(actions)
        actions.extend([[customer, r['vehicle'], r['position'], r['predecessor'], r['successor']] for r in rows)
        groups.append(dict(customer=customer, offset=offset, rows=rows))
    actions = np.asarray(actions, np.int32)
    if actions.ndim != 2 or actions.shape[1] != 5:
        raise ValueError('Full ordered action enumeration')
    return actions, groups


def binding_identity(data, key, step, routes, remaining, actions):
    return digest(dict(data=data, key=key, step=step, before=routes,
                       remaining=remaining, actions=actions.tolist()))


def consume_step(data, key, step, routes, remaining, actions, groups, packet,
                 expected_source_identity):
    expected = binding_identity(data, key, step, routes, remaining, actions)
    if packet.get('binding_digest') != expected:
        raise ValueError('Stale context/action/step/data identity')
    if packet.get('source_identity') != expected_source_identity:
        raise ValueError('Actual source/checkpoint/arm/head identity')
    if packet.get('source') != 'actual_initial_model_untrained':
        raise ValueError('This consumer only accepts declared initial-model engineering output')
    if packet.get('learned_output_qualified') is not False:
        raise ValueError('Initial engineering cannot claim learned output')
    scores = packet.get('scores')
    if not isinstance(scores, np.ndarray) or scores.dtype != np.float32:
        raise ValueError('Exact float32 score vector')
    if scores.shape != (len(actions),) or not np.isfinite(scores).all():
        raise ValueError('One finite score per ordered action')
    customers = []
    rebuilt = []
    for group in groups:
        customer = group['customer']
        rows = group['rows']
        if group['offset'] != len(rebuilt):
            raise ValueError('Noncontiguous customer group')
        packed = []
        for row in rows:
            a = [customer, row['vehicle'], row['position'], row['predecessor'], row['successor']]
            if row['customer'] != customer or a != actions[len(rebuilt)].tolist():
                raise ValueError('Group/action mismatch')
            route = routes[a[1]]
            if customer not in remaining or not 0 <= a[2] <= len(route):
                raise ValueError('Action outside current temporary domain')
            if a[3] != (route[a[2]-1] if a[2] else 0) or a[4] != (route[a[2]] if a[2] < len(route) else 0):
                raise ValueError('Stale predecessor/successor')
            value = float(scores[len(rebuilt)])
            tie = row['score']  # Zero-logit delta / original mean-distance scale.
            if not math.isfinite(tie):
                raise ValueError('Finite original distance tie')
            packed.append(a + [bool(row['retained_union_ok']), bool(row['original_route_ok']), value, tie])
            rebuilt.append(a)
        eligible = sorted((r for r in packed if r[5] and r[6]),
                          key=lambda r: (-r[7], r[8], r[1], r[2]))
        best = eligible[0] if eligible else None
        regret = None if len(eligible) == 1 else eligible[0][7] - eligible[1][7] if eligible else 0.
        customers.append(dict(customer=customer, rows=packed, best=best, regret=regret))
    if rebuilt != actions.tolist() or [r['customer'] for r in customers] != remaining:
        raise ValueError('Full ordered customer/action coverage')
    feasible = [r for r in customers if r['best'] is not None]
    chosen = max(feasible, key=lambda r: (r['regret'] is None,
                 0. if r['regret'] is None else r['regret'], -r['customer']))['best'] if feasible else None
    after = [r[:] for r in routes]
    left = remaining[:]
    if chosen is not None:
        after[chosen[1]].insert(chosen[2], chosen[0])
        left.remove(chosen[0])
    return dict(before=[r[:] for r in routes], remaining=remaining[:], customers=customers,
                chosen=chosen, after=after, next_remaining=left, binding_digest=expected,
                source_identity=expected_source_identity,
                status='inserted' if chosen is not None else 'no_feasible_action',
                learned_output_qualified=False)
