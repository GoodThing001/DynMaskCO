"""Lossless structural target codec; no learned scoring or physical certification."""
def validate_routes(routes):
    if not isinstance(routes, list) or not routes:
        raise ValueError('Explicit ordered vehicle slots required')
    orders = [o for r in routes for o in r]
    if any(type(o) is not int or o <= 0 for o in orders) or len(set(orders)) != len(orders):
        raise ValueError('Positive unique orders required')
    return set(orders)


def encode(routes):
    validate_routes(routes)
    arcs = []
    for v, route in enumerate(routes):
        if route:
            path = [0] + route + [0]
            arcs.extend([v, a, b] for a, b in zip(path, path[1:]))
    return {'schema': 'vehicle-directed-arcs-v1', 'vehicle_slots': len(routes), 'arcs': arcs}


def decode(target):
    if target.get('schema') != 'vehicle-directed-arcs-v1':
        raise ValueError('Unknown schema')
    n = target['vehicle_slots']
    if type(n) is not int or n < 1:
        raise ValueError('Invalid slots')
    successors = [{} for _ in range(n)]
    predecessors = [{} for _ in range(n)]
    for event in target['arcs']:
        if len(event) != 3 or any(type(x) is not int for x in event):
            raise ValueError('Invalid arc')
        v, a, b = event
        if not 0 <= v < n or min(a, b) < 0 or a == b or a in successors[v] or b in predecessors[v]:
            raise ValueError('Invalid degree or slot')
        successors[v][a] = b
        predecessors[v][b] = a
    routes = []
    for succ in successors:
        route = []
        if succ:
            if 0 not in succ:
                raise ValueError('Missing depot')
            at = succ[0]
            while at != 0:
                if at in route or at not in succ:
                    raise ValueError('Disconnected/cyclic target')
                route.append(at)
                at = succ[at]
            if len(succ) != len(route) + 1:
                raise ValueError('Unused/disconnected arcs')
        routes.append(route)
    validate_routes(routes)
    return routes


def insertion_targets(initial, final, mask):
    """Teacher-forced structural actions. Never a production greedy decoder."""
    orders = validate_routes(initial)
    if validate_routes(final) != orders or len(initial) != len(final):
        raise ValueError('Changed fleet/orders')
    masked = set(mask)
    if len(masked) != len(mask) or not masked <= orders:
        raise ValueError('Invalid mask')
    skeleton = [[o for o in r if o not in masked] for r in initial]
    if skeleton != [[o for o in r if o not in masked] for r in final]:
        raise ValueError('Changed unmasked vehicle/order')
    work = [r[:] for r in skeleton]
    events = []
    for v, route in enumerate(final):
        for at, customer in enumerate(route):
            if customer in masked:
                predecessor = work[v][at - 1] if at else 0
                successor = work[v][at] if at < len(work[v]) else 0
                events.append({'customer': customer, 'vehicle': v, 'position': at,
                               'predecessor': predecessor, 'successor': successor})
                work[v].insert(at, customer)
    if work != final:
        raise ValueError('Target replay mismatch')
    return skeleton, events


def replay(skeleton, events):
    work = [r[:] for r in skeleton]
    validate_routes(work)
    seen = {o for r in work for o in r}
    for e in events:
        o, v, at = e['customer'], e['vehicle'], e['position']
        if any(type(x) is not int for x in (o, v, at)) or o <= 0 or o in seen or not 0 <= v < len(work) or not 0 <= at <= len(work[v]):
            raise ValueError('Invalid insertion action')
        if e['predecessor'] != (work[v][at - 1] if at else 0) or e['successor'] != (work[v][at] if at < len(work[v]) else 0):
            raise ValueError('Stale insertion anchors')
        work[v].insert(at, o)
        seen.add(o)
    return work
