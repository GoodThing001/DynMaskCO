"""Directed masked cycle completion; retained arcs and waits are immutable."""
from __future__ import annotations

import numpy as np
from rrp_data import PRIMARY_PROFILE


def build_cycle(scores, kept_edges):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if scores.shape != (n, n) or n < 2 or not np.all(np.isfinite(scores)):
        raise ValueError('Invalid directed scores')
    parent = list(range(n))
    outgoing, incoming = {}, {}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def add(a, b, required=False):
        if a == b or not (0 <= a < n and 0 <= b < n) or a in outgoing or b in incoming:
            if required:
                raise ValueError('Inconsistent kept arcs')
            return False
        x, y = find(a), find(b)
        if x == y and len(outgoing) < n - 1:
            if required:
                raise ValueError('Kept subtour')
            return False
        outgoing[a], incoming[b] = b, a
        parent[x] = y
        return True

    for a, b in sorted(kept_edges):
        add(int(a), int(b), required=True)
    candidates = sorted(((-scores[a, b], a, b) for a in range(n) for b in range(n) if a != b))
    for _, a, b in candidates:
        if len(outgoing) == n:
            break
        add(a, b)
    if len(outgoing) != n:
        raise ValueError('Unable to reconstruct a complete directed cycle')
    route = [0]
    for _ in range(n - 1):
        route.append(outgoing[route[-1]])
    if set(route) != set(range(n)) or outgoing[route[-1]] != 0:
        raise ValueError('Incomplete/subtour reconstruction')
    return route


def reconstruct(model, instance, incumbent, kept_edges, waits_s, *, objective='with_wage', profile=PRIMARY_PROFILE, time_domain='source_day_domain'):
    from rrp_learning import make_batch, predict
    batch = make_batch(instance, incumbent, waits_s, kept_edges, objective=objective, profile=profile, time_domain=time_domain)
    scores, predicted_waits = predict(model, batch)
    route = build_cycle(scores, kept_edges)
    original_wait = dict(zip(incumbent, waits_s))
    known_sources = {a for a, _ in kept_edges}
    waits = [0.0 if node == 0 else float(original_wait[node] if node in known_sources else predicted_waits[node]) for node in route]
    return {'route': route, 'waits_s': waits, 'kept_edges': sorted(kept_edges),
            'actual_maskco_encoder_decoder_used': True, 'directed_head_used': True}
