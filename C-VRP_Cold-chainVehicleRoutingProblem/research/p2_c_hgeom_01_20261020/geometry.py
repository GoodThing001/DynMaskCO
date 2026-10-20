"""Contract-frozen geometry; no dataset or model loader."""
from __future__ import annotations
import math


def residual(plans, dist_mat):
    edges = []
    for vid in sorted(plans):
        p = plans[vid]
        seq = (int(p.anchor_node), *map(int, p.suffix), 0)
        for a, b in zip(seq, seq[1:]):
            d = float(dist_mat[0, a, b])
            if not math.isfinite(d) or d < 0:
                raise ValueError('METRIC_INDETERMINATE')
            edges.append(d)
    value = math.fsum(edges)
    if not math.isfinite(value):
        raise ValueError('METRIC_INDETERMINATE')
    return value


def relation(model, enum):
    if any(not math.isfinite(x) or x < 0 for x in (model, enum)):
        raise ValueError('METRIC_INDETERMINATE')
    g = model - enum
    label = 'ENUM_STRICTLY_BETTER' if g > 0 else 'ENUM_TIE' if g == 0 else 'CONSISTENCY_ERROR'
    gap = g / enum if enum else None
    return dict(distance_relation=label, g=g, relative_gap=gap,
                within_one_percent=None if gap is None else 0 <= gap <= .01,
                g_hex=g.hex(), model_hex=model.hex(), reference_hex=enum.hex(),
                ulp_model=math.ulp(model), ulp_reference=math.ulp(enum),
                gap_in_ulps=abs(g) / max(math.ulp(model), math.ulp(enum)))


def primary(model, distance_values):
    if not distance_values or any(not math.isfinite(x) or x < 0
                                  for x in (model, *distance_values)):
        raise ValueError('METRIC_INDETERMINATE')
    best = min(distance_values)
    delta = model - best
    ratio = delta / best if best else None
    return dict(delta=delta, ratio=ratio, improve=ratio is not None and ratio <= -.01,
                distance_best=best, delta_hex=delta.hex())


def package_gate(rows, n_units=19, n_targets=123):
    return (len(rows) == n_units and len({r['unit_key'] for r in rows}) == n_units
            and sum(r['n_targets'] for r in rows) == n_targets
            and all(r['targets_found'] == r['n_targets'] and r['uniqueness_proved']
                    and r['closure_exhausted'] and r['legal'] and not r['failure_codes']
                    for r in rows))
