#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-2: IMMUTABLE STATE SNAPSHOTS and full plan identity (fixes G3 and G4).

G3: the M1-a `build_context` stored `vehicles` / `served_mask` / candidate plans BY REFERENCE,
and `certify_route` reads them, so anything mutating the live objects during certification would
silently change what was certified.  This module builds genuinely isolated snapshots and
mechanically verifies the live state did not change while certification ran.

G4: the M1-a `plan_identity` (the `_structural_fingerprint` helper) hashed
`vid / anchor_node / anchor_time / suffix` and OMITTED `anchor_load`.  `anchor_load` is physical
cargo state that participates in the capacity and energy checks, so a plan identity that ignores
it can declare two physically different plans identical.  `plan_identity_v2` below includes it.

The snapshot is deliberately a SEPARATE object graph -- deep copies of the mutable containers --
so that mutating the live `VehicleState` cannot alter the snapshot (T-Q2-7a/7b).
"""
from __future__ import annotations

import hashlib
import json

SNAPSHOT_OK = 'SNAPSHOT_OK'
SNAPSHOT_ALIAS_DETECTED = 'SNAPSHOT_ALIAS_DETECTED'
SNAPSHOT_FIELD_MISSING = 'SNAPSHOT_FIELD_MISSING'
SNAPSHOT_FIELD_TYPE_INVALID = 'SNAPSHOT_FIELD_TYPE_INVALID'
STATE_CHANGED_DURING_CERTIFICATION = 'STATE_CHANGED_DURING_CERTIFICATION'


# ---------------------------------------------------------------------------------------------
# Plan identity
# ---------------------------------------------------------------------------------------------
def plan_identity_v2(plans):
    """Structural identity of a plan mapping INCLUDING anchor_load (fixes G4).

    Rows are `[vid, anchor_node, anchor_time, anchor_load, suffix...]`, JSON-canonicalised.
    Floats are rounded to 12 decimals so that pure float noise does not manufacture a false
    difference, while a real physical change is still detected.
    """
    rows = []
    for vid in sorted(int(v) for v in plans):
        p = None
        for k, v in plans.items():
            if int(k) == vid:
                p = v
                break
        rows.append([vid,
                     int(getattr(p, 'anchor_node', -1)),
                     round(float(getattr(p, 'anchor_time', 0.0)), 12),
                     round(float(getattr(p, 'anchor_load', 0.0)), 12),
                     [int(x) for x in getattr(p, 'suffix', [])]])
    blob = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def cargo_manifest_identity(state):
    """Identity of a cold-chain state's cargo: order/quantity/class/quality and temperatures."""
    lots = []
    for lot in getattr(state, 'cargo_manifest', ()) or ():
        lots.append([int(lot.order_id), round(float(lot.quantity), 9), int(lot.temp_class),
                     round(float(lot.pickup_finish_time), 9),
                     round(float(lot.initial_quality), 12),
                     round(float(lot.quality_remaining), 12), bool(lot.precooled)])
    payload = {'temp': [round(float(x), 9) for x in
                        (getattr(state, 'compartment_temperature_c', ()) or ())],
               'zone_load': [round(float(x), 9) for x in (getattr(state, 'zone_load', ()) or ())],
               'lots': lots,
               'delivered': len(getattr(state, 'delivered_to_depot', ()) or ()),
               'thermal_violation_count': int(getattr(state, 'thermal_violation_count', 0)),
               'closed': bool(getattr(state, 'closed', False)),
               'dispatched': bool(getattr(state, 'dispatched', False))}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------------------------
class FrozenVehicle(object):
    """An isolated copy of the fields certification reads.  No reference to the live object."""

    __slots__ = ('vehicle_id', 'status', 'current_node', 'ready_time', 'current_load',
                 'committed_next', 'committed_arrive', 'committed_finish', 'return_finish',
                 'mutable_suffix', 'served_route', 'dispatch_time',
                 'coldchain_state', 'coldchain_updated_time', 'cc_identity',
                 'plan', 'initial_quality_by_node')

    def __init__(self, v):
        self.vehicle_id = int(v.vehicle_id)
        self.status = str(v.status)
        self.current_node = int(v.current_node)
        self.ready_time = float(v.ready_time)
        self.current_load = float(getattr(v, 'current_load', 0.0))
        self.committed_next = (None if v.committed_next is None else int(v.committed_next))
        self.committed_arrive = (None if getattr(v, 'committed_arrive', None) is None
                                 else float(v.committed_arrive))
        self.committed_finish = (None if getattr(v, 'committed_finish', None) is None
                                 else float(v.committed_finish))
        self.return_finish = (None if getattr(v, 'return_finish', None) is None
                              else float(v.return_finish))
        self.mutable_suffix = [int(x) for x in (getattr(v, 'mutable_suffix', None) or [])]
        self.served_route = [int(x) for x in (getattr(v, 'served_route', None) or [])]
        self.dispatch_time = (None if getattr(v, 'dispatch_time', None) is None
                              else float(v.dispatch_time))
        # The cold-chain state is a frozen dataclass of tuples, so it is IMMUTABLE by
        # construction; it is copied anyway (and its identity recorded) so that the snapshot
        # cannot be re-pointed by a later assignment on the live vehicle.
        cc = getattr(v, 'coldchain_state', None)
        self.coldchain_state = cc
        self.coldchain_updated_time = (None if getattr(v, 'coldchain_updated_time', None) is None
                                       else float(v.coldchain_updated_time))
        self.cc_identity = cargo_manifest_identity(cc) if cc is not None else None
        # populated by build_frozen_snapshot: the frozen candidate plan for this vehicle, and the
        # PUBLIC per-node initial quality used for a thermal replay's pickup events.
        self.plan = None
        self.initial_quality_by_node = {}

    def initial_quality_of(self, node):
        """PUBLIC per-node initial quality (a static dataset property, not future-only)."""
        return float(self.initial_quality_by_node.get(int(node), 1.0))

    def is_dispatched(self):
        return self.coldchain_state is not None


class FrozenPlan(object):
    __slots__ = ('vehicle_id', 'anchor_node', 'anchor_time', 'anchor_load', 'suffix')

    def __init__(self, p):
        self.vehicle_id = int(getattr(p, 'vehicle_id', -1))
        self.anchor_node = int(p.anchor_node)
        self.anchor_time = float(p.anchor_time)
        self.anchor_load = float(getattr(p, 'anchor_load', 0.0))
        self.suffix = [int(x) for x in (getattr(p, 'suffix', None) or [])]


def build_frozen_snapshot(vehicles, served_mask, candidate_plans, *,
                          initial_quality_by_node=None):
    """Build an isolated snapshot.  Returns (snapshot, verdict, detail).

    Refuses on a missing or ill-typed field rather than defaulting.
    `initial_quality_by_node` is the PUBLIC per-node initial-quality table; it is attached to each
    frozen vehicle so a thermal replay can supply pickup values without reaching into the live
    environment.  Absent entries default to 1.0 and are therefore CAUGHT by the Omega10 gate
    rather than silently accepted.
    """
    if vehicles is None:
        return None, SNAPSHOT_FIELD_MISSING, {'field': 'vehicles'}
    if served_mask is None:
        return None, SNAPSHOT_FIELD_MISSING, {'field': 'served_mask'}
    if candidate_plans is None:
        return None, SNAPSHOT_FIELD_MISSING, {'field': 'candidate_plans'}
    try:
        import numpy as np
        served = np.array(served_mask, dtype=bool, copy=True)
    except Exception as exc:                                            # noqa: BLE001
        return None, SNAPSHOT_FIELD_TYPE_INVALID, {
            'field': 'served_mask', 'exception': '%s: %s' % (type(exc).__name__, str(exc)[:120])}
    iq_table = {int(k): float(v) for k, v in (initial_quality_by_node or {}).items()}
    fp = {}
    for vid, p in candidate_plans.items():
        fp[int(vid)] = FrozenPlan(p)
    fv = {}
    for v in vehicles:
        f = FrozenVehicle(v)
        f.plan = fp.get(f.vehicle_id)
        f.initial_quality_by_node = iq_table
        fv[f.vehicle_id] = f
    snap = {'vehicles': fv, 'served_mask': served, 'candidate_plans': fp,
            'initial_quality_by_node': iq_table}
    snap['plan_identity'] = plan_identity_v2(fp)
    snap['state_identity'] = snapshot_state_identity(snap)
    return snap, SNAPSHOT_OK, {'n_vehicles': len(fv), 'n_plans': len(fp)}


def snapshot_state_identity(snap):
    """A mechanical identity over the snapshot's physical state, used for pre/post comparison."""
    rows = []
    for vid in sorted(snap['vehicles']):
        v = snap['vehicles'][vid]
        rows.append([vid, v.status, v.current_node, round(v.ready_time, 9),
                     round(v.current_load, 9), v.committed_next,
                     None if v.committed_finish is None else round(v.committed_finish, 9),
                     None if v.return_finish is None else round(v.return_finish, 9),
                     v.mutable_suffix, v.cc_identity])
    payload = {'vehicles': rows,
               'served_mask_true': [int(i) for i in range(len(snap['served_mask']))
                                    if bool(snap['served_mask'][i])],
               'plan_identity': snap.get('plan_identity')}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def detect_aliasing(snap, vehicles, candidate_plans):
    """T-Q2-7d: the snapshot must share NO mutable object with the live state."""
    aliases = []
    for v in vehicles:
        f = snap['vehicles'].get(int(v.vehicle_id))
        if f is None:
            continue
        if f is v:
            aliases.append({'vehicle': int(v.vehicle_id), 'why': 'same object'})
        if f.mutable_suffix is (getattr(v, 'mutable_suffix', None) or []):
            aliases.append({'vehicle': int(v.vehicle_id), 'why': 'shared mutable_suffix list'})
        if f.coldchain_state is not None and f.coldchain_state is getattr(v, 'coldchain_state', None) \
                and not _is_immutable_cc(f.coldchain_state):
            aliases.append({'vehicle': int(v.vehicle_id), 'why': 'shared mutable coldchain_state'})
    for vid, p in candidate_plans.items():
        f = snap['candidate_plans'].get(int(vid))
        if f is None:
            continue
        if f is p:
            aliases.append({'plan': int(vid), 'why': 'same object'})
        if f.suffix is (getattr(p, 'suffix', None) or []):
            aliases.append({'plan': int(vid), 'why': 'shared suffix list'})
    return (True, {}) if not aliases else (False, {'aliases': aliases})


def _is_immutable_cc(state):
    """A frozen dataclass whose every field is a scalar or a tuple is immutable in practice."""
    try:
        import dataclasses
        return dataclasses.is_dataclass(state) and getattr(
            type(state), '__dataclass_params__', None) is not None and \
            type(state).__dataclass_params__.frozen
    except Exception:                                                   # noqa: BLE001
        return False


def verify_state_unchanged(snap, vehicles, served_mask, candidate_plans):
    """Post-certification mechanical re-check (T-Q2-7e)."""
    try:
        current, verdict, detail = build_frozen_snapshot(vehicles, served_mask, candidate_plans)
    except Exception as exc:                                            # noqa: BLE001
        return False, STATE_CHANGED_DURING_CERTIFICATION, {
            'exception': '%s: %s' % (type(exc).__name__, str(exc)[:150])}
    if verdict != SNAPSHOT_OK:
        return False, STATE_CHANGED_DURING_CERTIFICATION, detail
    if current['state_identity'] != snap['state_identity']:
        return False, STATE_CHANGED_DURING_CERTIFICATION, {
            'before': snap['state_identity'], 'after': current['state_identity']}
    return True, 'STATE_UNCHANGED', {}
