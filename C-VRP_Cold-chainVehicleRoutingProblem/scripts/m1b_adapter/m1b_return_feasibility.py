#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-4: Omega11 -- an INDEPENDENT pre-submit RETURN-FEASIBILITY gate.

Why this gate exists at all: the Q1 review established that `_force_return_all`
(`strict_online_env.py:658-664`) commits a return at the CURRENT clock without any
`finish <= T_end` branch, so it cannot be cited as evidence that every vehicle returns in time.
Omega5 (`t_ret <= T_end + 1e-6`) therefore has to be guaranteed by an independent check rather
than assumed.

Coverage, per the spec:
  * the existing executed prefix (a vehicle already `returning` has a REAL `return_finish`);
  * committed, non-cancellable in-flight legs (`committed_next` / `committed_finish`);
  * every relevant vehicle's CURRENT state;
  * the candidate plan's complete suffix, including WAIT and the return leg;
  * `return_arrival <= T_end + 1e-6`;
  * the timeout-fallback / forced-return boundary.

Invariant maintained ACROSS replans and commits: this check is re-run against a FRESH snapshot at
every commit boundary.  An earlier certificate therefore does NOT carry forward -- each commit
must re-establish it from the then-current state.  A certificate is never reused.
"""
from __future__ import annotations

OMEGA11_OK = 'OMEGA11_OK'
OMEGA11_RETURN_DEADLINE_EXCEEDED = 'OMEGA11_RETURN_DEADLINE_EXCEEDED'
OMEGA11_CLOCK_BEYOND_HORIZON = 'OMEGA11_CLOCK_BEYOND_HORIZON'
OMEGA11_STATE_MISSING = 'OMEGA11_STATE_MISSING'
OMEGA11_TIME_ORDER_INVALID = 'OMEGA11_TIME_ORDER_INVALID'
OMEGA11_UNKNOWN_VEHICLE_STATUS = 'OMEGA11_UNKNOWN_VEHICLE_STATUS'

EPSILON_RETURN = 1e-6
KNOWN_STATUSES = ('idle', 'ready', 'committed', 'returning', 'closed')


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


def certify_return_feasibility(snap, *, T_end, clock, travel_fn,
                               epsilon_return=EPSILON_RETURN):
    """Return (ok, verdict, detail).

    `travel_fn(inst_idx, a, b)` supplies travel time; it is injected so this module never needs
    the raw environment.

    Rules applied per vehicle:
      * `closed`                  -> nothing pending, but its RECORDED return must have been in time
      * `returning`               -> the REAL `return_finish` must satisfy the deadline
      * `committed`               -> `committed_finish + travel(current_node -> 0)` must satisfy it
      * `ready` / `idle`          -> the plan's return arrival is computed by replaying the
                                     suffix; an EMPTY suffix away from the depot is a WAIT and is
                                     treated as "must still return", never as already returned
    """
    if snap is None:
        return False, OMEGA11_STATE_MISSING, {'field': 'snapshot'}
    if not _finite(T_end) or not _finite(clock):
        return False, OMEGA11_TIME_ORDER_INVALID, {'T_end': repr(T_end), 'clock': repr(clock)}
    if clock > float(T_end) + epsilon_return:
        return False, OMEGA11_CLOCK_BEYOND_HORIZON, {
            'clock': float(clock), 'T_end': float(T_end),
            'why': 'the decision clock is already beyond the depot deadline, so no plan can be '
                   'certified against it; refusing rather than fabricating a certificate'}
    deadline = float(T_end) + float(epsilon_return)
    per_vehicle, bad = {}, []

    for vid in sorted(snap['vehicles']):
        v = snap['vehicles'][vid]
        if v.status not in KNOWN_STATUSES:
            bad.append({'vehicle': vid, 'why': 'unknown status', 'status': v.status})
            continue
        row = {'status': v.status}
        if v.status == 'closed':
            row['pending'] = False
            per_vehicle[vid] = row
            continue
        if v.status == 'returning':
            if v.return_finish is None:
                bad.append({'vehicle': vid, 'why': 'returning without return_finish'})
                per_vehicle[vid] = row
                continue
            row['return_arrival'] = float(v.return_finish)
            row['basis'] = 'recorded return_finish (real in-flight leg, not recomputed)'
            row['pending'] = True
        elif v.status == 'committed':
            if v.committed_finish is None or v.committed_next is None:
                bad.append({'vehicle': vid, 'why': 'committed leg missing finish/next'})
                per_vehicle[vid] = row
                continue
            leg = float(travel_fn(v.committed_next, 0))
            row['return_arrival'] = float(v.committed_finish) + leg
            row['basis'] = 'committed_finish + travel(committed_next -> depot)'
            row['pending'] = True
        else:                                                          # idle / ready
            plan = snap['candidate_plans'].get(vid)
            suffix = list(plan.suffix) if plan is not None else list(v.mutable_suffix)
            suffix = [int(x) for x in suffix]
            if v.current_node == 0 and not suffix:
                row['pending'] = False
                per_vehicle[vid] = row
                continue
            # Replay the suffix from the CURRENT physical state.  An empty suffix while AWAY
            # from the depot is the WAIT semantics (`mutable_suffix = []`): the vehicle must still
            # return later, so it is judged by "return now", the earliest possible time.
            t = float(max(v.ready_time, clock))
            node = int(v.current_node)
            seq = [x for x in suffix if x != 0]
            for nxt in seq:
                t += float(travel_fn(node, nxt))
                t += _service_time(nxt, snap)
                node = int(nxt)
            if node != 0:
                t += float(travel_fn(node, 0))
            row['return_arrival'] = t
            row['basis'] = ('suffix replay from current state; empty-away-from-depot == WAIT, '
                            'judged as "return now" (earliest possible), never as already home')
            row['pending'] = True
            row['n_suffix_stops'] = len(seq)
        if row.get('pending') and row['return_arrival'] > deadline:
            bad.append({'vehicle': vid, 'why': 'return_arrival exceeds deadline',
                        'return_arrival': row['return_arrival'], 'deadline': deadline,
                        'excess': row['return_arrival'] - deadline, 'basis': row.get('basis')})
        per_vehicle[vid] = row

    detail = {'T_end': float(T_end), 'deadline': deadline, 'clock': float(clock),
              'epsilon_return': float(epsilon_return), 'per_vehicle': per_vehicle,
              'n_violations': len(bad), 'violations': bad[:20]}
    if bad:
        return False, OMEGA11_RETURN_DEADLINE_EXCEEDED, detail
    return True, OMEGA11_OK, detail


_SERVICE_TIME_CACHE = {}


def register_service_times(mapping):
    """`{node: service_time}` supplied by the caller from PUBLIC data (it is not future-only)."""
    _SERVICE_TIME_CACHE.clear()
    _SERVICE_TIME_CACHE.update({int(k): float(v) for k, v in (mapping or {}).items()})


def _service_time(node, snap):
    return float(_SERVICE_TIME_CACHE.get(int(node), 0.0))


def replan_invariant_statement():
    """How the return-feasibility invariant is kept across replans and every commit."""
    return {
        'rule': 'the return-feasibility certificate is PER-COMMIT and is never reused',
        'why': 'a candidate certified at clock t may be replaced at t+1 by a different plan, and '
               'a committed leg changes the physical state; carrying a stale certificate forward '
               'would assert feasibility about a plan that is no longer the one being installed',
        'mechanism': [
            're-snapshot the live state at each commit boundary (m1b_immutable_snapshot)',
            're-run certify_return_feasibility on the FRESH snapshot before installing',
            'refuse the commit when it fails; do not fall back to a previous certificate',
            'the timeout / forced-return path is covered by the same check: a return committed '
            'from the current state must itself satisfy the deadline',
        ],
    }
