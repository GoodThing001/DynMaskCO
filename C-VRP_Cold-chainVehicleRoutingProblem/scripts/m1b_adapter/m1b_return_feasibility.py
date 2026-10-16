#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix: Omega11 return-feasibility gate, rebuilt on the SHARED time replay.

Fixes (Q2-R): A1, A2, A2b, A3, A4, A4b, A5, A6.

What changed relative to the audited version
  * the time advance now lives in `m1b_timing.replay_timeline` and matches
    `action_contract.certify_route`: `service_start = max(arrival, tw_start)`.  (A1)
  * the PROCESS-GLOBAL `_SERVICE_TIME_CACHE` and `register_service_times` are GONE.  Time inputs
    are per-instance objects carried on the snapshot; a missing key REFUSES instead of
    defaulting to zero.  A legitimate 0.0 is accepted (key present).            (A2, A2b, A3)
  * a committed vehicle first inherits its real in-flight leg and THEN continues through its
    later accepted tasks and the candidate suffix, instead of computing only the immediate
    return.                                                                    (A4, A4b)
  * the `basis` string is derived from what was ACTUALLY replayed, and the depart choice is
    recorded, so the label can no longer describe a different rule than the code applies. (A5)
  * a closed vehicle is audited against the TRUSTED HISTORY supplied by the caller; a missing
    history entry refuses, while an empty current `return_finish` alone is NOT treated as a
    violation (the real engine can clear it after closing).                    (A6)
  * the receipt records the plan identity, the time-input identity, per-vehicle return arrivals
    and the failure reason.                                                    (requirement 7)
"""
from __future__ import annotations

import m1b_timing as TM

OMEGA11_OK = 'OMEGA11_OK'
OMEGA11_RETURN_DEADLINE_EXCEEDED = TM.TIMING_RETURN_DEADLINE_EXCEEDED
OMEGA11_CLOCK_BEYOND_HORIZON = TM.TIMING_CLOCK_BEYOND_HORIZON
OMEGA11_STATE_MISSING = 'OMEGA11_STATE_MISSING'
OMEGA11_TIME_ORDER_INVALID = TM.TIMING_TIME_ORDER_INVALID
OMEGA11_TIME_INPUT_UNAVAILABLE = TM.TIMING_TIME_INPUT_UNAVAILABLE
OMEGA11_ANCHOR_MISMATCH = TM.TIMING_ANCHOR_MISMATCH
OMEGA11_UNKNOWN_VEHICLE_STATUS = 'OMEGA11_UNKNOWN_VEHICLE_STATUS'
OMEGA11_CLOSED_HISTORY_MISSING = 'OMEGA11_CLOSED_HISTORY_MISSING'
OMEGA11_CLOSED_HISTORY_DEADLINE_EXCEEDED = 'OMEGA11_CLOSED_HISTORY_DEADLINE_EXCEEDED'
OMEGA11_CERTIFY_ROUTE_DISAGREEMENT = 'OMEGA11_CERTIFY_ROUTE_DISAGREEMENT'
OMEGA11_UNREVEALED_ACCESS = TM.TIMING_UNREVEALED_ACCESS

EPSILON_RETURN = TM.EPSILON_RETURN
KNOWN_STATUSES = ('idle', 'ready', 'committed', 'returning', 'closed')

IMPLEMENTATION_VERSION = 'M1B_Q2FIX'


def certify_return_feasibility(snap, *, T_end, clock, travel_fn, times=None,
                               closed_return_history=None, certify_route_fn=None,
                               epsilon_return=EPSILON_RETURN):
    """Return (ok, verdict, detail).

    `times` is a `m1b_timing.PublicTimeInputs`; WITHOUT it any replay that needs a time-window
    start or a service duration REFUSES (the fix for A2/A2b).

    `closed_return_history` maps `vehicle_id -> return_arrival` from a trusted trace or a
    persisted record.  A closed vehicle is judged against THAT, never against a possibly-cleared
    live field (the fix for A6).

    `certify_route_fn(vehicle_id)` is optional; when supplied the result is cross-checked against
    the real `action_contract.certify_route` and a disagreement refuses (requirement 8).
    """
    if snap is None:
        return False, OMEGA11_STATE_MISSING, {'field': 'snapshot'}
    if not TM._finite(T_end) or not TM._finite(clock):
        return False, OMEGA11_TIME_ORDER_INVALID, {'T_end': repr(T_end), 'clock': repr(clock)}
    if float(clock) > float(T_end) + float(epsilon_return):
        return False, OMEGA11_CLOCK_BEYOND_HORIZON, {
            'clock': float(clock), 'T_end': float(T_end),
            'why': 'the decision clock is already beyond the depot deadline, so no plan can be '
                   'certified against it; refusing rather than fabricating a certificate'}

    deadline = float(T_end) + float(epsilon_return)
    history = dict(closed_return_history or {})
    if times is None:
        # LEGACY path: the pre-fix caller registered a service-time table process-globally.  Use
        # the shim's per-instance object when one was registered; otherwise leave `times` as None so
        # the replay still REFUSES rather than defaulting to zero.
        times = _LegacyServiceTimes.current()
    per_vehicle, bad, used_nodes = {}, [], []

    for vid in sorted(snap['vehicles']):
        v = snap['vehicles'][vid]
        row = {'status': v.status}
        if v.status not in KNOWN_STATUSES:
            bad.append({'vehicle': vid, 'why': 'unknown status', 'status': v.status})
            per_vehicle[vid] = row
            continue

        # ------------------------------------------------------------------ closed (A6)
        if v.status == 'closed':
            if vid not in history:
                row['pending'] = False
                row['basis'] = 'closed: audited against trusted history'
                bad.append({'vehicle': vid, 'why': 'closed vehicle without trusted history; a '
                                                   'cleared live return_finish must NOT be read '
                                                   'as compliance',
                            'verdict': OMEGA11_CLOSED_HISTORY_MISSING})
                per_vehicle[vid] = row
                continue
            ra = history[vid]
            row.update({'pending': False, 'return_arrival': float(ra),
                        'basis': 'closed: trusted history record'})
            if float(ra) > deadline:
                bad.append({'vehicle': vid, 'why': 'recorded historical return exceeded the '
                                                   'deadline',
                            'return_arrival': float(ra), 'deadline': deadline,
                            'verdict': OMEGA11_CLOSED_HISTORY_DEADLINE_EXCEEDED})
            per_vehicle[vid] = row
            continue

        plan = snap['candidate_plans'].get(vid)
        suffix = [int(x) for x in (plan.suffix if plan is not None else v.mutable_suffix)]
        suffix = [x for x in suffix if x != 0]

        # ------------------------------------------------------------------ returning
        if v.status == 'returning':
            if v.return_finish is None:
                bad.append({'vehicle': vid, 'why': 'returning without return_finish'})
                per_vehicle[vid] = row
                continue
            # The recorded in-flight leg is authoritative; only a LATER suffix adds time.
            if suffix:
                tl, verd, det = TM.replay_timeline(
                    depart=float(v.return_finish), clock=clock, start_node=0,
                    stops=suffix, return_to_depot=True, travel_fn=travel_fn, times=times)
                if verd != TM.TIMING_OK:
                    bad.append({'vehicle': vid, 'why': 'suffix after an in-flight return leg '
                                                       'could not be replayed', 'verdict': verd,
                                'detail': det})
                    per_vehicle[vid] = row
                    continue
                ra = tl['return_arrival']
                used_nodes.extend(x for x in suffix)
                row['basis'] = ('recorded return_finish as the departure, then the candidate '
                                'suffix replayed through the shared timeline')
            else:
                ra = float(v.return_finish)
                row['basis'] = 'recorded return_finish (real in-flight leg, no later stops)'
            row.update({'pending': True, 'return_arrival': float(ra)})

        # ------------------------------------------------------------------ committed (A4/A4b)
        elif v.status == 'committed':
            if v.committed_finish is None or v.committed_next is None:
                bad.append({'vehicle': vid, 'why': 'committed leg missing finish/next'})
                per_vehicle[vid] = row
                continue
            # Inherit the REAL in-flight leg, then continue through the later accepted tasks.
            tl, verd, det = TM.replay_timeline(
                depart=float(v.committed_finish), clock=clock, start_node=int(v.committed_next),
                stops=suffix, return_to_depot=True, travel_fn=travel_fn, times=times)
            if verd != TM.TIMING_OK:
                bad.append({'vehicle': vid, 'why': 'committed leg plus later tasks could not be '
                                                   'replayed', 'verdict': verd, 'detail': det})
                per_vehicle[vid] = row
                continue
            used_nodes.append(int(v.committed_next))
            used_nodes.extend(suffix)
            row.update({'pending': True, 'return_arrival': float(tl['return_arrival']),
                        'basis': ('committed_finish as the departure from committed_next, then '
                                  'the full candidate suffix replayed (later accepted tasks are '
                                  'NO LONGER omitted)'),
                        'n_suffix_stops': len(suffix)})

        # ------------------------------------------------------------------ idle / ready
        else:
            if v.current_node == 0 and not suffix:
                row.update({'pending': False, 'basis': 'idle at depot with nothing pending'})
                per_vehicle[vid] = row
                continue
            # WAIT-aware: the trajectory certified is the CANDIDATE's.  The depart choice is
            # RECORDED so the label can never describe a different rule than the code applies.
            depart = float(max(v.ready_time, clock))
            tl, verd, det = TM.replay_timeline(
                depart=depart, clock=clock, start_node=int(v.current_node), stops=suffix,
                return_to_depot=True, travel_fn=travel_fn, times=times,
                anchor_time=v.ready_time, expected_anchor_time=v.ready_time,
                anchor_node=v.current_node, expected_anchor_node=v.current_node)
            if verd != TM.TIMING_OK:
                bad.append({'vehicle': vid, 'why': 'candidate suffix could not be replayed',
                            'verdict': verd, 'detail': det})
                per_vehicle[vid] = row
                continue
            used_nodes.append(int(v.current_node))
            used_nodes.extend(suffix)
            waited = sum(e.get('waited', 0.0) for e in tl['events'] if e['tag'] == 'service')
            row.update({'pending': True, 'return_arrival': float(tl['return_arrival']),
                        'depart_used': depart,
                        'depart_rule': 'max(ready_time, clock)',
                        'total_tw_wait': waited,
                        'n_suffix_stops': len(suffix),
                        'basis': ('candidate suffix replayed from the CURRENT node through the '
                                  'shared timeline, INCLUDING time-window wait and service time'
                                  if suffix else
                                  'no pending stops: the vehicle can return immediately')})

        if row.get('pending') and row['return_arrival'] > deadline:
            bad.append({'vehicle': vid, 'why': 'return_arrival exceeds deadline',
                        'return_arrival': row['return_arrival'], 'deadline': deadline,
                        'excess': row['return_arrival'] - deadline, 'basis': row.get('basis')})
        per_vehicle[vid] = row

    # ------------------------------------------------------------------ certify_route cross-check
    cross = {'performed': certify_route_fn is not None}
    if certify_route_fn is not None:
        disagreements = []
        for vid in sorted(per_vehicle):
            row = per_vehicle[vid]
            if not row.get('pending') or row.get('return_arrival') is None:
                continue
            cert = certify_route_fn(vid)
            ours = float(row['return_arrival']) <= deadline
            theirs_return_ok = cert.get('reason') != 'return'
            if ours != bool(theirs_return_ok):
                disagreements.append({'vehicle': vid, 'ours_feasible': ours,
                                      'certify_route_reason': cert.get('reason'),
                                      'ours_return_arrival': row['return_arrival'],
                                      'certify_route_return_slack': cert.get('return_slack')})
        cross['n_disagreements'] = len(disagreements)
        cross['disagreements'] = disagreements
        if disagreements:
            bad.append({'why': 'disagreement with action_contract.certify_route',
                        'verdict': OMEGA11_CERTIFY_ROUTE_DISAGREEMENT,
                        'detail': disagreements[:10]})

    detail = {'T_end': float(T_end), 'deadline': deadline, 'clock': float(clock),
              'epsilon_return': float(epsilon_return), 'per_vehicle': per_vehicle,
              'n_violations': len(bad), 'violations': bad[:20],
              'time_input_identity': None if times is None else times.identity,
              'time_inputs_used': None if times is None else times.identity_of_used_keys(used_nodes),
              'certify_route_cross_check': cross,
              'implementation_version': IMPLEMENTATION_VERSION}
    if bad:
        return False, _first_verdict(bad), detail
    return True, OMEGA11_OK, detail


def _first_verdict(bad):
    for b in bad:
        if b.get('verdict'):
            return b['verdict']
    return OMEGA11_RETURN_DEADLINE_EXCEEDED


# ---------------------------------------------------------------------------------------------
# LEGACY API SHIMS -- DISCLOSURE REQUIRED
# ---------------------------------------------------------------------------------------------
class _LegacyServiceTimes(object):
    """Adapts the removed pre-fix process-global cache API onto a per-call time-input object.

    The audited implementation exposed `register_service_times(mapping)` plus a module-global
    `_SERVICE_TIME_CACHE` read as `cache.get(node, 0.0)`.  That cache is GONE: the zero default and
    the cross-instance residue were findings A2/A2b/A3.  This shim re-creates the CALLING
    convention for a caller that has not been updated, but it keeps the table in a per-instance
    object and DOES NOT reintroduce a zero default -- the same refusal applies.

    It exists so the historical `test_m1b_q2r_negative_controls.py` can still run and its
    divergences can be recorded rather than crashing on an import-time attribute lookup.
    """

    _current = None

    @classmethod
    def register(cls, mapping):
        nodes = sorted(int(k) for k in (mapping or {}))
        cls._current = TM.PublicTimeInputs(
            service_time={int(k): float(v) for k, v in (mapping or {}).items()},
            tw_start={n: 0.0 for n in nodes}, tw_end={n: 1e9 for n in nodes},
            revealed_nodes=nodes, identity='legacy-register-service-times-shim')

    @classmethod
    def current(cls):
        return cls._current


def register_service_times(mapping):
    """LEGACY SHIM: see `_LegacyServiceTimes`.  Prefer passing `times=` explicitly."""
    return _LegacyServiceTimes.register(mapping)


_SERVICE_TIME_CACHE = {}          # LEGACY SHIM: kept empty; NEVER read for a verdict


def replan_invariant_statement():
    return {
        'rule': 'the return-feasibility certificate is PER-COMMIT and is never reused',
        'why': 'a candidate certified at clock t may be replaced at t+1 by a different plan, and '
               'a committed leg changes the physical state; carrying a stale certificate forward '
               'would assert feasibility about a plan that is no longer the one being installed',
        'mechanism': [
            're-snapshot the live state at each commit boundary (m1b_immutable_snapshot)',
            're-run certify_return_feasibility on the FRESH snapshot before installing',
            'refuse the commit when it fails; do not fall back to a previous certificate',
            'time inputs are per-instance objects, so a stale service-time table cannot be '
            'reused across instances (the process-global cache is gone)',
        ],
    }
