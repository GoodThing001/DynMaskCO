#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix: the SINGLE, SHARED time replay used by BOTH the Omega11 return gate and the
thermal gate.

Why this module exists (Q2-R findings A1, A2, A2b, A4b):
  * the old Omega11 advanced with `t = max(ready_time, clock)` and NEVER applied
    `service_start = max(arrival, tw_start)`, so it omitted the time-window WAIT that
    `action_contract.certify_route:187` does include.  A1/A2b are the counterexamples.
  * the old Omega11 read service durations from a PROCESS-GLOBAL cache that DEFAULTS TO ZERO
    (`_SERVICE_TIME_CACHE.get(node, 0.0)`), so an unregistered service time silently SHORTENED
    the replay.  A2b is the counterexample.  A zero default can only produce optimistic
    verdicts.
  * the old committed branch computed only `committed_finish + travel(committed_next -> depot)`
    and ignored the vehicle's later accepted tasks.  A4b is the counterexample.
  * the old thermal gate had its own, different time advance, so the two gates could disagree
    about the same trajectory.

This module fixes all four by providing ONE replay whose semantics MATCH `certify_route`:

    depart         = max(anchor_time, clock)
    arrival        = depart + travel(u, v)
    service_start  = max(arrival, tw_start[v])
    service_finish = service_start + service_time[v]
    return_arrival = final_service_finish + travel(last, depot)

MISSING TIME INPUT IS A REFUSAL, NEVER A ZERO.  A legitimate zero is expressed by the key
being PRESENT with value 0.0, which is different from the key being absent.
"""
from __future__ import annotations

TIMING_OK = 'TIMING_OK'
TIMING_TIME_INPUT_UNAVAILABLE = 'OMEGA11_TIME_INPUT_UNAVAILABLE'
TIMING_ANCHOR_MISMATCH = 'OMEGA11_ANCHOR_MISMATCH'
TIMING_CLOCK_BEYOND_HORIZON = 'OMEGA11_CLOCK_BEYOND_HORIZON'
TIMING_TIME_ORDER_INVALID = 'OMEGA11_TIME_ORDER_INVALID'
TIMING_UNREVEALED_ACCESS = 'TIMING_UNREVEALED_NODE_ACCESS'
TIMING_NUMERIC_INVALID = 'TIMING_NUMERIC_INVALID'
TIMING_RETURN_DEADLINE_EXCEEDED = 'OMEGA11_RETURN_DEADLINE_EXCEEDED'

EPSILON_RETURN = 1e-6


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


class PublicTimeInputs(object):
    """Per-INSTANCE, identity-carrying time inputs.  No process-global state.

    `service_time` and `tw_start` must be supplied as explicit mappings.  A node that is part of
    the replayed trajectory but absent from EITHER mapping makes the replay REFUSE -- the fix for
    A2/A2b.  A present key with value 0.0 is legitimate and accepted.

    `revealed_nodes` bounds what may be read at all: querying an unrevealed node raises, so a
    caller cannot use this object to launder future information into the replay.
    """

    __slots__ = ('service_time', 'tw_start', 'tw_end', 'revealed_nodes', 'identity')

    def __init__(self, *, service_time, tw_start, tw_end, revealed_nodes, identity):
        if service_time is None or tw_start is None or tw_end is None:
            raise ValueError('service_time / tw_start / tw_end are all required')
        if revealed_nodes is None:
            raise ValueError('revealed_nodes is required (it bounds what may be read)')
        self.service_time = {int(k): float(v) for k, v in service_time.items()}
        self.tw_start = {int(k): float(v) for k, v in tw_start.items()}
        self.tw_end = {int(k): float(v) for k, v in tw_end.items()}
        self.revealed_nodes = frozenset(int(x) for x in revealed_nodes)
        self.identity = str(identity)

    def _check_visible(self, node):
        if int(node) in self.revealed_nodes:
            return
        raise PermissionError(TIMING_UNREVEALED_ACCESS + ':%d' % int(node))

    def service_time_of(self, node):
        self._check_visible(node)
        if int(node) not in self.service_time:
            raise KeyError(TIMING_TIME_INPUT_UNAVAILABLE + ':service_time:%d' % int(node))
        v = self.service_time[int(node)]
        if not _finite(v) or v < 0:
            raise ValueError(TIMING_NUMERIC_INVALID + ':service_time:%d' % int(node))
        return v

    def tw_start_of(self, node):
        self._check_visible(node)
        if int(node) not in self.tw_start:
            raise KeyError(TIMING_TIME_INPUT_UNAVAILABLE + ':tw_start:%d' % int(node))
        v = self.tw_start[int(node)]
        if not _finite(v):
            raise ValueError(TIMING_NUMERIC_INVALID + ':tw_start:%d' % int(node))
        return v

    def tw_end_of(self, node):
        self._check_visible(node)
        if int(node) not in self.tw_end:
            raise KeyError(TIMING_TIME_INPUT_UNAVAILABLE + ':tw_end:%d' % int(node))
        v = self.tw_end[int(node)]
        if not _finite(v):
            raise ValueError(TIMING_NUMERIC_INVALID + ':tw_end:%d' % int(node))
        return v

    def identity_of_used_keys(self, nodes):
        """A stable identity of exactly the inputs the replay consumed (for the receipt)."""
        used = sorted(int(n) for n in nodes)
        return {'time_inputs_identity': self.identity,
                'n_nodes': len(used),
                'nodes': used,
                'service_time': [self.service_time.get(n) for n in used],
                'tw_start': [self.tw_start.get(n) for n in used]}


def replay_timeline(*, depart, clock, start_node, stops, return_to_depot, travel_fn, times,
                    anchor_time=None, anchor_node=None, anchor_load=None,
                    expected_anchor_time=None, expected_anchor_node=None,
                    expected_anchor_load=None):
    """Replay a trajectory and return (timeline, verdict, detail).

    `stops` is the ordered list of customer nodes to visit.  `times` is a `PublicTimeInputs`.

    ANCHOR VALIDATION (requirement 1): when the expected anchor values are supplied they must
    match, otherwise the replay refuses rather than certifying a different trajectory than the
    one the receipt claims.
    """
    if times is None:
        return None, TIMING_TIME_INPUT_UNAVAILABLE, {'field': 'times'}
    for nm, got, want in (('anchor_time', anchor_time, expected_anchor_time),
                          ('anchor_node', anchor_node, expected_anchor_node),
                          ('anchor_load', anchor_load, expected_anchor_load)):
        if want is None:
            continue
        if got is None:
            return None, TIMING_ANCHOR_MISMATCH, {'field': nm, 'expected': want, 'got': None}
        if nm == 'anchor_load':
            if abs(float(got) - float(want)) > 1e-9:
                return None, TIMING_ANCHOR_MISMATCH, {'field': nm, 'expected': want, 'got': got}
        elif nm == 'anchor_node':
            if int(got) != int(want):
                return None, TIMING_ANCHOR_MISMATCH, {'field': nm, 'expected': want, 'got': got}
        else:
            if abs(float(got) - float(want)) > 1e-9:
                return None, TIMING_ANCHOR_MISMATCH, {'field': nm, 'expected': want, 'got': got}
    if not _finite(depart) or not _finite(clock):
        return None, TIMING_NUMERIC_INVALID, {'depart': repr(depart), 'clock': repr(clock)}
    t = float(max(float(depart), float(clock)))
    node = int(start_node)
    events = [{'tag': 'start', 'time': t, 'node': node}]
    try:
        for nxt in stops:
            nxt = int(nxt)
            if nxt == 0:
                return None, TIMING_TIME_ORDER_INVALID, {'why': 'depot id inside the stop list'}
            leg = float(travel_fn(node, nxt))
            if not _finite(leg) or leg < 0:
                return None, TIMING_TIME_ORDER_INVALID, {'leg': [node, nxt], 'travel': leg}
            arrive = t + leg
            if not _finite(arrive) or arrive < t - 1e-9:
                return None, TIMING_TIME_ORDER_INVALID, {'from': node, 'to': nxt,
                                                         'depart': t, 'arrive': arrive}
            # THE fix for A1: the customer time-window wait is applied here, exactly as
            # action_contract.certify_route:187 does.
            start = max(arrive, times.tw_start_of(nxt))
            svc = times.service_time_of(nxt)
            finish = start + svc
            if not _finite(finish) or finish < start - 1e-9:
                return None, TIMING_TIME_ORDER_INVALID, {'node': nxt, 'start': start,
                                                         'finish': finish}
            events.append({'tag': 'service', 'node': nxt, 'depart': t, 'arrive': arrive,
                           'service_start': start, 'service_finish': finish,
                           'waited': start - arrive, 'service_time': svc})
            t, node = finish, nxt
        return_arrival = None
        if return_to_depot:
            leg = float(travel_fn(node, 0))
            if not _finite(leg) or leg < 0:
                return None, TIMING_TIME_ORDER_INVALID, {'leg': [node, 0], 'travel': leg}
            return_arrival = t + leg
            events.append({'tag': 'return', 'depart': t, 'arrive': return_arrival})
    except PermissionError as exc:
        return None, TIMING_UNREVEALED_ACCESS, {'exception': str(exc)[:160]}
    except KeyError as exc:
        return None, TIMING_TIME_INPUT_UNAVAILABLE, {'exception': str(exc)[:160]}
    except ValueError as exc:
        return None, TIMING_NUMERIC_INVALID, {'exception': str(exc)[:160]}

    timeline = {'start_time': float(max(float(depart), float(clock))), 'events': events,
                'n_stops': len(list(stops)), 'return_arrival': return_arrival,
                'final_node': node}
    return timeline, TIMING_OK, {}


def certify_return_deadline(timeline, *, T_end, epsilon_return=EPSILON_RETURN):
    """(ok, verdict, detail) for `return_arrival <= T_end + epsilon_return`."""
    if timeline is None:
        return False, TIMING_TIME_INPUT_UNAVAILABLE, {}
    ra = timeline.get('return_arrival')
    if ra is None:
        return False, TIMING_TIME_ORDER_INVALID, {'why': 'no return leg replayed'}
    deadline = float(T_end) + float(epsilon_return)
    if float(ra) > deadline:
        return False, TIMING_RETURN_DEADLINE_EXCEEDED, {
            'return_arrival': float(ra), 'deadline': deadline,
            'excess': float(ra) - deadline, 'T_end': float(T_end)}
    return True, TIMING_OK, {'return_arrival': float(ra), 'deadline': deadline}


def compare_with_certify_route(*, timeline, certify_route_fn, T_end, epsilon_return=EPSILON_RETURN,
                               epsilon_agree=1e-6):
    """Independent cross-check against the real `action_contract.certify_route` (requirement 8).

    `certify_route_fn()` must return the certify_route dict.  The two verdicts MUST agree; a
    disagreement refuses rather than silently preferring one.
    """
    cert = certify_route_fn()
    ours_feasible = bool(timeline is not None and timeline.get('return_arrival') is not None
                         and timeline['return_arrival'] <= float(T_end) + float(epsilon_return))
    theirs_feasible = bool(cert.get('feasible'))
    theirs_return_ok = cert.get('reason') not in ('return',)
    agree = (ours_feasible == (theirs_feasible or theirs_return_ok))
    detail = {'ours_feasible': ours_feasible, 'certify_route_feasible': theirs_feasible,
              'certify_route_reason': cert.get('reason'),
              'certify_route_return_slack': cert.get('return_slack'),
              'ours_return_arrival': (timeline or {}).get('return_arrival')}
    if not agree:
        return False, 'OMEGA11_CERTIFY_ROUTE_DISAGREEMENT', detail
    return True, TIMING_OK, detail
