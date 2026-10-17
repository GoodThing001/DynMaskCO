#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 S3 (bitwise executor replay) qualification.

THE CLAIM S3 IS ALLOWED TO MAKE
===============================
    FLOAT_EXECUTOR_REPLAY_CONDITIONAL_ON_IDENTITY

Given the SAME frozen executor code, the SAME interpreter/libm, the SAME contract, the SAME start
state, the SAME per-call inputs AND **the same call sequence** (partition + order), re-running
`transition_segment` reproduces the stored floats bit for bit.  Nothing more.

WHY S3 CANNOT CERTIFY A FUTURE SUFFIX UNDER STRICT-ONLINE
=========================================================
The environment's advance is partitioned at EVENT CLOCKS
(`strict_online_env.py:488-510, 209-230, 318-331`): between two doors the door-less advance is cut at
every reveal, booking-horizon tick and foreign service/return completion.  Splitting is **not**
neutral: refining a door-less interval raises its endpoint by `c*b*(1 - exp(-h*a))` (lemma below), so
the state after the next door depends on the partition, and the partition depends on the future reveal
clock sequence, which strict-online forbids reading.  Therefore:

  * a replay reproduces the executor only on a partition that is already **fixed** (past events);
  * for a future suffix the partition is **not determined by the plan**, so "bitwise parity" is not a
    property that can be certified in advance;
  * and re-running the same implementation twice is **not** an independent proof of anything -- it is
    one implementation agreeing with itself.

WHAT REPLACES IT: A PARTITION-INDEPENDENT BRACKET
=================================================
Lemma BR-1 (refinement raises):   Phi_a(Phi_b(T)) - Phi_{a+b}(T) = c*b*(1 - exp(-h*a)) >= 0.
Lemma BR-2 (flow dominates a step): Phi_d(T) <= F_d(T)  (proved in the event-segmentation module).
Hence for ANY partition of one door-less interval of total `D`:

    Phi_D(T)  <=  composition(T)  <=  F_D(T)                       (bracket, partition-independent)
    max(tau, Phi_D(T)) <= floored composition <= max(tau, F_D(T))

so the *response set* of the executor over all admissible partitions is an interval whose two ends are
computed without knowing the partition at all.  The upper end is exactly the ODE flow used by P5, which
is why P5 needs no event-partition knowledge; the lower end is the unsplit replay.  The width of the
bracket is the honest "we do not know how the future will cut this interval" uncertainty.

This module implements the bracket, the precondition record and a POST-HOC parity audit (replay the
recorded partition and compare against the recorded floats).  It is not wired into any certifier and
cannot emit a production receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_FLOAT_REPLAY_QUALIFICATION_V1'
SCHEMA = 'q3_float_replay_qualification'

REPLAY_QUALIFIED = 'FLOAT_EXECUTOR_REPLAY_CONDITIONAL_ON_IDENTITY'
REPLAY_REFUSED = 'FLOAT_EXECUTOR_REPLAY_REFUSED'
REPLAY_PARITY_FAILED = 'FLOAT_EXECUTOR_REPLAY_PARITY_FAILED'

#: every precondition of a bitwise-parity claim.  `verifiable` says whether this module can check it
#: from the repository; `record_only` items can only be recorded in the identity, never proven.
PARITY_PRECONDITIONS = (
    ('P1', 'executor source bytes', 'verifiable',
     'sha256 of scripts/coldchain/coldchain_state.py', 'recomputed and compared'),
    ('P2', 'interpreter + libm', 'record_only',
     'sys.version / platform / math module identity', 'recorded; portability NOT provable'),
    ('P3', 'thermal contract', 'verifiable', 'contract.contract_hash', 'compared against the claim'),
    ('P4', 'start state', 'verifiable', 'digest of compartment temperatures + manifest + counters',
     'compared against the claim'),
    ('P5', 'per-call inputs', 'verifiable',
     'digest of (depart, arrival, service_finish, served_customer, temp_class, quantity, '
     'initial_quality, distance, active_zone_mask)', 'compared against the claim'),
    ('P6', 'call sequence / partition', 'verifiable_after_the_fact',
     'digest of the ordered call list', 'compared; NOT knowable in advance (see module docstring)'),
    ('P7', 'call order and count', 'verifiable', 'len(calls) + ordered served_customer list',
     'compared against the claim'),
)


def _digest(obj):
    raw = json.dumps(obj, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def code_identity(paths):
    """sha256 of every frozen module in the executor call path, so a drift is detectable."""
    out = {}
    for p in paths:
        out[str(p)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def state_digest(coldchain_state):
    """Malformed-safe digest of a `VehicleColdChainState` (read-only; never mutates)."""
    if coldchain_state is None:
        return None
    return _digest({
        'temperatures': [repr(float(x)) for x in
                         getattr(coldchain_state, 'compartment_temperature_c', ())],
        'zone_load': [repr(float(x)) for x in getattr(coldchain_state, 'zone_load', ())],
        'manifest': [[int(getattr(l, 'order_id', -1)), repr(float(getattr(l, 'quantity', 0.0))),
                      repr(float(getattr(l, 'quality_remaining', 0.0))),
                      int(getattr(l, 'temp_class', -1))]
                     for l in (getattr(coldchain_state, 'cargo_manifest', ()) or ())],
        'door_open_count': int(getattr(coldchain_state, 'door_open_count', 0)),
        'closed': bool(getattr(coldchain_state, 'closed', False)),
        'violation_count': int(getattr(coldchain_state, 'thermal_violation_count', 0)),
    })


def call_digest(depart, arrival, service_finish, served_customer, order_temp_class,
                order_quantity, initial_quality, segment_distance_units, active_zone_mask):
    return _digest({'depart': repr(float(depart)), 'arrival': repr(float(arrival)),
                    'finish': repr(float(service_finish)),
                    'served': None if served_customer is None else int(served_customer),
                    'temp_class': None if order_temp_class is None else int(order_temp_class),
                    'quantity': None if order_quantity is None else repr(float(order_quantity)),
                    'initial_quality': None if initial_quality is None else
                    repr(float(initial_quality)),
                    'distance': None if segment_distance_units is None else
                    repr(float(segment_distance_units)),
                    'mask': [bool(x) for x in (active_zone_mask or ())]})


def partition_digest(calls):
    """Digest of the ORDERED call sequence: this is the object S3 must pin (P6/P7)."""
    seq = []
    for c in calls:
        seq.append(call_digest(c['depart'], c['arrival'], c['finish'], c.get('served'),
                               c.get('temp_class'), c.get('quantity'), c.get('initial_quality'),
                               c.get('distance'), c.get('mask')))
    return _digest(seq), len(seq)


# ---------------------------------------------------------------------------------------------
# the partition-independent bracket
# ---------------------------------------------------------------------------------------------
def t_infinity(z):
    return z.ambient_c - z.cooling_rate_c_per_hour / z.heat_transfer_per_hour


def flow(z, t0, total_h):
    ti = t_infinity(z)
    return ti + (float(t0) - ti) * math.exp(-z.heat_transfer_per_hour * float(total_h))


def unsplit_raw(z, t0, total_h):
    """The COARSEST partition (one piece), WITHOUT the max(tau, .) floor.

    Keeping the floor out is essential: the lower end of the bracket must be floored **after** the
    comparison, otherwise a floored `low` can sit above an unfloored `high` and the "bracket" is
    inverted (this bit the first version of REP-04 and is recorded there).
    """
    return (z.ambient_c + (float(t0) - z.ambient_c)
            * math.exp(-z.heat_transfer_per_hour * float(total_h))
            - z.cooling_rate_c_per_hour * float(total_h))


def unsplit(contract, z, t0, total_h):
    """Executor value of ONE piece (floored), for direct comparison with `_advance_temperatures`."""
    import coldchain_state as cs
    temps = [float(t) for t in contract.thermal.target_temperature_c]
    temps[z.zone] = float(t0)
    temps, _e = cs._advance_temperatures(tuple(temps), float(total_h), (True,) * len(temps),
                                         contract)
    return temps[z.zone]


def bracket(contract, z, t0, total_h):
    """`[max(tau, unsplit_raw), max(tau, flow)]`: every admissible FLOORED partition lands inside.

    lower: floored composition >= max(tau, raw composition) >= max(tau, unsplit_raw)   (BR-1)
    upper: floored composition <= max(tau, flow)                                      (BR-2)
    """
    lo_raw = unsplit_raw(z, t0, total_h)
    hi_raw = flow(z, t0, total_h)
    lo = max(z.target_c, lo_raw)
    hi = max(z.target_c, hi_raw)
    return {'zone': z.zone, 't0': float(t0), 'total_h': float(total_h),
            'lower_raw_c': lo_raw, 'upper_raw_c': hi_raw,
            'lower_c': lo, 'upper_c': hi, 'width_c': hi - lo,
            'tau': z.target_c, 'partition_independent': True,
            'single_piece_floored_c': unsplit(contract, z, t0, total_h)}


def bracket_series(contract, z, entry_state, intervals):
    """Propagate the bracket across a list of door-less intervals.

    Both ends are monotone in the entry state, so propagating the two ends separately gives a valid
    outer bracket for the whole series -- no matter how each interval is partitioned.
    """
    rows = []
    lo = hi = float(entry_state)
    for i, total in enumerate(intervals):
        lo_next = max(z.target_c, unsplit_raw(z, lo, total))
        hi_next = max(z.target_c, flow(z, hi, total))
        lo, hi = lo_next, hi_next
        rows.append({'index': i, 'total_h': float(total), 'lower_c': lo, 'upper_c': hi})
    return {'zone': z.zone, 'entry_state': float(entry_state), 'rows': rows,
            'lower_c': lo, 'upper_c': hi}


# ---------------------------------------------------------------------------------------------
# the qualification record
# ---------------------------------------------------------------------------------------------
def qualify_replay(*, contract, contract_identity, executor_paths, claimed_code_identity,
                   claimed_contract_identity, start_state, claimed_state_digest,
                   calls, claimed_partition_digest=None,
                   runtime_record=None, replay_results=None, recorded_results=None):
    """Assemble the S3 qualification record and refuse anything that is not pinned.

    `replay_results` / `recorded_results` are optional per-call temperature tuples; when both are
    supplied the module performs the POST-HOC parity check (this is the only direction in which S3
    can produce evidence: after the partition is known).
    """
    refusals, records = [], {}
    actual_code = code_identity(executor_paths) if executor_paths else {}
    records['code_identity'] = actual_code
    if not executor_paths:
        refusals.append('EXECUTOR_PATHS_NOT_SUPPLIED')
    elif claimed_code_identity is not None and dict(claimed_code_identity) != actual_code:
        refusals.append('EXECUTOR_CODE_DRIFT')
    try:
        cid = SF.require_contract_identity(contract, contract_identity)
    except SF.Refusal as exc:
        cid = None
        refusals.append(exc.code)
    records['contract_identity'] = cid
    if claimed_contract_identity is not None and cid != claimed_contract_identity:
        refusals.append('CONTRACT_IDENTITY_DRIFT')
    sd = state_digest(start_state)
    records['start_state_digest'] = sd
    if claimed_state_digest is None:
        refusals.append('STATE_DIGEST_NOT_SUPPLIED')
    elif sd != claimed_state_digest:
        refusals.append('START_STATE_DRIFT')
    if not calls:
        refusals.append('NO_CALLS')
        pd, n = None, 0
    else:
        pd, n = partition_digest(calls)
    records['partition_digest'] = pd
    records['n_calls'] = n
    if claimed_partition_digest is not None and pd != claimed_partition_digest:
        refusals.append('PARTITION_DRIFT')
    records['runtime'] = runtime_record or {'note': 'not supplied; P2 can only be recorded'}
    if runtime_record is None:
        refusals.append('RUNTIME_IDENTITY_NOT_RECORDED')

    parity = None
    if replay_results is not None and recorded_results is not None:
        if len(replay_results) != len(recorded_results):
            refusals.append('PARITY_LENGTH_MISMATCH')
            parity = False
        else:
            worst = 0.0
            for a, b in zip(replay_results, recorded_results):
                for x, y in zip(a, b):
                    worst = max(worst, abs(float(x) - float(y)))
            parity = (worst == 0.0)
            records['parity_max_abs_diff_c'] = worst
            if not parity:
                refusals.append('BITWISE_PARITY_FAILED')

    verdict = REPLAY_REFUSED if refusals else (
        REPLAY_PARITY_FAILED if parity is False else REPLAY_QUALIFIED)
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'verdict': verdict,
        'refusals': refusals,
        'records': records,
        'preconditions': [{'id': a, 'name': b, 'status': c, 'object': d, 'how': e}
                          for a, b, c, d, e in PARITY_PRECONDITIONS],
        'claim': ("re-running the frozen executor on the SAME state and the SAME ordered call "
                  "sequence reproduces the stored floats bit for bit"),
        'not_claimed': [
            'continuous / intra-segment temperature safety -- the executor defines no intra-segment '
            'trajectory at all, so a replay says nothing about it',
            'any FUTURE suffix: the call sequence is partitioned by future event clocks, which '
            'strict-online forbids reading (see the module docstring and SEG-05)',
            'independence of the check: two calls into the same implementation are one implementation '
            'agreeing with itself, not two proofs',
            'portability: P2 (interpreter + libm) is recorded, never proven',
            'that the stored floats are within the hard bounds -- that still needs the bracket or P5'],
        'replacement_when_the_partition_is_unknown': (
            'bracket [unsplit, flow] per interval: partition-independent outer bounds, with the upper '
            'end equal to the ODE flow used by P5'),
    }


def post_hoc_audit(*, contract, calls, recorded_states, replay_fn):
    """Replay the RECORDED call sequence and compare with the recorded floats (audit, not prophecy).

    `replay_fn(call, state) -> state` must be the frozen executor wrapper supplied by the caller.
    Returns (ok, detail).  A single mismatched float is a FAILURE: it means either the recorded state
    did not come from this code path, or the code/runtime drifted.
    """
    worst, worst_at = 0.0, None
    if len(calls) != len(recorded_states):
        return False, {'why': 'call/state length mismatch', 'n_calls': len(calls),
                       'n_states': len(recorded_states)}
    for i, (call, recorded) in enumerate(zip(calls, recorded_states)):
        got = replay_fn(call)
        for z, (a, b) in enumerate(zip(got, recorded)):
            d = abs(float(a) - float(b))
            if d > worst:
                worst, worst_at = d, {'index': i, 'zone': z, 'replay': float(a),
                                      'recorded': float(b)}
    return (worst == 0.0), {'max_abs_diff_c': worst, 'worst_at': worst_at,
                            'n_calls': len(calls),
                            'reading': 'exact equality is required; a single differing bit means the '
                                       'recorded state did not come from this code path'}
