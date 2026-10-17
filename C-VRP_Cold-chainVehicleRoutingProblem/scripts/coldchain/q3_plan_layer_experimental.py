#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 PLAN-LAYER reachability checker (read-only, non-production).

WHAT THIS ANSWERS
=================
The refuted Theorem B claimed a fixed band [tau, tau+delta] for ANY segment sequence; the errata
replaced it with P1 (U recursion), P2 (door-count), P3 (every segment at least Delta*) and P4 (door
segments exist only at service stops).  P3 is needlessly strong: the temperature can only be RAISED
by a door event, so a segment that opens no door on zone z cannot push zone z out of the band.  This
module implements the resulting **weaker, still valid** condition and checks it on a concrete plan:

  P3' (proved in the accompanying report, section B)
      Let Delta*_z = delta_z / margin_z with margin_z = c_z - h_z*(A - tau_z)^+ > 0.
      If (i) the state at the start of the checked suffix is inside [tau_z, tau_z + delta_z], and
             (ii) every segment that OPENS zone z has duration >= Delta*_z,
      then every zone stays inside [tau_z, tau_z + delta_z] at every segment boundary.
      Segments that open no door on z are UNCONSTRAINED -- in particular arbitrarily short ones are
      harmless, because the advance map is non-increasing in t (L3) and the door is the only
      mechanism that raises a zone.

  The two branches `Delta <= _EPS` and `Delta > _EPS` are handled separately: the executor
  short-circuits `Delta <= _EPS` to the identity (`coldchain_state.py:361-362`), so a no-door
  short segment leaves the value unchanged, while a door on such a segment adds delta on top of the
  unchanged value -- which is exactly why a door segment shorter than Delta* is unsafe.

WHAT IT DELIBERATELY DOES NOT DO
================================
  * it never reads a hidden future order: a served node must be revealed at `clock`;
  * it never claims that a plan violates the hard bounds.  A failing condition means THIS
    certification route is insufficient (`NOT_CERTIFIABLE`), not that the plan is infeasible;
  * it does not use, restore or re-derive the refuted Theorem B / B1 / B2 sufficiency;
  * it is read-only: no file, no environment object and no plan is mutated;
  * it cannot emit a production receipt (`production_consumable` hard-coded False).

VERDICT LADDER (mutually exclusive, never silently promoted)
==========================================================
  REFUSED            an input is inconsistent (identity, reveal, commitment, malformed, hidden future)
  NOT_CERTIFIABLE    a door segment is shorter than Delta*_z (this route does not reach the band)
  DIAGNOSTIC_ONLY    the arithmetic condition was evaluated but a binding needed to TURN IT INTO A
                     CLAIM is missing (entry temperature unknown, commitment list absent, frozen data
                     with no live state identity)
  PROVED_CONDITIONAL every condition of P3' holds under the declared domain D1-D6 and the bindings
                     recorded in the receipt
"""
from __future__ import annotations

import math
from fractions import Fraction

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_PLAN_LAYER_EXPERIMENTAL_V1'
SCHEMA = 'q3_plan_layer_experimental'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: the propositions this checker is allowed to rely on.  Theorem B / B1 / B2s are NOT in this list.
ALLOWED_PROPOSITIONS = ('P1_U_RECURSION', 'P2_DOOR_COUNT', "P3'DOOR_SEGMENT_ONLY", 'L1_FLOOR',
                        'L3_DRIFT_NON_INCREASING', 'THEOREM_A_PER_SEGMENT')
FORBIDDEN_PROPOSITIONS = ('THEOREM_B_BAND_ANY_SEQUENCE', 'COROLLARY_B1_INTRA_SEGMENT',
                          'COROLLARY_B2_SUFFICIENCY')


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


def euclidean_travel_fn(coords):
    """Default travel model: unit-speed Euclidean distance, i.e. the time unit *is* the distance unit.

    The certificate does NOT depend on this: in `strict_service_lower_bound` mode the condition is
    `service_time >= Delta*`, which holds for every non-negative travel/wait model.  The replay mode
    uses this only to report the ACTUAL segment durations.
    """
    def _fn(a, b):
        if not (_finite(a) and _finite(b)):
            raise SF.Refusal('NON_FINITE_NODE_ID', {'a': repr(a), 'b': repr(b)})
        ia, ib = int(a), int(b)
        for n in (ia, ib):
            if n < 0 or n >= len(coords):
                raise SF.Refusal('NODE_OUT_OF_RANGE', {'node': n, 'n_nodes': len(coords)})
        ax, ay = float(coords[ia][0]), float(coords[ia][1])
        bx, by = float(coords[ib][0]), float(coords[ib][1])
        return math.hypot(ax - bx, ay - by)
    return _fn


# --------------------------------------------------------------------------------------------
# instance extraction (read-only)
# --------------------------------------------------------------------------------------------
class InstanceView(object):
    """A minimal, read-only view over ONE instance of the frozen NPZ layout.

    Only the arrays the thermal condition needs are read, and node identity is verified rather than
    assumed: node 0 must look like a depot (zero service time, zero demand), and any node that is
    SERVED with a zero (or missing) service time is a refusal, not a silent default.
    """

    REQUIRED = ('coords', 'service_time', 'temp_class', 'reveal_time')

    def __init__(self, raw, inst_idx=None):
        """`inst_idx=None` means `raw` is ALREADY one instance (node-major arrays).

        `inst_idx=i` means `raw` is the whole frozen NPZ and instance `i` is selected.  The two cases
        must be distinguished explicitly: a coords array of shape (n_nodes, 2) is indistinguishable
        from (n_inst, n_nodes) by shape alone, and guessing was exactly how the first version of this
        view silently read a single coordinate as a whole instance.
        """
        self.inst_idx = None if inst_idx is None else int(inst_idx)
        self.raw = raw
        missing = [k for k in self.REQUIRED if k not in raw]
        if missing:
            raise SF.Refusal('INSTANCE_MISSING_FIELDS', {'missing': missing})
        self.coords = self._row(raw['coords'])
        self.service_time = [float(x) for x in self._row(raw['service_time'])]
        self.temp_class = [int(x) for x in self._row(raw['temp_class'])]
        self.reveal_time = [float(x) for x in self._row(raw['reveal_time'])]
        self.demands = ([float(x) for x in self._row(raw['demands'])] if 'demands' in raw else None)
        self.tw_start = ([float(x) for x in self._row(raw['tw_start'])] if 'tw_start' in raw else None)
        self.n_nodes = len(self.service_time)
        if len(self.coords) != self.n_nodes or len(self.temp_class) != self.n_nodes:
            raise SF.Refusal('INSTANCE_ROW_LENGTH_MISMATCH',
                             {'coords': len(self.coords), 'service_time': self.n_nodes,
                              'temp_class': len(self.temp_class)})

    def _row(self, arr):
        """Select one instance ONLY when the caller said the array is the whole NPZ."""
        if self.inst_idx is None or isinstance(arr, list):
            return arr
        shape = getattr(arr, 'shape', None)
        if shape is not None and len(shape) >= 2 and self.inst_idx < shape[0]:
            return arr[self.inst_idx]
        return arr

    # -- identity ---------------------------------------------------------------------------
    def depot_profile(self):
        """Evidence that node 0 is the depot / placeholder rather than a serviceable order."""
        return {'node': 0, 'service_time': self.service_time[0],
                'temp_class': self.temp_class[0],
                'demand': None if self.demands is None else self.demands[0],
                'coord': [float(self.coords[0][0]), float(self.coords[0][1])]}

    def zero_service_nodes(self):
        return [i for i, s in enumerate(self.service_time) if s <= 0.0]

    def visible_at(self, clock):
        """Strict-online visibility: a node is revealed iff reveal_time <= clock."""
        return {i for i in range(self.n_nodes) if self.reveal_time[i] <= float(clock) + 1e-12}

    def order_nodes(self):
        """Nodes that can legitimately be SERVED: everything except node 0 (the depot)."""
        return [i for i in range(1, self.n_nodes)]


# --------------------------------------------------------------------------------------------
# the checker
# --------------------------------------------------------------------------------------------
def check_plan_layer(*, contract, contract_identity, instance, clock, plan_by_vehicle,
                     committed_order_ids=None, revealed_nodes=None, state_identity=None,
                     expected_state_identity=None, initial_temperature_by_vehicle=None,
                     ready_by_vehicle=None, start_node_by_vehicle=None, active_zone_mask=None,
                     travel_fn=None, T_end=None, mode='strict_service_lower_bound'):
    """Check proposition P3' on a concrete candidate plan.  Read-only, fail-closed, non-production.

    Parameters (all keyword-only, all explicit -- there are no defaults that could hide a leak):
      contract                    ColdChainContract(V2) whose `contract_hash` is authoritative
      contract_identity           the identity the caller CLAIMS; must equal `contract.contract_hash`
      instance                    InstanceView or the raw NPZ dict for one instance
      clock                       current strict-online clock
      plan_by_vehicle             {vehicle_id: [node_id, ...]} ordered candidate suffix per vehicle
      committed_order_ids         the accepted/committed order set; None => DIAGNOSTIC_ONLY
      revealed_nodes              explicit visible set; None => derived from `reveal_time <= clock`
      state_identity              digest of the state the plan was produced from
      expected_state_identity     digest the caller asserts; mismatch => REFUSED
      initial_temperature_by_vehicle  {vid: (T_zone0, T_zone1, ...)} entry state of the suffix
      ready_by_vehicle            {vid: ready_time}; default = clock (vehicle idle at the depot)
      start_node_by_vehicle       {vid: current node}; default = 0 (depot)
      active_zone_mask            must be all-True (domain D1); anything else is outside the domain
      travel_fn                   optional (a, b) -> travel time; default Euclidean
      T_end                       certification horizon, reported only
      mode                        'strict_service_lower_bound' (default) or 'replay'
    """
    refusals, notes = [], []
    ctx = {'mode': mode, 'clock': float(clock), 'inst_idx': getattr(instance, 'inst_idx', None)}

    # ---- 0 identity bindings --------------------------------------------------------------
    try:
        contract_hash = SF.require_contract_identity(contract, contract_identity)
    except SF.Refusal as exc:
        refusals.append({'code': exc.code, 'detail': exc.detail})
        contract_hash = None
    if state_identity is None or expected_state_identity is None:
        notes.append('STATE_IDENTITY_UNBOUND')
    elif str(state_identity) != str(expected_state_identity):
        refusals.append({'code': 'STATE_IDENTITY_MISMATCH',
                         'detail': {'supplied': str(state_identity),
                                    'expected': str(expected_state_identity)}})

    # ---- 1 domain D1: every zone active ---------------------------------------------------
    if not isinstance(instance, InstanceView):
        try:
            instance = InstanceView(instance)
        except SF.Refusal as exc:
            refusals.append({'code': exc.code, 'detail': exc.detail})
            return _receipt(REFUSED, refusals, notes, {}, ctx, contract_hash, None, None)
    zones = SF.zone_thermals_exact(contract) if contract is not None else ()
    n_zones = len(zones)
    mask = (True,) * n_zones if active_zone_mask is None else tuple(bool(x) for x in active_zone_mask)
    if len(mask) != n_zones:
        refusals.append({'code': 'ZONE_MASK_LENGTH_MISMATCH',
                         'detail': {'mask': len(mask), 'zones': n_zones}})
    elif not all(mask):
        refusals.append({'code': 'ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN',
                         'detail': {'mask': list(mask)}})

    # ---- 2 drift condition D3 + the per-zone return time ----------------------------------
    per_zone = {}
    for z in zones:
        margin = z.drift_margin_c_per_hour
        if margin <= 0:
            refusals.append({'code': 'DRIFT_CONDITION_VIOLATED',
                             'detail': {'zone': z.zone, 'margin': str(margin)}})
            per_zone[z.zone] = {'drift_margin': str(margin), 'return_time_h': None}
            continue
        per_zone[z.zone] = {'drift_margin_c_per_hour': float(margin),
                            'return_time_h': float(z.return_time_h),
                            'delta_door_c': float(z.delta_real),
                            'hard_bounds_c': [z.hard_low_c, z.hard_high_c],
                            'door_segments': [], 'n_door_segments': 0,
                            'min_door_segment_h': None, 'slack_h': None,
                            'all_door_segments_long_enough': None}

    # ---- 3 plan shape, visibility, commitment ---------------------------------------------
    if not isinstance(plan_by_vehicle, dict) or not plan_by_vehicle:
        refusals.append({'code': 'MISSING_PLAN', 'detail': {'type': type(plan_by_vehicle).__name__}})
        plan_by_vehicle = {}
    allowed = instance.order_nodes()
    served, seen = [], set()
    for vid in sorted(plan_by_vehicle):
        route = plan_by_vehicle[vid]
        if route is None:
            refusals.append({'code': 'NULL_ROUTE', 'detail': {'vehicle': str(vid)}})
            continue
        for pos, n in enumerate(route):
            if isinstance(n, bool) or not isinstance(n, (int,)) and not _finite(n):
                refusals.append({'code': 'MALFORMED_STOP', 'detail': {'vehicle': str(vid),
                                                                     'pos': pos, 'node': repr(n)}})
                continue
            n = int(n)
            if n == 0:
                refusals.append({'code': 'DEPOT_IN_STOP_LIST',
                                 'detail': {'vehicle': str(vid), 'pos': pos,
                                            'why': 'node 0 is the depot/placeholder; the executor '
                                                   'rejects it inside a stop list and it must not be '
                                                   'treated as a served order'}})
                continue
            if n not in allowed:
                refusals.append({'code': 'STOP_OUT_OF_RANGE',
                                 'detail': {'node': n, 'n_nodes': instance.n_nodes}})
                continue
            if n in seen:
                refusals.append({'code': 'ORDER_SERVED_TWICE', 'detail': {'node': n}})
            seen.add(n)
            served.append(n)
    visible = (set(int(x) for x in revealed_nodes) if revealed_nodes is not None
               else instance.visible_at(clock))
    hidden = sorted(set(served) - visible)
    if hidden:
        refusals.append({'code': 'SERVES_UNREVEALED_NODE',
                         'detail': {'nodes': hidden[:20], 'n_hidden': len(hidden),
                                    'why': 'strict-online: a served node must already be revealed at '
                                           'the clock; reading a future order is a protocol leak'}})
    if committed_order_ids is None:
        notes.append('COMMITMENT_UNBOUND')
    else:
        committed = set(int(x) for x in committed_order_ids)
        if committed != set(served):
            refusals.append({'code': 'COMMITTED_SET_MISMATCH',
                             'detail': {'missing_from_plan': sorted(committed - set(served))[:20],
                                        'planned_but_not_committed': sorted(set(served) - committed)[:20],
                                        'why': 'an accepted order must be served exactly once and no '
                                               'unaccepted order may be served'}})
    if instance.service_time[0] != 0.0 or (instance.demands is not None
                                           and instance.demands[0] != 0.0):
        refusals.append({'code': 'DEPOT_PROFILE_UNEXPECTED',
                         'detail': instance.depot_profile()})

    # ---- 4 per-vehicle replay: segment durations and door segments -------------------------
    travel = travel_fn or euclidean_travel_fn(instance.coords)
    per_vehicle = {}
    for vid in sorted(plan_by_vehicle):
        route = plan_by_vehicle[vid]
        if route is None:
            continue
        node = int((start_node_by_vehicle or {}).get(vid, 0))
        first_depart = max(float((ready_by_vehicle or {}).get(vid, clock)), float(clock))
        depart = first_depart
        segs = []
        for n in route:
            n = int(n)
            if n <= 0 or n not in allowed:
                continue
            svc = instance.service_time[n]
            if not _finite(svc) or svc < 0:
                refusals.append({'code': 'NON_FINITE_SERVICE_TIME',
                                 'detail': {'node': n, 'service_time': repr(svc)}})
                continue
            if svc <= 0.0:
                refusals.append({'code': 'SERVED_ORDER_WITH_ZERO_SERVICE_TIME',
                                 'detail': {'node': n, 'service_time': svc,
                                            'why': 'a served ORDER node with zero service time breaks '
                                                   'the P4 premise; zero service time is only '
                                                   'legitimate for the depot/placeholder, which must '
                                                   'not appear as a served stop'}})
                continue
            zc = instance.temp_class[n]
            if zc < 0 or zc >= n_zones:
                refusals.append({'code': 'TEMP_CLASS_OUT_OF_RANGE',
                                 'detail': {'node': n, 'temp_class': zc, 'n_zones': n_zones}})
                continue
            try:
                leg = float(travel(node, n))
            except SF.Refusal as exc:
                refusals.append({'code': exc.code, 'detail': exc.detail})
                continue
            if not _finite(leg) or leg < 0:
                refusals.append({'code': 'NON_FINITE_TRAVEL',
                                 'detail': {'from': node, 'to': n, 'travel': repr(leg)}})
                continue
            arrive = depart + leg
            tws = instance.tw_start[n] if instance.tw_start is not None else 0.0
            start = max(arrive, float(tws))
            finish = start + svc
            actual = finish - depart                      # == replay_thermal_envelope's segment
            lower = svc                                   # travel/wait-independent lower bound
            segs.append({'node': n, 'zone': zc, 'depart': depart, 'arrive': arrive,
                         'service_start': start, 'service_finish': finish,
                         'waited': start - arrive, 'travel_h': leg, 'service_time_h': svc,
                         'segment_h_actual': actual, 'segment_h_lower_bound': lower,
                         'door_zone': zc})
            depart, node = finish, n
        # the return leg opens no door on any zone -> unconstrained by P3'.  It is recorded as a
        # first-class segment because it is exactly the case the relaxed proposition buys: the
        # pre-fix P3 required EVERY segment to be at least Delta* long, so a short hop home would
        # have blocked certification of a plan that is perfectly safe.
        try:
            ret_leg = float(travel(node, 0))
        except SF.Refusal:
            ret_leg = None
        per_vehicle[vid] = {
            'stops': [s['node'] for s in segs], 'n_stops': len(segs),
            'start_node': int((start_node_by_vehicle or {}).get(vid, 0)),
            'first_depart': first_depart,
            'last_service_finish': depart,
            'return_arrival': (depart + ret_leg) if ret_leg is not None else None,
            'return_segment_h': ret_leg,
            'empty_suffix': not segs, 'segments': segs}
        if not segs:
            notes.append('EMPTY_SUFFIX_VEHICLE_%s' % vid)

    # ---- 5 the P3' condition, per zone ----------------------------------------------------
    metric = 'segment_h_lower_bound' if mode == 'strict_service_lower_bound' else 'segment_h_actual'
    if mode not in ('strict_service_lower_bound', 'replay'):
        refusals.append({'code': 'UNKNOWN_MODE', 'detail': {'mode': mode}})
    for z in zones:
        info = per_zone[z.zone]
        if info.get('return_time_h') is None:
            continue
        need = Fraction(z.return_time_h)
        rows = []
        for vid in sorted(per_vehicle):
            for s in per_vehicle[vid]['segments']:
                if s['zone'] != z.zone:
                    continue
                raw = s[metric]
                dur = Fraction(raw)
                rows.append({'vehicle': vid, 'node': s['node'],
                             'segment_h': float(dur), 'short_by_h': float(max(Fraction(0), need - dur)),
                             'ok': dur >= need})
        if rows:
            info['door_segments'] = rows
            info['n_door_segments'] = len(rows)
            info['min_door_segment_h'] = min(r['segment_h'] for r in rows)
            info['slack_h'] = float(info['min_door_segment_h'] - float(need))
            info['all_door_segments_long_enough'] = all(r['ok'] for r in rows)
        else:
            info['all_door_segments_long_enough'] = True          # vacuously true
    # segments that open no door are unconstrained -- recorded explicitly in the receipt so the
    # relaxation is visible instead of implied.  `short_door_segments` are the failures;
    # `short_but_doorless` are segments shorter than some zone's Delta* that the relaxed condition
    # deliberately tolerates (the pre-fix P3 would have refused them).
    short_door_segments, short_but_doorless = [], []
    for vid in sorted(per_vehicle):
        for s in per_vehicle[vid]['segments']:
            for z in zones:
                dur = Fraction(s[metric])
                if dur >= Fraction(z.return_time_h):
                    continue
                if s['zone'] == z.zone:
                    short_door_segments.append({'vehicle': vid, 'node': s['node'],
                                                'zone': z.zone, 'segment_h': s[metric],
                                                'needed_h': float(z.return_time_h)})
                else:
                    short_but_doorless.append({'vehicle': vid, 'node': s['node'],
                                               'serves_zone': s['zone'],
                                               'zone_checked': z.zone,
                                               'segment_h': s[metric],
                                               'needed_h': float(z.return_time_h)})
        ret = per_vehicle[vid].get('return_segment_h')
        if ret is not None:
            for z in zones:
                if Fraction(ret) < Fraction(z.return_time_h):
                    short_but_doorless.append({'vehicle': vid, 'node': 0,
                                               'serves_zone': None, 'zone_checked': z.zone,
                                               'segment_h': ret,
                                               'needed_h': float(z.return_time_h),
                                               'kind': 'return_leg_opens_no_door'})

    # ---- 6 entry-temperature binding (induction base) -------------------------------------
    entry_ok, entry_detail = True, {}
    if initial_temperature_by_vehicle is None:
        entry_ok = False
        notes.append('ENTRY_TEMPERATURE_UNBOUND')
    else:
        for vid in sorted(per_vehicle):
            temps = initial_temperature_by_vehicle.get(vid)
            if temps is None:
                entry_ok = False
                entry_detail[str(vid)] = 'NO_ENTRY_TEMPERATURE'
                continue
            temps = tuple(float(t) for t in temps)
            if len(temps) != n_zones:
                entry_ok = False
                entry_detail[str(vid)] = 'ENTRY_TEMPERATURE_LENGTH_MISMATCH'
                continue
            for z in zones:
                t = temps[z.zone]
                lo, hi = z.target_c, z.target_c + float(z.delta_real)
                if not (lo - 1e-12 <= t <= hi + 1e-12):
                    entry_ok = False
                    entry_detail['%s/zone%d' % (vid, z.zone)] = {'t': t, 'band': [lo, hi],
                                                                 'why': 'the induction base of P3\' '
                                                                        'needs the entry state inside '
                                                                        'the band'}

    # ---- 7 verdict ------------------------------------------------------------------------
    bad_zone = [z for z, i in sorted(per_zone.items())
                if i.get('all_door_segments_long_enough') is False]
    if refusals:
        verdict = REFUSED
    elif bad_zone:
        verdict = NOT_CERTIFIABLE
    elif (not entry_ok) or notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return _receipt(verdict, refusals, notes, per_zone, ctx, contract_hash, per_vehicle,
                    {'entry_state_in_band': entry_ok, 'entry_detail': entry_detail,
                     'n_door_segments_total': sum(i.get('n_door_segments') or 0
                                                  for i in per_zone.values()),
                     'min_door_slack_h': min([i['slack_h'] for i in per_zone.values()
                                              if i.get('slack_h') is not None] or [None]),
                     'short_door_segments': short_door_segments,
                     'short_but_doorless': short_but_doorless,
                     'n_short_but_doorless': len(short_but_doorless),
                     'bad_zones': bad_zone, 'metric': metric,
                     'n_stops_total': sum(v['n_stops'] for v in per_vehicle.values()),
                     'served_nodes': sorted(seen), 'n_served': len(seen),
                     'T_end': None if T_end is None else float(T_end)})


def _receipt(verdict, refusals, notes, per_zone, ctx, contract_hash, per_vehicle, summary):
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'verdict': verdict,
        'experimental_verdict': verdict,
        'contract_hash': contract_hash,
        'context': ctx,
        'refusals': refusals,
        'binding_notes': list(notes),
        'per_zone': {str(k): v for k, v in sorted(per_zone.items())},
        'per_vehicle': {str(k): v for k, v in sorted((per_vehicle or {}).items())},
        'summary': summary or {},
        'propositions_used': list(ALLOWED_PROPOSITIONS),
        'propositions_forbidden': list(FORBIDDEN_PROPOSITIONS),
        'refuted_proposition_used': False,
        'claim': ("P3' holds for this plan under the declared domain D1-D6 and the bindings recorded "
                  "above: every zone stays in [tau, tau+delta] at every segment boundary"),
        'not_claimed': [
            'that the plan is feasible, profitable or installable',
            'that any hard temperature bound is respected beyond the stated band',
            'that the executor STORED FLOATS stay in the band: the condition proved here is about the '
            'real-valued model, and promoting it to stored floats needs either a declared exp error '
            'bound or bit-identical replay (see q3_strict_float_experimental and the report)',
            'that the intra-segment (continuous) profile is inside the bounds (the executor defines '
            'no intra-segment trajectory at all)',
            'that the result transfers to any instance, node count or data family not checked here'],
    }


def instance_census(paths, contract, *, reader=None):
    """Read-only census over frozen instance files.  Returns facts, never a certificate.

    For each file: node-0 depot evidence, the zero-service-time node set, the minimum positive
    service time, and the comparison against the largest Delta* of the contract.  The caller supplies
    `reader(path) -> dict` so this function itself does no I/O policy decisions.
    """
    zones = SF.zone_thermals_exact(contract)
    worst = max(zones, key=lambda z: z.return_time_h)
    rows = []
    for p in paths:
        raw = reader(p)
        view = InstanceView(raw)
        zeros = view.zero_service_nodes()
        positives = [s for s in view.service_time if s > 0]
        rows.append({
            'file': str(p),
            'n_nodes': view.n_nodes,
            'depot_profile': view.depot_profile(),
            'zero_service_nodes': zeros,
            'zero_service_is_only_depot': zeros == [0],
            'min_positive_service_time_h': min(positives) if positives else None,
            'min_positive_service_time_min': (min(positives) * 60.0) if positives else None,
            'max_return_time_h': float(worst.return_time_h),
            'min_service_above_max_return_time': (bool(positives)
                                                  and Fraction(min(positives))
                                                  >= Fraction(worst.return_time_h)),
        })
    mins = [r['min_positive_service_time_min'] for r in rows
            if r['min_positive_service_time_min'] is not None]
    return {'rows': rows, 'n_files': len(rows),
            'all_zero_service_is_depot_only': all(r['zero_service_is_only_depot'] for r in rows),
            'all_min_service_above_return_time': all(r['min_service_above_max_return_time']
                                                     for r in rows),
            'worst_zone': worst.zone,
            'max_return_time_h': float(worst.return_time_h),
            'min_positive_service_time_min': min(mins) if mins else None,
            'any_non_depot_zero_service': any(not r['zero_service_is_only_depot'] for r in rows)}
