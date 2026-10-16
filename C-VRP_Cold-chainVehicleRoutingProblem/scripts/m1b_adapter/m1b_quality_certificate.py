#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix: Omega10 as a COVERAGE-COMPLETENESS certificate, plus the analytic v2 quality
lower bound.

Fixes (Q2-R): B1 (coverage hole) and B3 (unconstrained full-node truth tables / arbitrary
callables accepted as trusted input).

THE B1 HOLE, precisely
  The audited version's default lot provider harvested only the cargo manifests of vehicles that
  had a plan.  With nothing on board it returned an EMPTY set; `check_omega10_initialization`
  returned `ok=True` with `NO_RELEVANT_LOTS`; and the orchestrator read `ok=True` as a pass.  So a
  candidate plan carrying a revealed fish order whose true initial quality was 1.0 (contract says
  9.0) produced `M1B_Q2_ENGINEERING_CERTIFIED` with `n_checked = 0`.  "Nothing to check" and
  "checked and fine" were indistinguishable one layer up.

THE FIX
  Omega10 is now a statement about an EXPLICIT REQUIRED SET E and an EXPLICIT VERIFIED SET C:
      E = E_loaded (on-board fish) UNION E_committed (accepted, unfinished fish commitments)
          UNION E_candidate (fish orders the CANDIDATE plan carries, currently revealed)
  The certificate requires C >= E, reports E \\ C as COVERAGE_INCOMPLETE and C \\ E as
  COVERAGE_EXTRA, and refuses any duplicate-identity conflict.  `checked == 0` is NOT a blanket
  refusal: it is admissible ONLY when E is genuinely empty AND the source is complete, and it is
  then reported as `NO_FISH_IN_SCOPE` with an explicit statement that no existing fish was
  verified.  A candidate fish whose value cannot be established refuses.

  Note on the Q2 report: its Omega10 description claimed the gate also reads "revealed orders the
  candidate references".  No code did.  This module makes that claim TRUE.

STRUCTURAL VISIBILITY (B3)
  The audited version accepted `initial_quality_by_node` (a whole-node table) and arbitrary
  `temp_class_of` / `quantity_of` callables, so a caller could pass a table that happened to be
  complete because it held FUTURE values.  Those are replaced by an identity-carrying, READ-ONLY
  `PublicOrderAttributes` provider that (a) declares the set of nodes it is allowed to serve and
  (b) raises on any other node, so future information cannot be laundered through it.
"""
from __future__ import annotations

OMEGA10_OK = 'OMEGA10_OK'
OMEGA10_INITIAL_VALUE_MISMATCH = 'OMEGA10_INITIAL_VALUE_MISMATCH'
OMEGA10_LOT_FIELD_INVALID = 'OMEGA10_LOT_FIELD_INVALID'
OMEGA10_NO_FISH_IN_SCOPE = 'OMEGA10_NO_FISH_IN_SCOPE'
OMEGA10_COVERAGE_INCOMPLETE = 'OMEGA10_COVERAGE_INCOMPLETE'
OMEGA10_COVERAGE_EXTRA = 'OMEGA10_COVERAGE_EXTRA'
OMEGA10_COVERAGE_IDENTITY_CONFLICT = 'OMEGA10_COVERAGE_IDENTITY_CONFLICT'
OMEGA10_CANDIDATE_VALUE_UNKNOWN = 'OMEGA10_CANDIDATE_VALUE_UNKNOWN'
OMEGA10_SOURCE_UNIDENTIFIABLE = 'OMEGA10_SOURCE_UNIDENTIFIABLE'

QUALITY_LB_OK = 'QUALITY_LB_OK'
QUALITY_LB_BELOW_THRESHOLD = 'QUALITY_LB_BELOW_THRESHOLD'
QUALITY_CLASS_OUT_OF_SCOPE = 'QUALITY_CLASS_OUT_OF_SCOPE'
QUALITY_TIME_UNIT_INVALID = 'QUALITY_TIME_UNIT_INVALID'
QUALITY_NUMERIC_INVALID = 'QUALITY_NUMERIC_INVALID'
QUALITY_HORIZON_UNAVAILABLE = 'QUALITY_HORIZON_UNAVAILABLE'

IMPLEMENTATION_VERSION = 'M1B_Q2FIX'

EPSILON_INIT = 1e-9
EPSILON_THRESHOLD = 1e-9
EPSILON_RETURN = 1e-6

#: the legacy name kept so the orchestration error vocabulary stays stable
OMEGA10_NO_RELEVANT_LOTS = OMEGA10_NO_FISH_IN_SCOPE


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


class PublicOrderAttributes(object):
    """Identity-carrying, read-only order attributes for a DECLARED node set.

    Replaces the audited `initial_quality_by_node` table and the `temp_class_of` / `quantity_of`
    callables.  Any node outside `served_nodes` raises, which is what stops a caller from feeding
    future truth values into certification (the B3 fix).

    `source_identity` is mandatory: a provider that cannot name its source is refused, because
    "trusted" has to be attributable to something.
    """

    __slots__ = ('_iq', '_tc', '_qty', 'served_nodes', 'source_identity')

    def __init__(self, *, initial_quality, temp_class, quantity, served_nodes, source_identity):
        if not source_identity:
            raise ValueError(OMEGA10_SOURCE_UNIDENTIFIABLE + ': source_identity is required')
        if served_nodes is None:
            raise ValueError(OMEGA10_SOURCE_UNIDENTIFIABLE + ': served_nodes is required')
        self._iq = {int(k): float(v) for k, v in (initial_quality or {}).items()}
        self._tc = {int(k): int(v) for k, v in (temp_class or {}).items()}
        self._qty = {int(k): float(v) for k, v in (quantity or {}).items()}
        self.served_nodes = frozenset(int(x) for x in served_nodes)
        self.source_identity = str(source_identity)

    def _gate(self, node):
        if int(node) not in self.served_nodes:
            raise PermissionError('%s:%d' % (OMEGA10_SOURCE_UNIDENTIFIABLE, int(node)))

    def initial_quality_of(self, node):
        self._gate(node)
        if int(node) not in self._iq:
            raise KeyError('%s:initial_quality:%d' % (OMEGA10_SOURCE_UNIDENTIFIABLE, int(node)))
        return self._iq[int(node)]

    def temp_class_of(self, node):
        self._gate(node)
        if int(node) not in self._tc:
            raise KeyError('%s:temp_class:%d' % (OMEGA10_SOURCE_UNIDENTIFIABLE, int(node)))
        return self._tc[int(node)]

    def quantity_of(self, node):
        self._gate(node)
        if int(node) not in self._qty:
            raise KeyError('%s:quantity:%d' % (OMEGA10_SOURCE_UNIDENTIFIABLE, int(node)))
        return self._qty[int(node)]

    def identity(self):
        return {'source_identity': self.source_identity,
                'n_served_nodes': len(self.served_nodes)}


# ---------------------------------------------------------------------------------------------
# Omega10: enumeration + coverage
# ---------------------------------------------------------------------------------------------
def enumerate_required_fish_scope(*, on_board_lots, committed_fish_nodes, candidate_fish_nodes,
                                  revealed_nodes, provider, in_scope_classes):
    """Build the REQUIRED set E and report the evidence for each member.

    Returns (E, verdict, detail).  `E` is a dict `{order_id: {'origin':..., 'value':..., 'lot'}}`.

    Only REVEALED nodes may be read.  An unrevealed node appearing in `committed_fish_nodes` or
    `candidate_fish_nodes` is a SOURCE/ordering error and refuses, because reading it would use
    future information.
    """
    if provider is None:
        return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {'field': 'provider'}
    if not getattr(provider, 'source_identity', None):
        return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {'why': 'provider has no source identity'}
    revealed = frozenset(int(x) for x in (revealed_nodes or ()))
    E, provenance = {}, []

    def _class_of(node):
        try:
            return int(provider.temp_class_of(node))
        except PermissionError:
            return None
        except KeyError:
            return None

    # ---- loaded --------------------------------------------------------------
    for lot in (on_board_lots or ()):
        oid = getattr(lot, 'order_id', None)
        if oid is None:
            return None, OMEGA10_LOT_FIELD_INVALID, {'why': 'on-board lot without order_id'}
        if int(oid) in E:
            return None, OMEGA10_COVERAGE_IDENTITY_CONFLICT, {
                'order_id': int(oid), 'why': 'duplicate identity between loaded and other sources'}
        E[int(oid)] = {'origin': 'loaded', 'lot': lot}
        provenance.append({'order_id': int(oid), 'origin': 'loaded'})

    # ---- committed -----------------------------------------------------------
    for node in (committed_fish_nodes or ()):
        n = int(node)
        if n not in revealed:
            return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {
                'node': n, 'why': 'a committed node is not revealed; reading it would use future '
                                  'information'}
        c = _class_of(n)
        if c is None:
            return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {'node': n,
                                                         'why': 'temp class unavailable'}
        if c not in in_scope_classes:
            continue
        if n in E:
            return None, OMEGA10_COVERAGE_IDENTITY_CONFLICT, {'order_id': n,
                                                              'why': 'duplicate identity'}
        E[n] = {'origin': 'committed', 'node': n}
        provenance.append({'order_id': n, 'origin': 'committed'})

    # ---- candidate -----------------------------------------------------------
    for node in (candidate_fish_nodes or ()):
        n = int(node)
        if n not in revealed:
            return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {
                'node': n, 'why': 'a candidate node is not revealed; reading it would use future '
                                  'information'}
        c = _class_of(n)
        if c is None:
            # THIS is the B1 fix: a candidate fish whose class/value cannot be established is a
            # REFUSAL, never a silent pass.
            return None, OMEGA10_CANDIDATE_VALUE_UNKNOWN, {
                'node': n, 'why': 'the candidate carries a node whose attributes are unavailable '
                                  'from the declared source'}
        if c not in in_scope_classes:
            continue
        if n in E:
            continue                       # already required via loaded/committed
        E[n] = {'origin': 'candidate', 'node': n}
        provenance.append({'order_id': n, 'origin': 'candidate'})

    return E, OMEGA10_OK, {'n_required': len(E), 'provenance': provenance,
                           'revealed_nodes_supplied': len(revealed)}


def check_omega10_coverage(E, provider, per_class_constants, *, epsilon_init=EPSILON_INIT,
                           verified_order_ids=None):
    """Verify EVERY member of E and mechanically compare C against E.

    Returns (C_or_None, verdict, detail).
    """
    if E is None:
        return None, OMEGA10_SOURCE_UNIDENTIFIABLE, {}
    if not _finite(epsilon_init) or epsilon_init < 0:
        return None, OMEGA10_LOT_FIELD_INVALID, {'field': 'epsilon_init', 'got': repr(epsilon_init)}
    C, mismatches, unknown, bad = {}, [], [], []
    for oid in sorted(E):
        rec = E[oid]
        lot = rec.get('lot')
        if lot is not None:
            cls = getattr(lot, 'temp_class', None)
            iq = getattr(lot, 'initial_quality', None)
        else:
            try:
                cls = provider.temp_class_of(rec['node'])
                iq = provider.initial_quality_of(rec['node'])
            except (PermissionError, KeyError) as exc:
                unknown.append({'order_id': oid, 'origin': rec['origin'],
                                'why': 'value unavailable from the declared source',
                                'exception': type(exc).__name__})
                continue
        if cls is None or iq is None or not _finite(iq):
            bad.append({'order_id': oid, 'why': 'missing or non-finite field',
                        'temp_class': repr(cls), 'initial_quality': repr(iq)})
            continue
        const = (per_class_constants or {}).get(int(cls))
        if const is None:
            bad.append({'order_id': oid, 'why': 'temperature class absent from the contract',
                        'temp_class': int(cls)})
            continue
        if not const.get('in_scope', True):
            C[oid] = {'origin': rec['origin'], 'in_scope': False}
            continue
        want, got = float(const['initial_value']), float(iq)
        C[oid] = {'origin': rec['origin'], 'temp_class': int(cls), 'expected': want,
                  'actual': got, 'abs_dev': abs(got - want), 'in_scope': True}
        if abs(got - want) > float(epsilon_init):
            mismatches.append({'order_id': oid, 'origin': rec['origin'], 'temp_class': int(cls),
                               'expected': want, 'actual': got, 'abs_dev': abs(got - want)})

    extra = sorted(set(int(x) for x in (verified_order_ids or ())) - set(int(x) for x in E))
    detail = {'n_required': len(E), 'n_covered': len(C), 'n_mismatch': len(mismatches),
              'mismatches': mismatches[:20], 'n_value_unknown': len(unknown),
              'value_unknown': unknown[:20], 'n_field_errors': len(bad), 'field_errors': bad[:20],
              'extra_verified_not_required': extra, 'epsilon_init': float(epsilon_init),
              'required_provenance': sorted(int(x) for x in E)}
    if bad:
        return C, OMEGA10_LOT_FIELD_INVALID, detail
    if mismatches:
        return C, OMEGA10_INITIAL_VALUE_MISMATCH, detail
    if unknown or (len(C) < len(E)):
        return C, OMEGA10_COVERAGE_INCOMPLETE, detail
    if extra:
        return C, OMEGA10_COVERAGE_EXTRA, detail
    if not E:
        # E genuinely empty.  Admissible, but it must NOT be read as having verified any existing
        # fish -- hence a DISTINCT verdict rather than OMEGA10_OK.
        detail['statement'] = ('the required fish set is EMPTY and the source is complete; this '
                              'certifies NOTHING about any fish, and must not be reported as a '
                              'positive verification of a fish initial value')
        return C, OMEGA10_NO_FISH_IN_SCOPE, detail
    return C, OMEGA10_OK, detail


def certify_omega10_coverage(*, on_board_lots, committed_fish_nodes, candidate_fish_nodes,
                             revealed_nodes, provider, per_class_constants,
                             epsilon_init=EPSILON_INIT, verified_order_ids=None):
    """One call: build E, verify every member, compare C against E.  Returns (ok, verdict, detail)."""
    E, v1, d1 = enumerate_required_fish_scope(
        on_board_lots=on_board_lots, committed_fish_nodes=committed_fish_nodes,
        candidate_fish_nodes=candidate_fish_nodes, revealed_nodes=revealed_nodes,
        provider=provider,
        in_scope_classes={c for c, k in (per_class_constants or {}).items()
                          if k.get('in_scope', True)})
    if v1 != OMEGA10_OK:
        return False, v1, d1
    C, v2, d2 = check_omega10_coverage(E, provider, per_class_constants,
                                       epsilon_init=epsilon_init,
                                       verified_order_ids=verified_order_ids)
    d2['enumerate'] = d1
    # NO_FISH_IN_SCOPE is 'admissible but proves nothing' -- reported as ok=True so the flow can
    # continue, with a verdict that CANNOT be mistaken for a positive fish verification.
    ok = v2 in (OMEGA10_OK, OMEGA10_NO_FISH_IN_SCOPE)
    return ok, v2, d2


def check_omega10_initialization(lots, per_class_constants, *, epsilon_init=EPSILON_INIT):
    """LEGACY SHIM -- DISCLOSURE REQUIRED.

    Reproduces the PRE-FIX PER-LOT semantics, including `checked == 0 -> ok=True` with a
    NO_RELEVANT_LOTS-style verdict.  It exists ONLY so the historical
    `test_m1b_q2_certification.py` remains executable: that file cannot be edited (it is the
    record of the pre-fix 49 checks), and without this shim it would crash on import-level
    attribute lookups, which would hide its recorded failures.

    THE FIX DOES NOT USE THIS FUNCTION.  Certification goes through
    `certify_omega10_coverage`, which builds the required set E and REFUSES when a candidate fish
    is unverified -- the exact case this shim lets through, and the case B1 recorded as a false
    pass.  Any caller that uses this shim inherits the pre-fix hole.
    """
    if not _finite(epsilon_init) or epsilon_init < 0:
        return False, OMEGA10_LOT_FIELD_INVALID, {'field': 'epsilon_init',
                                                  'got': repr(epsilon_init)}
    checked, out_of_scope, mismatches, bad = 0, 0, [], []
    q0_safe = {}
    for lot in lots or ():
        c = getattr(lot, 'temp_class', None)
        iq = getattr(lot, 'initial_quality', None)
        if c is None or iq is None:
            bad.append({'order_id': getattr(lot, 'order_id', None), 'why': 'missing field'})
            continue
        if not _finite(c) or not _finite(iq):
            bad.append({'order_id': getattr(lot, 'order_id', None), 'why': 'non-finite'})
            continue
        c = int(c)
        const = (per_class_constants or {}).get(c)
        if const is None:
            bad.append({'order_id': getattr(lot, 'order_id', None),
                        'why': 'temperature class absent from contract', 'temp_class': c})
            continue
        if not const.get('in_scope', True):
            out_of_scope += 1
            continue
        checked += 1
        want, got = float(const['initial_value']), float(iq)
        q0_safe[c] = want - float(epsilon_init)
        if abs(got - want) > float(epsilon_init):
            mismatches.append({'order_id': getattr(lot, 'order_id', None), 'temp_class': c,
                               'expected': want, 'actual': got, 'abs_dev': abs(got - want),
                               'epsilon_init': float(epsilon_init)})
    detail = {'n_checked': checked, 'n_out_of_scope': out_of_scope,
              'n_mismatch': len(mismatches), 'mismatches': mismatches[:20],
              'n_field_errors': len(bad), 'field_errors': bad[:20],
              'epsilon_init': float(epsilon_init), 'q0_safe_used': q0_safe,
              'LEGACY_SHIM_WARNING': 'pre-fix per-lot semantics; NOT used by certification'}
    if bad:
        return False, OMEGA10_LOT_FIELD_INVALID, detail
    if mismatches:
        return False, OMEGA10_INITIAL_VALUE_MISMATCH, detail
    if checked == 0:
        return True, OMEGA10_NO_FISH_IN_SCOPE, detail
    return True, OMEGA10_OK, detail


def check_boundary_initial_value(order, per_class_constants, *, epsilon_init=EPSILON_INIT):
    """The SAME invariant at the reveal / accept / pickup boundary."""
    iq = getattr(order, 'initial_quality', None)
    cls = getattr(order, 'temp_class', None)
    if iq is None or cls is None or not _finite(iq):
        return False, OMEGA10_LOT_FIELD_INVALID, {'order_id': getattr(order, 'order_id', None)}
    const = (per_class_constants or {}).get(int(cls))
    if const is None or not const.get('in_scope', True):
        return False, OMEGA10_SOURCE_UNIDENTIFIABLE, {'temp_class': cls}
    want, got = float(const['initial_value']), float(iq)
    if abs(got - want) > float(epsilon_init):
        return False, OMEGA10_INITIAL_VALUE_MISMATCH, {
            'order_id': getattr(order, 'order_id', None), 'expected': want, 'actual': got}
    return True, OMEGA10_OK, {}


def required_initial_value_invariant_report():
    return {
        'invariant': 'for every temperature class c and every in-scope fish order i: '
                     'initial_quality(i) == QualityConfigV2.initial_value[c] within '
                     'epsilon_init = 1e-9',
        'enforcement_points': ['data loading', 'reveal / accept (check_boundary_initial_value)',
                              'pickup (the lot written into CargoLotState)',
                              'pre-submit coverage certificate (certify_omega10_coverage)'],
        'required_set_definition': 'E = on-board fish UNION accepted-unfinished fish commitments '
                                   'UNION fish orders the candidate plan carries (revealed only)',
        'note': 'an empty E is reported as NO_FISH_IN_SCOPE and certifies nothing about any fish',
    }


# ---------------------------------------------------------------------------------------------
# The analytic lower bound (unchanged mathematics; version-stamped)
# ---------------------------------------------------------------------------------------------
def horizon_hours(T_end, t_min, hours_per_time_unit, *,
                  epsilon_return=EPSILON_RETURN, require_t_min_verified=True):
    if not _finite(T_end) or not _finite(t_min) or not _finite(hours_per_time_unit):
        return None, QUALITY_NUMERIC_INVALID, {'T_end': repr(T_end), 't_min': repr(t_min),
                                               'hours_per_time_unit': repr(hours_per_time_unit)}
    if hours_per_time_unit <= 0:
        return None, QUALITY_TIME_UNIT_INVALID, {'hours_per_time_unit': hours_per_time_unit}
    tmin = float(t_min)
    if require_t_min_verified is False:
        tmin = 0.0
    span = (float(T_end) + float(epsilon_return)) - tmin
    if span < 0:
        span = 0.0
    return span * float(hours_per_time_unit), None, {
        'T_end': float(T_end), 't_min': tmin, 'epsilon_return': float(epsilon_return),
        'hours_per_time_unit': float(hours_per_time_unit),
        'implementation_version': IMPLEMENTATION_VERSION,
        'note': 'NOT a linear rescaling of a unitless horizon; recomputed from the formula'}


def certify_quality_lb(per_class_constants, *, T_end, t_min, hours_per_time_unit,
                       epsilon_init=EPSILON_INIT, epsilon_threshold=EPSILON_THRESHOLD):
    H, verdict, hdetail = horizon_hours(T_end, t_min, hours_per_time_unit)
    if H is None:
        return None, verdict, hdetail
    out, below, any_in_scope = {}, [], False
    for c, const in sorted((per_class_constants or {}).items()):
        if not const.get('in_scope', True):
            out[c] = {'in_scope': False, 'dynamics_order': const.get('dynamics_order'),
                      'threshold_type': const.get('threshold_type'),
                      'salable': None, 'claim': 'NOT_CERTIFIED_BY_THIS_BOUND'}
            continue
        any_in_scope = True
        q0_safe, k, thr = float(const['q0_safe']), float(const['rate_per_hour']), \
            float(const['threshold'])
        q_lb = max(0.0, q0_safe - k * H)
        if not _finite(q_lb):
            return out, QUALITY_NUMERIC_INVALID, {'class': c, 'q_lb': repr(q_lb)}
        salable = q_lb >= thr - float(epsilon_threshold)
        out[c] = {'in_scope': True, 'q0_contract': float(const['initial_value']),
                  'q0_safe': q0_safe, 'k': k, 'H_max_hours': H, 'q_lb': q_lb, 'threshold': thr,
                  'epsilon_threshold': float(epsilon_threshold), 'salable': salable,
                  'margin': q_lb - thr,
                  'H_crit': (float(const['initial_value']) - thr + float(epsilon_threshold)) / k
                  if k > 0 else None}
        if not salable:
            below.append(c)
    detail = {'H_max': H, 'horizon_detail': hdetail, 'per_class': out,
              'classes_below_threshold': below, 'implementation_version': IMPLEMENTATION_VERSION,
              'tomato_note': 'class 0 (first order, threshold_type none) is OUT OF SCOPE: '
                             'salable is None and is NOT counted as unsalable; this bound makes '
                             'NO claim about the tomato continuous metric'}
    if not any_in_scope:
        return out, QUALITY_CLASS_OUT_OF_SCOPE, detail
    if below:
        return out, QUALITY_LB_BELOW_THRESHOLD, detail
    return out, QUALITY_LB_OK, detail
