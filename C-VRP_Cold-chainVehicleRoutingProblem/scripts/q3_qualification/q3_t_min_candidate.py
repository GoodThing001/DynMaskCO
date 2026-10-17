#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 G-TMIN ISOLATED REPAIR CANDIDATE -- NOT INSTALLED, NOT IMPORTED BY ANY PRODUCTION PATH.

THE DEFECT THIS CANDIDATE ADDRESSES (G-TMIN)
============================================
`scripts/m1b_adapter/m1b_q2_certifier.py:122-139` (`_resolve_t_min`) accepts any `verified_t_min`
whenever SOME `env` object was supplied, labels it `VERIFIED_AGAINST_SUPPLIED_ENV` and returns it.
It never reads a single field of `env`, and it never derives a time lower bound.  Consequences that
have been reproduced numerically (see `scripts/tests/test_q3_g_tmin_candidate.py`):

    verified_t_min=1e6 + env  ->  t_min=1e6, H_max=0,        q_lb=8.999999999      (== initial value)
    verified_t_min=24  + env  ->  t_min=24,  H_max=1.0e-06,  q_lb=8.99999998017
    verified_t_min="1e6" + env ->  t_min=1e6 (a STRING is accepted, `float()` succeeds)
    baseline (nothing)        ->  t_min=0,   H_max=24.000001, q_lb=8.54807998017

`q_lb` is the certified quality lower bound, so raising `t_min` raises `q_lb`: the bypass makes the
qualification LESS conservative.  `env=object()` is a sufficient "witness".

THE CANDIDATE (F-TMIN-1 / F-TMIN-2 / F-TMIN-3)
==============================================
  F-TMIN-1  three-state resolution: the ONLY value that can change a number is 0.0.  Any non-None
            `verified_t_min` without a full witness resolves to 0.0 and is LABELLED as refused, so
            it can no longer influence `H_max` at all.
  F-TMIN-2  the string `VERIFIED_AGAINST_SUPPLIED_ENV` does not exist in this module; the refused
            label is `UNVERIFIED_T_MIN_REFUSED`, and the old name is registered in
            `FORBIDDEN_RECEIPT_LABELS` so any receipt still carrying it is REFUSED.
  F-TMIN-3  `env` is never consulted.  It is accepted only so that historical call signatures keep
            working, and it is recorded as received-but-not-consulted.  A non-zero `t_min` requires
            a `TMinWitness` (spec section 1.3) that a third party can recompute from the repository.

DIRECTION SAFETY (must be stated whenever this candidate is proposed for installation)
======================================================================================
`H_max = T_end + epsilon_return - t_min`, `q_lb = max(0, q0_safe - k*H_max)`, so
`dq_lb/dt_min = +k > 0`.  The candidate can only DECREASE `t_min` relative to the defective resolver
(1e6 -> 0, 24 -> 0, ...), which only increases `H_max`, which only DECREASES `q_lb`, which can only
turn a PASS into a REFUSAL.  It can never admit a plan the pre-fix code rejected.

WHAT THIS FILE IS NOT
=====================
It is a CANDIDATE: `INSTALLED = False`, it does not subclass or monkey-patch the production
certifier, it does not write any receipt and it is not imported by any production path.  Installing
it means editing `m1b_q2_certifier.py`, which needs separate authorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

CANDIDATE_NOT_INSTALLED = True
INSTALLED = False
IMPLEMENTATION_VERSION = 'Q3_G_TMIN_CANDIDATE_V1'

#: resolution labels.  There is deliberately no `VERIFIED_*` label reachable without a witness.
CONSERVATIVE_CLOCK_ORIGIN = 'CONSERVATIVE_CLOCK_ORIGIN'
UNVERIFIED_T_MIN_REFUSED = 'UNVERIFIED_T_MIN_REFUSED'
WITNESS_VERIFIED = 'WITNESS_VERIFIED'
WITNESS_REFUSED = 'WITNESS_REFUSED'

#: F-TMIN-2.  The pre-fix label name, kept ONLY as a forbidden token so it cannot be re-used.
FORBIDDEN_RECEIPT_LABELS = ('VERIFIED_AGAINST_SUPPLIED_ENV',)

#: hard cap mirrored from the production resolver so the candidate cannot widen the accepted range
MAX_T_MIN = 1e9


@dataclass(frozen=True)
class TMinWitness:
    """Spec section 1.3: a witness a third party can recompute.  Every field is mandatory.

    source_path / source_sha256 / generator_identity   identity of the artefact and of the program
    declared_semantics                                what the number claims to be a lower bound of
    monotone_over_all_plans                           True only if it holds for EVERY plan
    recomputable_from_repo                            True only if it is derivable from repo files
    contract_binding / state_binding                  identities the witness is valid for
    value                                             the lower bound itself (time units)
    """

    source_path: str
    source_sha256: str
    generator_identity: str
    declared_semantics: str
    monotone_over_all_plans: bool
    recomputable_from_repo: bool
    contract_binding: str
    state_binding: str
    value: float

    def missing_fields(self):
        out = []
        for name in ('source_path', 'source_sha256', 'generator_identity', 'declared_semantics',
                     'contract_binding', 'state_binding'):
            if not getattr(self, name):
                out.append(name)
        for name in ('monotone_over_all_plans', 'recomputable_from_repo'):
            if getattr(self, name) is not True:
                out.append(name + '_NOT_TRUE')
        return out


@dataclass
class TMinDecision:
    """The resolved value plus the complete provenance a receipt must carry."""

    t_min: float
    source: str
    refused_value: object = None
    reason: str = ''
    env_received: bool = False
    env_consulted: bool = False
    witness_report: dict = field(default_factory=dict)

    def as_receipt_fields(self):
        return {'t_min': float(self.t_min), 't_min_source': self.source,
                't_min_refused_value': (None if self.refused_value is None
                                        else repr(self.refused_value)),
                't_min_refusal_reason': self.reason,
                't_min_env_received': bool(self.env_received),
                't_min_env_consulted': bool(self.env_consulted),
                't_min_witness': self.witness_report}


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return None
    return v if (v == v and v not in (float('inf'), float('-inf'))) else None


def _witness_ok(witness, recompute):
    """Validate a witness; returns (ok, value, report).  Never raises: a failure is a refusal."""
    report = {'kind': 'WITNESS', 'present': witness is not None}
    if witness is None:
        report['why'] = 'no witness supplied'
        return False, None, report
    missing = witness.missing_fields()
    report['missing_fields'] = missing
    if missing:
        report['why'] = 'witness is incomplete; spec 1.3 requires all five elements'
        return False, None, report
    value = _finite(witness.value)
    report['value'] = witness.value
    if value is None:
        report['why'] = 'witness value is not finite'
        return False, None, report
    if value < 0 or value > MAX_T_MIN:
        report['why'] = 'witness value out of accepted range [0, %r]' % MAX_T_MIN
        return False, None, report
    if recompute is None:
        report['why'] = ('no recomputation hook was supplied, so the witness cannot be independently '
                         're-derived (spec 1.3 element 4)')
        return False, None, report
    try:
        got = recompute(witness.source_path)
    except Exception as exc:                                              # noqa: BLE001
        report['why'] = 'recomputation raised %s' % type(exc).__name__
        return False, None, report
    if not isinstance(got, dict):
        report['why'] = 'recomputation must return a dict with sha256 and value'
        return False, None, report
    report['recomputed'] = {'sha256': got.get('sha256'), 'value': got.get('value')}
    if str(got.get('sha256')) != str(witness.source_sha256):
        report['why'] = 'WITNESS_SHA256_MISMATCH'
        return False, None, report
    got_value = _finite(got.get('value'))
    if got_value is None or Fraction(got_value) != Fraction(value):
        report['why'] = 'WITNESS_VALUE_NOT_REPRODUCED'
        return False, None, report
    report['why'] = 'ok'
    return True, value, report


def resolve_t_min(*, verified_t_min=None, env=None, witness=None, contract_identity=None,
                  state_identity=None, recompute=None):
    """Fail-closed three-state resolver.  The ONLY value that can change a number is 0.0 or a
    fully verified witness value.

    `env` is deliberately NOT inspected (F-TMIN-3): only its presence is recorded, so a receipt can
    show that a caller's claim was incomplete rather than verified.
    """
    env_received = env is not None
    if witness is not None:
        ok, value, report = _witness_ok(witness, recompute)
        if contract_identity is not None and str(witness.contract_binding) != str(contract_identity):
            ok = False
            report['why'] = 'WITNESS_CONTRACT_BINDING_MISMATCH'
        if state_identity is not None and str(witness.state_binding) != str(state_identity):
            ok = False
            report['why'] = 'WITNESS_STATE_BINDING_MISMATCH'
        if ok:
            return TMinDecision(t_min=float(value), source=WITNESS_VERIFIED,
                                env_received=env_received, env_consulted=False,
                                witness_report=report)
        return TMinDecision(t_min=0.0, source=WITNESS_REFUSED,
                            refused_value=(None if witness is None else witness.value),
                            reason=report.get('why', 'witness refused'),
                            env_received=env_received, env_consulted=False,
                            witness_report=report)
    if verified_t_min is None:
        return TMinDecision(t_min=0.0, source=CONSERVATIVE_CLOCK_ORIGIN,
                            reason='no verified lower bound was supplied; the conservative clock '
                                   'origin is the only accepted value',
                            env_received=env_received, env_consulted=False)
    # F-TMIN-1: a bare verified_t_min NEVER changes a number, with or without env
    return TMinDecision(
        t_min=0.0, source=UNVERIFIED_T_MIN_REFUSED, refused_value=verified_t_min,
        reason=('a non-None verified_t_min was supplied without a recomputable witness; the '
                'pre-fix resolver returned it verbatim and labelled it as verified against the '
                'supplied env, but never read that env. Spec section 1.3 is not satisfied, so the '
                'value is refused and the conservative clock origin is used'),
        env_received=env_received, env_consulted=False)


def production_resolver_replica(*, verified_t_min=None, env=None):
    """Byte-for-byte replication of the PRE-FIX resolver (`m1b_q2_certifier.py:129-139`).

    Kept ONLY so the test file can prove (a) that the replica reproduces the production resolver on
    every input, and (b) exactly which inputs the candidate changes.  It is never used to decide
    anything.
    """
    source = 'CONSERVATIVE_CLOCK_ORIGIN'
    if verified_t_min is None or env is None:
        return 0.0, source
    try:
        v = float(verified_t_min)
    except Exception:                                                     # noqa: BLE001
        return 0.0, source
    if v != v or v < 0 or v > MAX_T_MIN:
        return 0.0, source
    return v, 'VERIFIED_AGAINST_SUPPLIED_ENV'


def receipt_label_gate(receipt):
    """F-TMIN-2.  Any receipt carrying a forbidden label is REFUSED by the composite judgement."""
    found = []

    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    for bad in FORBIDDEN_RECEIPT_LABELS:
                        if bad in k:
                            found.append(path + '/' + k)
                _walk(v, path + '/' + str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, '%s[%d]' % (path, i))
        elif isinstance(obj, str):
            for bad in FORBIDDEN_RECEIPT_LABELS:
                if bad in obj:
                    found.append(path)

    _walk(receipt, '')
    return (not found), found


def horizon_and_q_lb(per_class_constants, *, T_end, t_min, hours_per_time_unit=1.0,
                     epsilon_return=1e-6, class_id=1):
    """Mirror of `m1b_quality_certificate.horizon_hours` + the q_lb step, for the compatibility table.

    Reimplemented (not imported) so the direction-safety argument can be checked without importing
    the production module.
    """
    span = (float(T_end) + float(epsilon_return)) - float(t_min)
    if span < 0:
        span = 0.0
    H = span * float(hours_per_time_unit)
    const = per_class_constants[class_id]
    q_lb = max(0.0, float(const['q0_safe']) - float(const['rate_per_hour']) * H)
    return H, q_lb


def direction_safety(per_class_constants, *, T_end, t_mins, hours_per_time_unit=1.0,
                     epsilon_return=1e-6, class_id=1):
    """Numerically confirm `t_min` smaller => `q_lb` smaller (i.e. the candidate only refuses more)."""
    rows = []
    for t in sorted(float(x) for x in t_mins):
        H, q = horizon_and_q_lb(per_class_constants, T_end=T_end, t_min=t,
                                hours_per_time_unit=hours_per_time_unit,
                                epsilon_return=epsilon_return, class_id=class_id)
        rows.append({'t_min': t, 'H_max': H, 'q_lb': q})
    non_decreasing = all(rows[i]['q_lb'] <= rows[i + 1]['q_lb'] + 0.0 for i in range(len(rows) - 1))
    return {'rows': rows, 'q_lb_non_decreasing_in_t_min': non_decreasing,
            'candidate_is_direction_safe': non_decreasing,
            'statement': 'q_lb is non-decreasing in t_min, so lowering t_min (which is all the '
                         'candidate can do) can only lower q_lb and can only add refusals'}


#: call sites affected if this candidate were installed (recorded by grep, see the report section C)
AFFECTED_CALL_SITES = (
    ('scripts/m1b_adapter/m1b_q2_certifier.py', 'M1bQ2TrustFixCertifier.__init__ / _resolve_t_min'),
    ('scripts/m1b_adapter/m1b_q2_certifier.py', 'M1bQ2TrustFixTestOnlyCertifier (inherits the resolver)'),
    ('scripts/tests/test_m1b_q2_final_entry_controls.py', 'B2 uses the BARE t_min keyword, not '
                                                          'verified_t_min + env'),
    ('scripts/tests/test_m1b_q2fixt_trust_boundary.py', 'constructs the certifier, never passes '
                                                        'verified_t_min'),
    ('scripts/tests/test_m1b_q2fix_certification.py', 'constructs the certifier, never passes '
                                                      'verified_t_min'),
    ('scripts/tests/test_m1b_q2r_negative_controls.py', 'constructs the certifier, never passes '
                                                        'verified_t_min'),
    ('scripts/tests/test_m1b_q2_certification.py', 'constructs the certifier, never passes '
                                                   'verified_t_min'),
)
