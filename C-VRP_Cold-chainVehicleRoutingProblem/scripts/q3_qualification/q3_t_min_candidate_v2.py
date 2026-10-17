#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-TMIN MINIMAL SAFE FIX -- CANDIDATE V2 with STRICT REFUSAL SEMANTICS.

STATUS: `INSTALLED = False`.  This module is NOT imported by any production path, does NOT subclass
or monkey-patch `m1b_q2_certifier.py`, and does NOT write any receipt.  Installing it requires a
separate authorisation; the minimal production diff is submitted for review as
`results/_q3_next_20261017/gtmin/minimal_patch_for_authorisation.diff` (NOT APPLIED).

WHAT CHANGED FROM CANDIDATE V1 (same defect, stricter failure semantics)
=======================================================================
V1 fell back to `t_min = 0` for an untrusted `verified_t_min` and RECORDED that it had refused.  The
number was therefore conservative, but a caller reading only `composite_pass` could still see "PASS"
and conclude the certification was fine.  The reviewer's requirement is stronger:

    an untrusted input must make THIS certification REFUSED, not merely numerically conservative.

V2 therefore separates two things that V1 merged:
    * `t_min`            -> always 0.0 this round (no trusted witness mechanism exists yet);
    * `refused`          -> True whenever `verified_t_min is not None`, i.e. whenever the caller
                            supplied a claim the code cannot check.
and `apply_to_receipt()` stamps `composite_pass = False` / `final = ..._ENGINEERING_REFUSED`.

THE DEFECT (unchanged, reproduced here so the candidate is self-documenting)
==========================================================================
`m1b_q2_certifier.py:122-139` accepts any `verified_t_min` whenever ANY `env` object was supplied,
labels it `VERIFIED_AGAINST_SUPPLIED_ENV` and RETURNS IT, without reading a single `env` field:

    verified_t_min=1e6 + env   -> t_min=1e6, H_max=0,        q_lb=8.999999999
    verified_t_min=24  + env   -> t_min=24,  H_max=1.0e-06,  q_lb=8.99999998017
    verified_t_min="1e6" + env -> t_min=1e6   (a STRING is accepted)
    verified_t_min=True  + env -> t_min=1.0   (a BOOL is accepted)
    baseline                   -> t_min=0,   H_max=24.000001, q_lb=8.54807998017

`q_lb` is the certified quality lower bound, so a larger `t_min` makes the qualification LESS
conservative.  `env = object()` is a sufficient "witness".

DIRECTION SAFETY (must accompany any proposal to install this)
=============================================================
`H_max = T_end + epsilon_return - t_min` and `q_lb = max(0, q0_safe - k*H_max)`, so
`dq_lb/dt_min = +k > 0`.  V2 can only set `t_min = 0` (never above), which can only increase `H_max`,
which can only decrease `q_lb`, which can only turn a PASS into a REFUSAL -- and additionally marks
the receipt REFUSED.  It can never admit anything the pre-fix code rejected.

OUT OF SCOPE THIS ROUND (declared in the pre-registration)
==========================================================
No trusted-witness mechanism for a non-zero `t_min`, no production edit, no install wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INSTALLED = False
CANDIDATE_NOT_INSTALLED = True
IMPLEMENTATION_VERSION = 'Q3_G_TMIN_CANDIDATE_V2'
PREREGISTRATION = ('scripts/q3_qualification/q3_gtmin_preregistration_2026-10-17.md')

#: The ONLY label a trusted clock origin may carry this round.
CONSERVATIVE_CLOCK_ORIGIN = 'CONSERVATIVE_CLOCK_ORIGIN'
#: Any caller-supplied claim that cannot be checked gets this label -- never a "VERIFIED_*" one.
UNVERIFIED_T_MIN_REFUSED = 'UNVERIFIED_T_MIN_REFUSED'

#: F-TMIN-2: the pre-fix label exists only as a FORBIDDEN token, so it cannot be re-used and any
#: receipt still carrying it is refused.
FORBIDDEN_RECEIPT_LABELS = ('VERIFIED_AGAINST_SUPPLIED_ENV',)

#: Mirrors the production constant so the candidate's stamped receipt is byte-compatible.
ENGINEERING_REFUSED_LABEL = 'M1B_Q2TRUSTFIX_ENGINEERING_REFUSED'

#: The value that can never be wrong: the clock origin.  No other value is accepted this round.
ACCEPTED_T_MIN = 0.0


@dataclass(frozen=True)
class TMinResolution(object):
    """The resolved value AND the decision about the certification it belongs to."""

    t_min: float
    source: str
    refused: bool
    reason: str
    received_repr: str = ''
    env_received: bool = False
    env_consulted: bool = False                 # always False: F-TMIN-3
    bare_t_min_ignored: bool = False
    quality_effect: str = 'NONE'                # the value never feeds q_lb differently
    extra: dict = field(default_factory=dict)

    def receipt_gate(self):
        """The gate entry that must be written into the certification receipt."""
        return {
            'verdict': self.source,
            'detail': {
                't_min_used': float(self.t_min),
                'refused': bool(self.refused),
                'reason': self.reason,
                'value_received': self.received_repr,
                'env_received': bool(self.env_received),
                'env_consulted': bool(self.env_consulted),
                'bare_t_min_ignored': bool(self.bare_t_min_ignored),
                'quality_effect': self.quality_effect,
                'implementation': IMPLEMENTATION_VERSION,
                'preregistration': PREREGISTRATION,
            },
        }

    def as_dict(self):
        out = self.receipt_gate()
        out['gate'] = out.pop('verdict')
        return out


def resolve_t_min(*, verified_t_min=None, env=None, t_min=None):
    """Fail-closed, three-state resolution WITH a refusal flag.

    The returned `t_min` is always 0.0 this round.  `refused` is True whenever the caller supplied a
    `verified_t_min` claim, because no witness mechanism exists to check it -- the number alone is not
    the deliverable, the refusal is.

    `env` is accepted for signature compatibility and is NEVER inspected (F-TMIN-3): only its
    presence is recorded.
    """
    env_received = env is not None
    bare_ignored = t_min is not None and float_safe(t_min) is not None
    if verified_t_min is None:
        return TMinResolution(
            t_min=ACCEPTED_T_MIN, source=CONSERVATIVE_CLOCK_ORIGIN, refused=False,
            reason=('no verified_t_min was supplied; the conservative clock origin is the only '
                    'accepted value and the certification is not affected'),
            received_repr='', env_received=env_received, env_consulted=False,
            bare_t_min_ignored=bare_ignored,
            extra={'bare_t_min_value': (None if not bare_ignored else repr(t_min))})
    return TMinResolution(
        t_min=ACCEPTED_T_MIN, source=UNVERIFIED_T_MIN_REFUSED, refused=True,
        reason=('a verified_t_min was supplied without a recomputable witness.  The pre-fix resolver '
                'returned it verbatim and labelled it VERIFIED_AGAINST_SUPPLIED_ENV without reading '
                'env.  A non-zero t_min cannot be established this round, so the value is rejected '
                'AND this certification is REFUSED -- it is not merely computed conservatively'),
        received_repr=repr(verified_t_min), env_received=env_received, env_consulted=False,
        bare_t_min_ignored=bare_ignored,
        extra={'witness_mechanism': 'NOT_IMPLEMENTED_THIS_ROUND'})


def float_safe(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return None
    return v if (v == v and v not in (float('inf'), float('-inf'))) else None


# ---------------------------------------------------------------------------------------------
# receipt semantics
# ---------------------------------------------------------------------------------------------
def apply_to_receipt(receipt, resolution):
    """Stamp the resolution into a certification receipt and FORCE a refusal when it refused.

    This is the whole point of V2: a refused `t_min` input cannot leave a receipt that still reads
    `composite_pass = True`.
    """
    if not isinstance(receipt, dict):
        raise TypeError('receipt must be a dict')
    receipt.setdefault('gates', {})['t_min'] = resolution.receipt_gate()
    receipt['t_min_used'] = float(resolution.t_min)
    receipt['t_min_refused'] = bool(resolution.refused)
    if resolution.refused:
        receipt['composite_pass'] = False
        receipt['final'] = ENGINEERING_REFUSED_LABEL
        receipt['refused_because'] = UNVERIFIED_T_MIN_REFUSED
        receipt.setdefault('non_substitutable', {})['t_min_gate'] = (
            'an untrusted verified_t_min REFUSES the certification; the conservative number is not a '
            'substitute for a passing judgement')
    return receipt


def receipt_gate(receipt):
    """F-TMIN-2 / C4: any receipt carrying a forbidden label, or a refused t_min, is refused."""
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
    refused_flag = bool(receipt.get('t_min_refused')) if isinstance(receipt, dict) else False
    return (not found and not refused_flag), {'forbidden_label_paths': found,
                                              't_min_refused': refused_flag}


def composite_would_pass(receipt):
    """The single question that matters for C8: can any reader call this a PASS?"""
    ok, detail = receipt_gate(receipt)
    if not ok:
        return False, detail
    return bool(receipt.get('composite_pass')) and str(receipt.get('final', '')).endswith(
        'ENGINEERING_CERTIFIED'), detail


# ---------------------------------------------------------------------------------------------
# the pre-fix resolver, replicated ONLY so the comparison is against the real semantics
# ---------------------------------------------------------------------------------------------
def production_resolver_replica(*, verified_t_min=None, env=None):
    """Verbatim replication of `m1b_q2_certifier.py:129-139` (pre-fix).  Never used to decide."""
    source = CONSERVATIVE_CLOCK_ORIGIN
    if verified_t_min is None or env is None:
        return 0.0, source
    try:
        v = float(verified_t_min)
    except Exception:                                                     # noqa: BLE001
        return 0.0, source
    if v != v or v < 0 or v > 1e9:
        return 0.0, source
    return v, 'VERIFIED_AGAINST_SUPPLIED_ENV'


#: Historical call sites recorded by grep (see the report for the impact analysis).
AFFECTED_CALL_SITES = (
    ('scripts/m1b_adapter/m1b_q2_certifier.py',
     'M1bQ2TrustFixCertifier.__init__ -> _resolve_t_min (the defect)'),
    ('scripts/m1b_adapter/m1b_q2_certifier.py',
     'M1bQ2TrustFixTestOnlyCertifier inherits the same resolver'),
    ('scripts/m1b_adapter/m1b_q2_certifier.py:407',
     'certify() passes self.t_min into QC.certify_quality_lb'),
    ('scripts/tests/test_m1b_q2_final_entry_controls.py',
     'B2 passes the BARE t_min keyword, so verified_t_min is never exercised (coverage gap)'),
    ('scripts/tests/test_m1b_q2fixt_trust_boundary.py',
     'constructs the certifier; never passes verified_t_min'),
    ('scripts/tests/test_m1b_q2fix_certification.py',
     'constructs the certifier; never passes verified_t_min'),
    ('scripts/tests/test_m1b_q2r_negative_controls.py',
     'constructs the certifier; never passes verified_t_min'),
    ('scripts/tests/test_m1b_q2_certification.py',
     'constructs the certifier; never passes verified_t_min'),
)
