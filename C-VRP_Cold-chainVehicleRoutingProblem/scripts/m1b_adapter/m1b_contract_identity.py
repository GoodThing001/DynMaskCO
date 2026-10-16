#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-1: CONTRACT IDENTITY resolution and validation (fail-closed).

Fixes G2.  The M1-a adapter bound `contract_version` from
`getattr(contract, 'version', None)`, but NEITHER `ColdChainContract` nor
`ColdChainContractV2` has a `version` attribute -- only `schema_version`.  So the value was
ALWAYS None and the strict comparison could only ever return a mismatch.  This module replaces
that with the repository's EXISTING semantics (`d3_quality_projection._contract_version`):
`quality_config_class` + `contract_schema_version` + `is_v2`, and additionally binds the
`contract_hash`.

`getattr(contract, 'version', None)` MUST NOT appear in this module or its callers; a negative
test (T-Q2-6e) checks that by AST.

Fail-closed on: missing field, wrong type, non-v2 contract, or hash mismatch.
"""
from __future__ import annotations

CONTRACT_IDENTITY_OK = 'CONTRACT_IDENTITY_OK'
CONTRACT_NOT_V2 = 'CONTRACT_NOT_V2'
CONTRACT_FIELD_MISSING = 'CONTRACT_FIELD_MISSING'
CONTRACT_FIELD_TYPE_INVALID = 'CONTRACT_FIELD_TYPE_INVALID'
CONTRACT_HASH_MISSING = 'CONTRACT_HASH_MISSING'
CONTRACT_HASH_MISMATCH = 'CONTRACT_HASH_MISMATCH'
CONTRACT_QUALITY_CONFIG_UNSUPPORTED = 'CONTRACT_QUALITY_CONFIG_UNSUPPORTED'

#: the ONLY quality configuration this certificate is allowed to reason about.
REQUIRED_QUALITY_CONFIG_CLASS = 'QualityConfigV2'
REQUIRED_SCHEMA_VERSION = 'coldchain-contract-v2'
CERTIFICATE_PROOF_VERSION = 'm1b_q2_quality_lb_v2'


def resolve_contract_identity(contract):
    """Return the identity dict using the repository's existing semantics.

    Never guesses: an absent or ill-typed field is reported, not defaulted.
    """
    if contract is None:
        return None, CONTRACT_FIELD_MISSING, {'field': 'contract'}
    quality = getattr(contract, 'quality', None)
    if quality is None:
        return None, CONTRACT_FIELD_MISSING, {'field': 'contract.quality'}
    cls = type(quality).__name__
    schema = getattr(contract, 'schema_version', None)
    chash = getattr(contract, 'contract_hash', None)
    if schema is None:
        return None, CONTRACT_FIELD_MISSING, {'field': 'contract.schema_version'}
    if not isinstance(schema, str) or not schema:
        return None, CONTRACT_FIELD_TYPE_INVALID, {'field': 'contract.schema_version',
                                                  'got': repr(schema)}
    if chash is None:
        return None, CONTRACT_HASH_MISSING, {'field': 'contract.contract_hash'}
    if not isinstance(chash, str) or not chash:
        return None, CONTRACT_FIELD_TYPE_INVALID, {'field': 'contract.contract_hash',
                                                   'got': repr(chash)}
    return {'quality_config_class': cls, 'contract_schema_version': schema,
            'is_v2': cls == REQUIRED_QUALITY_CONFIG_CLASS,
            'contract_hash': chash}, CONTRACT_IDENTITY_OK, {}


def validate_contract_for_quality_certificate(contract, *, registered_contract_hash=None):
    """Full gate.  Returns (identity, verdict, detail).

    Verdict is CONTRACT_IDENTITY_OK only when EVERY condition holds:
      * the identity resolves with every field present and well typed;
      * `is_v2` is True (a v1 contract is REFUSED -- never silently fall back to v1);
      * `contract_schema_version` equals the registered v2 schema string;
      * `contract_hash` matches the registered hash when one is supplied.
    """
    ident, verdict, detail = resolve_contract_identity(contract)
    if verdict != CONTRACT_IDENTITY_OK:
        return None, verdict, detail
    if ident['is_v2'] is not True:
        return ident, CONTRACT_NOT_V2, {
            'quality_config_class': ident['quality_config_class'],
            'why': 'this certificate only covers v2 zero-order absolute-threshold classes; '
                   'v1 uses Arrhenius rates and RATIO thresholds, so a v1 contract must be '
                   'refused rather than silently substituted'}
    if ident['contract_schema_version'] != REQUIRED_SCHEMA_VERSION:
        return ident, CONTRACT_FIELD_TYPE_INVALID, {
            'field': 'contract.schema_version', 'expected': REQUIRED_SCHEMA_VERSION,
            'got': ident['contract_schema_version']}
    if registered_contract_hash is not None and \
            ident['contract_hash'] != registered_contract_hash:
        return ident, CONTRACT_HASH_MISMATCH, {
            'declared': ident['contract_hash'], 'registered': registered_contract_hash}
    return ident, CONTRACT_IDENTITY_OK, {}


def check_quality_classes_are_certifiable(contract, *, epsilon_init):
    """Q2-5 applicability: only v2 ZERO-order + ABSOLUTE-threshold classes are covered.

    Returns (per_class, verdict, detail).  `per_class[c]` is a dict with the constants the
    certificate needs, or None when the class is out of scope.  Class 0 (tomato, first order,
    threshold_type 'none') is reported as out of scope and MUST NOT be treated as unsalable.
    """
    from coldchain_contract import QualityConfigV2                    # local import: read-only
    q = getattr(contract, 'quality', None)
    if not isinstance(q, QualityConfigV2):
        return {}, CONTRACT_QUALITY_CONFIG_UNSUPPORTED, {
            'why': 'quality config is not QualityConfigV2'}
    n = len(q.dynamics_order)
    per_class = {}
    for c in range(n):
        order = q.dynamics_order[c]
        ttype = q.threshold_type[c]
        if order == 'zero' and ttype == 'absolute':
            vals = {'in_scope': True, 'dynamics_order': order, 'threshold_type': ttype,
                    'initial_value': float(q.initial_value[c]),
                    'rate_per_hour': float(q.rate_per_hour[c]),
                    'threshold': float(q.threshold[c]),
                    'q0_safe': float(q.initial_value[c]) - float(epsilon_init)}
            for k, v in vals.items():
                if isinstance(v, float) and (v != v or v in (float('inf'), float('-inf'))):
                    return per_class, CONTRACT_FIELD_TYPE_INVALID, {
                        'field': 'quality[%d].%s' % (c, k), 'got': repr(v)}
            if vals['rate_per_hour'] < 0:
                return per_class, CONTRACT_FIELD_TYPE_INVALID, {
                    'field': 'quality[%d].rate_per_hour' % c, 'got': vals['rate_per_hour']}
            per_class[c] = vals
        else:
            per_class[c] = {'in_scope': False, 'dynamics_order': order,
                            'threshold_type': ttype,
                            'why': 'certificate covers zero-order + absolute-threshold only'}
    return per_class, CONTRACT_IDENTITY_OK, {}
