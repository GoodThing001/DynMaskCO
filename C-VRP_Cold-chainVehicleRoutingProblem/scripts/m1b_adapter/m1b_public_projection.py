#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-TrustFix F-T3: the TRUSTED STRICT-ONLINE DATA PROJECTION.

The Q2-Fix-T audit found that the pre-fix providers took `served_nodes`, `revealed_nodes` and
`source_identity` as FREE TEXT supplied by the caller, never consulted the environment, and kept
the whole node truth table inside a readable and MUTABLE dict.  A caller could therefore declare
any node visible -- even one the environment says is unrevealed -- and read its true attributes
(T3-2, T3-6), and could mutate the tables after construction (T3-3, T3-4).

This module is the ONE construction boundary allowed to read the environment's reveal state.
It derives the visible set ITSELF, copies ONLY the fields actually needed for the revealed nodes,
freezes the result, and binds an identity to (instance, event, visible set, contract, snapshot).

What this does and does not claim
    CLAIMED:    within the strict-online SOFTWARE protocol, an algorithm cannot obtain an
                unrevealed order's attributes through this projection, and cannot widen the
                visible set by declaring it wider.
    NOT CLAIMED: cryptographic isolation against arbitrary malicious code inside the same
                Python process.  This is a protocol boundary, not a security sandbox.

Design rules enforced here
  * the visible set comes from `env.reveal_time` and the CURRENT clock only;
  * only a whitelist of fields is copied (`temp_class`, `demand`, `tw_start`, `tw_end`,
    `service_time`, `initial_quality`) and ONLY for revealed nodes;
  * no whole-node table survives inside the object: internal mappings contain revealed nodes
    exclusively;
  * the internal mappings are wrapped in `MappingProxyType`, so post-construction mutation
    raises instead of silently changing what was certified;
  * `source_identity` is COMPUTED from the bound facts, never accepted as free text, so a
    caller cannot assert a source it does not have;
  * a new reveal requires a NEW projection; the old one stays valid for its own event.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from types import MappingProxyType

import m1b_quality_certificate as QC
import m1b_timing as TM

PROJECTION_OK = 'PROJECTION_OK'
PROJECTION_ENV_MISSING = 'PROJECTION_ENV_MISSING'
PROJECTION_FIELD_MISSING = 'PROJECTION_FIELD_MISSING'
PROJECTION_FIELD_INVALID = 'PROJECTION_FIELD_INVALID'
PROJECTION_DECLARED_SET_REJECTED = 'PROJECTION_DECLARED_SET_REJECTED'
PROJECTION_IMMUTABLE = 'PROJECTION_IMMUTABLE'
PROJECTION_NOT_FACTORY_BOUND = 'PROJECTION_NOT_FACTORY_BOUND'
PROJECTION_TOKEN_INVALID = 'PROJECTION_TOKEN_INVALID'
PROJECTION_UNBOUND = 'PROJECTION_UNBOUND'
PROJECTION_CHALLENGE_MISMATCH = 'PROJECTION_CHALLENGE_MISMATCH'
PROJECTION_VISIBLE_SET_NOT_REVEALED = 'PROJECTION_VISIBLE_SET_NOT_REVEALED'
PROJECTION_DATA_NOT_TRUSTED = 'PROJECTION_DATA_NOT_TRUSTED'

#: the ONLY order-attribute fields this projection may copy.
ALLOWED_ORDER_FIELDS = ('temp_class', 'demand', 'tw_start', 'tw_end', 'service_time',
                        'initial_quality')

IMPLEMENTATION_VERSION = 'M1B_Q2ENTRYFIX'

# ---------------------------------------------------------------------------------------------
# The unforgeable construction token.
#
# Q2-Fix-T and the final-entry audit both showed that a caller could build a projection object
# DIRECTLY and have the certifier accept it.  The token below is created at module import and is
# never exported: only the factory functions in THIS module can mint a certifiable projection.
# A caller that constructs `TrustedPublicProjection` by hand cannot obtain a valid token, so the
# certifier refuses it (PROJECTION_TOKEN_INVALID).
#
# Honest scope: within one Python process a determined caller could still reach the private
# symbol.  This is a PROTOCOL boundary that removes accidental and casual forgery -- it is not
# cryptographic isolation, and it is not claimed to be.
# ---------------------------------------------------------------------------------------------
_FACTORY_TOKEN = object()
#: ids of projections actually issued by the trusted factory (weak, in-process only)
_ISSUED = set()
_TOKEN_ISSUED_BY = {}


def _new_token():
    return object()


def factory_available():
    return True


def compute_source_identity(*, inst_idx, event_id, clock, revealed_nodes, contract_hash,
                            snapshot_identity):
    """The identity is COMPUTED from bound facts.  It is not a caller-supplied string."""
    payload = {'inst_idx': int(inst_idx), 'event_id': int(event_id),
               'clock': round(float(clock), 9),
               'revealed_nodes': sorted(int(x) for x in revealed_nodes),
               'contract_hash': contract_hash,
               'snapshot_identity': snapshot_identity,
               'schema': 'm1b_trusted_projection_v1',
               'implementation_version': IMPLEMENTATION_VERSION}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'source_identity': 'proj:' + hashlib.sha256(blob).hexdigest()[:32],
            'bound': payload}


def projection_challenge(projection):
    """A digest over the projection's TRUSTED FIELDS AND BINDINGS.

    The certifier recomputes this from the live object and refuses on a mismatch, so a trusted
    field cannot be replaced after construction (A2 in the previous audit was a hand-built object
    with `snapshot_identity = None`).
    """
    payload = {'source_identity': projection.source_identity,
               'instance_id': projection.instance_id,
               'event_id': projection.event_id,
               'clock': round(float(projection.clock), 9),
               'revealed_nodes': sorted(int(x) for x in projection.revealed_nodes),
               'contract_hash': projection.contract_hash,
               'snapshot_identity': projection.snapshot_identity,
               'iq': [[int(k), round(float(v), 12)] for k, v in sorted(projection._iq.items())],
               'tc': [[int(k), int(v)] for k, v in sorted(projection._tc.items())],
               'qty': [[int(k), round(float(v), 12)] for k, v in sorted(projection._qty.items())],
               'svc': [[int(k), round(float(v), 12)] for k, v in sorted(projection._svc.items())],
               'tws': [[int(k), round(float(v), 12)] for k, v in sorted(projection._tws.items())],
               'twe': [[int(k), round(float(v), 12)] for k, v in sorted(projection._twe.items())],
               'schema': 'm1b_projection_challenge_v1'}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


class TrustedPublicProjection(object):
    """An immutable, identity-bound view of the CURRENTLY REVEALED orders only.

    Satisfies BOTH consumer interfaces:
      * `QC` provider interface: `initial_quality_of` / `temp_class_of` / `quantity_of` /
        `identity()` / `served_nodes` / `source_identity`
      * `TM` time-input interface: `service_time_of` / `tw_start_of` / `tw_end_of` /
        `revealed_nodes` / `identity`
    """

    __slots__ = ('_iq', '_tc', '_qty', '_svc', '_tws', '_twe', 'revealed_nodes',
                 'instance_id', 'event_id', 'clock', 'contract_hash', 'snapshot_identity',
                 'source_identity', '_bound', '_identity_digest', '_factory_token',
                 '_data_trusted')

    def __init__(self, *, iq, tc, qty, svc, tws, twe, revealed_nodes, instance_id, event_id,
                 clock, contract_hash, snapshot_identity, _token=None, _data_trusted=False):
        # A projection that is not bound to a snapshot cannot be certified.  `None` is REFUSED
        # here rather than silently skipping the certifier's binding check.
        if snapshot_identity is None:
            raise ValueError(
                '%s: snapshot_identity is required; an unbound projection must not be able to '
                'skip the binding check' % PROJECTION_UNBOUND)
        rev = tuple(sorted(int(x) for x in revealed_nodes))
        # every retained mapping contains REVEALED nodes ONLY -- there is no whole-node table
        self._iq = MappingProxyType({int(k): float(v) for k, v in iq.items()})
        self._tc = MappingProxyType({int(k): int(v) for k, v in tc.items()})
        self._qty = MappingProxyType({int(k): float(v) for k, v in qty.items()})
        self._svc = MappingProxyType({int(k): float(v) for k, v in svc.items()})
        self._tws = MappingProxyType({int(k): float(v) for k, v in tws.items()})
        self._twe = MappingProxyType({int(k): float(v) for k, v in twe.items()})
        self.revealed_nodes = frozenset(rev)
        self.instance_id = int(instance_id)
        self.event_id = int(event_id)
        self.clock = float(clock)
        self.contract_hash = contract_hash
        self.snapshot_identity = snapshot_identity
        ident = compute_source_identity(inst_idx=instance_id, event_id=event_id, clock=clock,
                                        revealed_nodes=rev, contract_hash=contract_hash,
                                        snapshot_identity=snapshot_identity)
        self.source_identity = ident['source_identity']
        self._bound = MappingProxyType(ident['bound'])
        self._factory_token = _token
        self._data_trusted = bool(_data_trusted)
        self._identity_digest = hashlib.sha256(
            json.dumps(ident['bound'], sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()

    def challenge(self):
        """Digest of the trusted fields and bindings, for post-construction tamper detection."""
        return projection_challenge(self)

    # ------------------------------------------------------------------ gating
    @property
    def served_nodes(self):
        return self.revealed_nodes

    def _gate(self, node):
        if int(node) not in self.revealed_nodes:
            raise PermissionError('PROJECTION_NODE_NOT_REVEALED:%d' % int(node))

    def _need(self, table, node, what):
        self._gate(node)
        n = int(node)
        if n not in table:
            raise KeyError('PROJECTION_FIELD_MISSING:%s:%d' % (what, n))
        return table[n]

    # ------------------------------------------------------------------ QC consumer interface
    def initial_quality_of(self, node):
        return self._need(self._iq, node, 'initial_quality')

    def temp_class_of(self, node):
        return self._need(self._tc, node, 'temp_class')

    def quantity_of(self, node):
        return self._need(self._qty, node, 'quantity')

    # ------------------------------------------------------------------ TM consumer interface
    def service_time_of(self, node):
        return self._need(self._svc, node, 'service_time')

    def tw_start_of(self, node):
        return self._need(self._tws, node, 'tw_start')

    def tw_end_of(self, node):
        return self._need(self._twe, node, 'tw_end')

    def identity(self):
        return {'source_identity': self.source_identity,
                'identity_digest': self._identity_digest,
                'bound': dict(self._bound),
                'n_served_nodes': len(self.revealed_nodes),
                'contract_hash': self.contract_hash,
                'snapshot_identity': self.snapshot_identity}

    def assert_same_identity(self, other):
        """Pre/post-certification identity consistency check."""
        if not isinstance(other, TrustedPublicProjection):
            return False, 'NOT_A_PROJECTION', {}
        same = (self._identity_digest == other._identity_digest)
        return same, ('PROJECTION_IDENTITY_STABLE' if same
                      else 'PROJECTION_IDENTITY_CHANGED'), {
            'before': self.source_identity, 'after': other.source_identity}


def build_projection_from_env(env, *, inst_idx, event_id, clock, contract,
                              snapshot_identity, fields=None, _require_token=True):
    """THE construction boundary.  Returns (projection, verdict, detail).

    `fields` supplies the raw per-node arrays/dicts.  Only `ALLOWED_ORDER_FIELDS` are read, and
    only for nodes that `env.reveal_time` says are revealed at `clock`.
    """
    if env is None:
        return None, PROJECTION_ENV_MISSING, {'field': 'env'}
    if contract is None:
        return None, PROJECTION_FIELD_MISSING, {'field': 'contract'}
    if snapshot_identity is None:
        return None, PROJECTION_FIELD_MISSING, {'field': 'snapshot_identity'}
    fields = dict(fields or {})
    try:
        import numpy as np
        reveal_row = env.reveal_time[int(inst_idx)]
    except Exception as exc:                                            # noqa: BLE001
        return None, PROJECTION_ENV_MISSING, {'exception': '%s' % type(exc).__name__}
    n = int(getattr(env, 'num_nodes', len(reveal_row)))
    revealed = [c for c in range(1, n)
                if bool(np.asarray(reveal_row)[c] <= float(clock) + 1e-6)]
    if not revealed:
        revealed = []

    def _val(name, node):
        src = fields.get(name)
        if src is None:
            return None
        try:
            v = src[int(node)]
        except Exception:                                                # noqa: BLE001
            try:
                v = src[int(node)] if not hasattr(src, 'shape') else None
            except Exception:                                            # noqa: BLE001
                return None
        try:
            import numpy as np
            if hasattr(v, 'item'):
                v = v.item()
        except Exception:                                                # noqa: BLE001
            pass
        return v

    missing = []
    iq, tc, qty, svc, tws, twe = {}, {}, {}, {}, {}, {}
    for node in revealed:
        for name, table, conv in (('temp_class', tc, int), ('demand', qty, float),
                                  ('tw_start', tws, float), ('tw_end', twe, float),
                                  ('service_time', svc, float),
                                  ('initial_quality', iq, float)):
            v = _val(name, node)
            if v is None:
                missing.append({'node': int(node), 'field': name})
                continue
            try:
                f = conv(v)
            except Exception:                                            # noqa: BLE001
                return None, PROJECTION_FIELD_INVALID, {'node': int(node), 'field': name,
                                                        'value': repr(v)}
            if f != f:
                return None, PROJECTION_FIELD_INVALID, {'node': int(node), 'field': name,
                                                        'value': 'NaN'}
            table[int(node)] = f
    contract_hash = getattr(contract, 'contract_hash', None)
    if snapshot_identity is None:
        return None, PROJECTION_UNBOUND, {
            'why': 'the projection must be bound to the snapshot it certifies'}
    proj = TrustedPublicProjection(iq=iq, tc=tc, qty=qty, svc=svc, tws=tws, twe=twe,
                                   revealed_nodes=revealed, instance_id=inst_idx,
                                   event_id=event_id, clock=clock, contract_hash=contract_hash,
                                   snapshot_identity=snapshot_identity, _token=_FACTORY_TOKEN,
                                   _data_trusted=True)
    _ISSUED.add(id(proj))
    detail = {'n_revealed': len(revealed), 'revealed_nodes': revealed,
              'fields_copied': [f for f in ALLOWED_ORDER_FIELDS],
              'missing_fields': missing[:20],
              'source_identity': proj.source_identity,
              'contract_hash': contract_hash}
    return proj, PROJECTION_OK, detail


def projection_from_revealed_maps(*, iq, tc, qty, svc, tws, twe, revealed_nodes, instance_id=-1,
                                  event_id=-1, clock=0.0, contract_hash=None,
                                  snapshot_identity=None):
    """DIAGNOSTIC-ONLY constructor.  It CANNOT produce a certifiable projection.

    The final-entry audit found that this helper let a caller create an object the production gate
    accepted (by passing `snapshot_identity = None`, which skipped the binding check).  The helper
    is retained because many tests and diagnostics need a projection-shaped object, but the object
    it returns is stamped `data_trusted = False` AND carries a token the certifier REFUSES, so it
    can never be used for certification.
    """
    proj = TrustedPublicProjection(iq=iq, tc=tc, qty=qty, svc=svc, tws=tws, twe=twe,
                                   revealed_nodes=revealed_nodes, instance_id=instance_id,
                                   event_id=event_id, clock=clock, contract_hash=contract_hash,
                                   snapshot_identity=('DIAGNOSTIC-UNBOUND'
                                                      if snapshot_identity is None
                                                      else snapshot_identity),
                                   _token=object(), _data_trusted=False)
    return proj


def is_certifiable(projection):
    """(ok, verdict, detail).  The certifier's provenance test.

    Certifiable requires BOTH the factory token AND `data_trusted`, plus a real snapshot binding.
    """
    if not isinstance(projection, TrustedPublicProjection):
        return False, PROJECTION_DECLARED_SET_REJECTED, {'why': 'not a TrustedPublicProjection'}
    if (getattr(projection, '_factory_token', None) is not _FACTORY_TOKEN
            or id(projection) not in _ISSUED):
        return False, PROJECTION_TOKEN_INVALID, {
            'why': 'the projection was not produced by the trusted construction boundary '
                   '(build_projection_from_env); a diagnostic or hand-built object carries a '
                   'token the certifier refuses'}
    if getattr(projection, '_data_trusted', False) is not True:
        return False, PROJECTION_DATA_NOT_TRUSTED, {
            'why': 'the projection is flagged as diagnostic data'}
    if projection.snapshot_identity in (None, 'DIAGNOSTIC-UNBOUND'):
        return False, PROJECTION_UNBOUND, {'why': 'no usable snapshot binding'}
    return True, PROJECTION_OK, {}


def assert_visible_set_matches_env(projection, env, *, inst_idx, clock):
    """Independently re-verify the visible set against the environment (requirement A).

    The construction boundary is trusted for what it COPIED; this check is the certifier's own,
    so a forged projection whose declared set exceeds what the environment reveals is refused
    even if it somehow carried a valid token.
    """
    if env is None:
        return False, PROJECTION_ENV_MISSING, {'why': 'the certifier needs the environment to '
                                                       'verify the visible set'}
    try:
        import numpy as np
        row = np.asarray(env.reveal_time[int(inst_idx)])
    except Exception as exc:                                            # noqa: BLE001
        return False, PROJECTION_ENV_MISSING, {'exception': type(exc).__name__}
    excess = [int(n) for n in sorted(projection.revealed_nodes)
              if not bool(row[int(n)] <= float(clock) + 1e-6)]
    if excess:
        return False, PROJECTION_VISIBLE_SET_NOT_REVEALED, {
            'nodes_declared_visible_but_not_revealed': excess[:20],
            'clock': float(clock)}
    return True, PROJECTION_OK, {'n_verified': len(projection.revealed_nodes)}


def reject_declared_set(*, declared_served, declared_revealed, required):
    """An EXTERNALLY DECLARED visible set must not grant itself trust (T3-2 / T3-6).

    Returns (ok, verdict, detail).  A declaration may only NARROW what the trusted projection
    already allows; declaring anything the projection has not revealed is refused.
    """
    trusted = frozenset(int(x) for x in (required or ()))
    problems = []
    for label, declared in (('served_nodes', declared_served), ('revealed_nodes', declared_revealed)):
        for n in (declared or ()):
            if int(n) not in trusted:
                problems.append({'declaration': label, 'node': int(n),
                                 'why': 'declared visible but NOT revealed by the trusted '
                                        'projection at this clock'})
    if problems:
        return False, PROJECTION_DECLARED_SET_REJECTED, {'violations': problems[:20],
                                                         'trusted_revealed': sorted(trusted)}
    return True, PROJECTION_OK, {'trusted_revealed': sorted(trusted)}
