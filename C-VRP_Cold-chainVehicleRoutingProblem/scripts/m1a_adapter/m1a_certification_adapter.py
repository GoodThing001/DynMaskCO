#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-a: the FAIL-CLOSED L2 CERTIFICATION ADAPTER (prototype).

SCOPE AND STATUS
----------------
This module is the M1-a PROTOTYPE.  It is NOT a production entry point, it is NOT registered on
any environment, and it does NOT install anything.

Its status can be AT MOST `M1A_SYNTHETIC_ENGINEERING_ACCEPTED` -- an ENGINEERING state meaning
"the interface and the rejection mechanisms hold under isolated synthetic tests".  It is NOT
`FULL_INSTALLABILITY_ESTABLISHED`, NOT `L2_PASS`, NOT `A3_PASS`.

With no audited real quality implementation on the registry, `quality.state` REMAINS
`QUALITY_CAPABILITY_GAP` and full installability REMAINS unavailable, EVEN IF every synthetic
test passes.  A test callback's success must never be read as real quality capability.

WHY THIS MODULE EXISTS (three source facts, each independently verified)
-----------------------------------------------------------------------
F1  The composer's FORMAL quality branch is NOT gated by the legality gate.
    `d3_composed_certifier.py`: the `if legality_ok:` block runs `:206`-`:277`; the formal branch
    `if quality_check_fn is None:` / `else:` runs `:283`-`:295` at the SAME indent level (4
    spaces), i.e. OUTSIDE it.  So an injected `quality_check_fn` is called at `:291` EVEN WHEN
    the plan is illegal (future / already-served / duplicate orders).  The module's own `note`
    at `:307-309` claims the quality component is "NOT RUN" on an illegal order set; that holds
    for `quality_projection` (`:246`, inside the block) but NOT for the formal branch.
    => This adapter therefore runs STAGE 1 with `quality_check_fn=None`, which routes around
    `:291` entirely, and enforces legality itself before any quality implementation is called.
    The composer defect is NOT fixed here and must not be reported as fixed.

F2  `blocking_gaps = [] if qok else []` (`:293`) clears the blocking gaps UNCONDITIONALLY --
    both branches are the empty list.  The behavioural fail-closed property survives via the
    three-state assignment (`:292`) and `reasons.append('quality_failed')` (`:294-295`), but the
    TYPE of `qok` is never checked.  A truthy non-bool (`"no"`, `[0]`, `{...}`, `1`) is read as a
    pass.  => Stage 2 therefore performs STRICT type checking itself (T1) instead of relying on
    the composer.

F3  The composer has NO quality-implementation registry (no such parameter; `:291` calls whatever
    callable it is handed without checking implementation identity, contract version or
    applicable domain) and it passes the callback the RAW `env`
    (`quality_check_fn(env, inst_idx, clock, vehicles, served_mask, plans)` at `:291`).
    => Registry admission and the applicable-domain check live HERE, and the quality
    implementation is called through a MINIMAL-VISIBILITY view that does not expose the raw env.

FIRST-STAGE PASS CONDITION (explicit, because it is easy to get wrong)
---------------------------------------------------------------------
Stage 1 calls the composer with `quality_check_fn=None`, so the composer ALWAYS reports
`quality.state == QUALITY_CAPABILITY_GAP` and `installable is False`, even when every executable
check passes.  Stage 1 therefore MUST NOT be judged by `rep['installable']`.  It is judged by an
INDEPENDENT stage-1 predicate over the required legality gates, `repair_plan`, `full_partition`,
`route_feasible`, `energy_budget` and the report structure.

QUALITY PROJECTION RULING (pre-committed, see `project_quality_stage1`)
----------------------------------------------------------------------
Stage 1 passes `project_quality=False`.  Rationale: with `project_quality=True` the composer
appends `quality_projection_failed` to `reasons` at `:268-269`, which makes a projection whose
conservativeness is NOT established behave as a HARD gate on stage 1.  That would silently
promote an unproven projection into part of the certification.  The projection is therefore NOT
part of stage 1's verdict; a caller that wants it may run the composer separately and treat it
as DIAGNOSTIC ONLY.  This is a pre-committed ruling, not an incidental default.
"""
from __future__ import annotations

import hashlib
import json

# --------------------------------------------------------------------------------------------
# Verdict vocabularies.  Pre-registered: a verdict not in these tables is a CODE ERROR.
# --------------------------------------------------------------------------------------------
STAGE1_PASS = 'STAGE1_PASS'
STAGE1_MISSING_BASE_PLANS = 'STAGE1_MISSING_BASE_PLANS'
STAGE1_LEGALITY_GATE_FAILED = 'STAGE1_LEGALITY_GATE_FAILED'
STAGE1_REPAIR_PLAN_FAILED = 'STAGE1_REPAIR_PLAN_FAILED'
STAGE1_FULL_PARTITION_FAILED = 'STAGE1_FULL_PARTITION_FAILED'
STAGE1_ROUTE_INFEASIBLE = 'STAGE1_ROUTE_INFEASIBLE'
STAGE1_ENERGY_BUDGET_EXCEEDED = 'STAGE1_ENERGY_BUDGET_EXCEEDED'
STAGE1_REPORT_SCHEMA_INVALID = 'STAGE1_REPORT_SCHEMA_INVALID'
STAGE1_COMPOSER_EXCEPTION = 'STAGE1_COMPOSER_EXCEPTION'
STAGE1_MUTABLE_SCOPE_VIOLATION = 'STAGE1_MUTABLE_SCOPE_VIOLATION'
STAGE1_P0_PROVENANCE_INVALID = 'STAGE1_P0_PROVENANCE_INVALID'

QUALITY_RESULT_NOT_A_PAIR = 'QUALITY_RESULT_NOT_A_PAIR'
QUALITY_QOK_NOT_BUILTIN_BOOL = 'QUALITY_QOK_NOT_BUILTIN_BOOL'
QUALITY_DETAIL_NOT_A_MAPPING = 'QUALITY_DETAIL_NOT_A_MAPPING'
QUALITY_EVIDENCE_MISSING = 'QUALITY_EVIDENCE_MISSING'
QUALITY_IMPL_IDENTITY_MISMATCH = 'QUALITY_IMPL_IDENTITY_MISMATCH'
QUALITY_CONTRACT_VERSION_MISMATCH = 'QUALITY_CONTRACT_VERSION_MISMATCH'
QUALITY_DOMAIN_IDENTITY_MISMATCH = 'QUALITY_DOMAIN_IDENTITY_MISMATCH'
QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED = 'QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED'
QUALITY_READ_UNREVEALED_ORDER = 'QUALITY_READ_UNREVEALED_ORDER'
QUALITY_CHECKED_FAILED = 'QUALITY_CHECKED_FAILED'
QUALITY_CERTIFIED = 'QUALITY_CERTIFIED'

TEST_ONLY_REGISTRY_REFUSED = 'TEST_ONLY_REGISTRY_REFUSED'
UNCERTIFIED_IMPL_REFUSED = 'UNCERTIFIED_IMPL_REFUSED'
NO_OMEGA_DECLARATION_REFUSED = 'NO_OMEGA_DECLARATION_REFUSED'
IMPL_SHA_MISMATCH_REFUSED = 'IMPL_SHA_MISMATCH_REFUSED'
REGISTRY_NOT_SUPPLIED = 'REGISTRY_NOT_SUPPLIED'

S12_PROVENANCE_ISOLATED = 'S12_PROVENANCE_ISOLATED'
S12_BASE_IS_MISSING = 'S12_BASE_IS_MISSING'
S12_BASE_IS_CANDIDATE_ALIAS = 'S12_BASE_IS_CANDIDATE_ALIAS'
S12_SHARED_MUTABLE_STORAGE = 'S12_SHARED_MUTABLE_STORAGE'
S12_BASE_NOT_FROM_FROZEN_SOURCE = 'S12_BASE_NOT_FROM_FROZEN_SOURCE'
S12_BASE_SHA_MISMATCH = 'S12_BASE_SHA_MISMATCH'
S12_CANDIDATE_PATH_MUTATED_P0 = 'S12_CANDIDATE_PATH_MUTATED_P0'

FINAL_INSTALLABLE = 'FINAL_INSTALLABLE'
FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE = 'FULL_INSTALLABILITY_NOT_AVAILABLE'
FINAL_STAGE1_REFUSED = 'FINAL_STAGE1_REFUSED'
FINAL_QUALITY_REFUSED = 'FINAL_QUALITY_REFUSED'
FINAL_PLAN_IDENTITY_MISMATCH = 'FINAL_PLAN_IDENTITY_MISMATCH'

REQUIRED_ORDER_SET_CHECKS = ('plan_vehicle_set', 'revealed_only', 'no_served_in_plan',
                             'no_duplicate_order')
REQUIRED_CERTIFICATION_CHECKS = ('repair_plan', 'full_partition', 'route_feasible',
                                 'energy_budget')
REQUIRED_STAGE1_COMPONENTS = REQUIRED_ORDER_SET_CHECKS + REQUIRED_CERTIFICATION_CHECKS
REQUIRED_REPORT_KEYS = ('certification_schema', 'components', 'legality_ok', 'quality',
                        'blocking_capability_gaps', 'reasons', 'ok', 'installable')
REQUIRED_QUALITY_EVIDENCE = ('impl_sha256', 'contract_version', 'applicable_domain_id',
                             'conservativeness_proof_ref', 'omega_membership_checked',
                             'read_order_ids')

# The minimal-visibility allow-list.  A quality implementation is NEVER handed the raw `env`;
# it may only ask for these keys.  Keys absent from this tuple must not be reachable at all.
ALLOWED_VISIBLE_FIELDS = ('clock', 'inst_idx', 'served_mask', 'visible_mask',
                          'visible_order_ids', 'visible_order_attrs', 'vehicle_cc_state',
                          'contract_version', 'contract_hash')
FORBIDDEN_VISIBLE_FIELDS = ('reveal_time', 'env', 'future_order_coords', 'future_order_demands',
                            'future_order_zones', 'unrevealed_order_ids', 'raw_env')


def _structural_fingerprint(plans):
    """A deterministic structural hash of a plan mapping, used for mutation detection."""
    rows = []
    for vid in sorted(int(v) for v in plans):
        p = plans[vid] if vid in plans else None
        if p is None:
            for k, v in plans.items():
                if int(k) == vid:
                    p = v
                    break
        rows.append([vid, int(getattr(p, 'anchor_node', -1)), float(getattr(p, 'anchor_time', 0.0)),
                     [int(x) for x in getattr(p, 'suffix', [])]])
    blob = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def plan_identity(plans):
    return _structural_fingerprint(plans)


def _shares_mutable_storage(a, b):
    """True when two plan mappings can mutate each other's state."""
    if a is b:
        return True
    try:
        ka, kb = set(a.keys()), set(b.keys())
    except Exception:                                             # noqa: BLE001
        return True
    for k in ka & kb:
        pa, pb = a[k], b[k]
        if pa is pb:
            return True
        sa, sb = list(getattr(pa, 'suffix', [])), list(getattr(pb, 'suffix', []))
        for x, y in zip(sa, sb):
            if x is y and not isinstance(x, int):
                return True
    return False


# ==============================================================================================
# S-12: P0 PROVENANCE AND ALIASING  (NOT value equality)
# ==============================================================================================
def check_p0_provenance(base_plans, candidate_plans, *, p0_provenance, p0_sha256,
                        registered_p0_sha256):
    """Verify the base is an INDEPENDENT FROZEN P0, not an alias of the candidate.

    Deliberately does NOT compare base CONTENT with candidate content: a legal candidate may
    equal P0 exactly (a mask repair that reconverges, or a KEEP selection).  Content equality is
    NOT evidence of cheating.
    """
    if base_plans is None:
        return S12_BASE_IS_MISSING, {}
    detail = {'p0_provenance': p0_provenance, 'p0_sha256': p0_sha256,
              'registered_p0_sha256': registered_p0_sha256,
              'content_equal_is_not_checked': True,
              'content_equal': (_structural_fingerprint(base_plans)
                                == _structural_fingerprint(candidate_plans))}
    if base_plans is candidate_plans:
        return S12_BASE_IS_CANDIDATE_ALIAS, detail
    if id(base_plans) == id(candidate_plans):
        return S12_BASE_IS_CANDIDATE_ALIAS, detail
    if _shares_mutable_storage(base_plans, candidate_plans):
        return S12_SHARED_MUTABLE_STORAGE, detail
    if p0_provenance != 'FROZEN_P0_ARTIFACT':
        return S12_BASE_NOT_FROM_FROZEN_SOURCE, detail
    if registered_p0_sha256 is not None and p0_sha256 != registered_p0_sha256:
        return S12_BASE_SHA_MISMATCH, detail
    return S12_PROVENANCE_ISOLATED, detail


# ==============================================================================================
# S-17: THE MINIMAL-VISIBILITY VIEW.  Built from an env but NEVER handing the env over.
# ==============================================================================================
class MinimalVisibilityView(object):
    """Everything a quality implementation may see, and nothing else.

    Constructed once per decision state.  `__getattr__` REFUSES any field outside the
    allow-list, including `reveal_time` and any raw env access, so a quality implementation
    cannot read decision-time-invisible order data through this object.
    """

    def __init__(self, *, clock, inst_idx, served_mask, visible_mask, visible_order_ids,
                 visible_order_attrs, vehicle_cc_state, contract_version, contract_hash):
        self._d = {
            'clock': float(clock), 'inst_idx': int(inst_idx),
            'served_mask': served_mask, 'visible_mask': visible_mask,
            'visible_order_ids': tuple(int(x) for x in visible_order_ids),
            'visible_order_attrs': dict(visible_order_attrs or {}),
            'vehicle_cc_state': vehicle_cc_state,
            'contract_version': contract_version, 'contract_hash': contract_hash,
        }

    def get(self, key):
        if key in FORBIDDEN_VISIBLE_FIELDS:
            raise PermissionError('FORBIDDEN_VISIBLE_FIELD:%s' % key)
        if key not in ALLOWED_VISIBLE_FIELDS:
            raise PermissionError('FIELD_NOT_ON_ALLOWLIST:%s' % key)
        return self._d[key]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self.get(name)

    def keys(self):
        return tuple(sorted(self._d))

    @classmethod
    def from_state(cls, env, inst_idx, clock, served_mask, vehicles, contract):
        """Build the view from an env WITHOUT exposing the env.

        `visible_order_ids` is derived from `env.reveal_time` HERE, inside the adapter, so the
        implementation never needs (and never gets) the reveal array itself.
        """
        reveal_row = env.reveal_time[inst_idx]
        visible_ids = [int(c) for c in range(1, int(env.num_nodes))
                       if bool(reveal_row[c] <= clock + 1e-6)]
        visible_mask = [bool(reveal_row[c] <= clock + 1e-6) for c in range(int(env.num_nodes))]
        return cls(clock=clock, inst_idx=inst_idx, served_mask=served_mask,
                   visible_mask=visible_mask, visible_order_ids=visible_ids,
                   visible_order_attrs={}, vehicle_cc_state=vehicles,
                   contract_version=getattr(contract, 'version', None),
                   contract_hash=getattr(contract, 'contract_hash', None))


# ==============================================================================================
# S-16: STRICT TYPE CHECKING OF THE QUALITY RESULT
# ==============================================================================================
def strict_quality_verdict(result, expected, view):
    """T1 no truthiness / T2 missing-means-refuse / T3 adapter-first.

    Returns (verdict, detail).  `verdict == QUALITY_CERTIFIED` only when `qok` is EXACTLY the
    builtin `True` AND every identity field matches AND omega membership was explicitly checked
    AND no order outside the visible set was declared as read.
    """
    # (1) structure: exactly a 2-tuple
    if not (isinstance(result, tuple) and len(result) == 2):
        return QUALITY_RESULT_NOT_A_PAIR, {'got_type': type(result).__name__}
    qok, qdetail = result
    # (2) type: exactly the builtin bool.  `type(x) is bool` rejects numpy.bool_, 1, 0.0, "no",
    #     [0], {...} -- a truthiness test (`if qok:`) or `qok == True` would accept 1.
    if type(qok) is not bool:                                     # noqa: E721  (deliberate)
        return QUALITY_QOK_NOT_BUILTIN_BOOL, {'got_type': type(qok).__name__, 'got_repr': repr(qok)[:60]}
    # (3) evidence must be a mapping containing EVERY required key (no .get default)
    if not isinstance(qdetail, dict):
        return QUALITY_DETAIL_NOT_A_MAPPING, {'got_type': type(qdetail).__name__}
    missing = [k for k in REQUIRED_QUALITY_EVIDENCE if k not in qdetail]
    if missing:
        return QUALITY_EVIDENCE_MISSING, {'missing': missing}
    # (4) identity must match the admitted entry field by field
    if qdetail['impl_sha256'] != expected['impl_sha256']:
        return QUALITY_IMPL_IDENTITY_MISMATCH, {'declared': qdetail['impl_sha256'],
                                                'registered': expected['impl_sha256']}
    if qdetail['contract_version'] != expected['contract_version']:
        return QUALITY_CONTRACT_VERSION_MISMATCH, {'declared': qdetail['contract_version'],
                                                   'registered': expected['contract_version']}
    if qdetail['applicable_domain_id'] != expected['applicable_domain_id']:
        return QUALITY_DOMAIN_IDENTITY_MISMATCH, {'declared': qdetail['applicable_domain_id'],
                                                  'registered': expected['applicable_domain_id']}
    # (5) omega membership must be EXPLICITLY declared True -- absent/None/non-bool is a refusal
    if qdetail['omega_membership_checked'] is not True:
        return QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED, {
            'value': repr(qdetail['omega_membership_checked'])}
    # (6) SECONDARY DETECTOR (NOT an isolation proof -- see the errata): a declared read of an
    #     order that is not visible at this clock is a hard failure.
    declared = qdetail['read_order_ids']
    if not isinstance(declared, (list, tuple, set, frozenset)):
        return QUALITY_EVIDENCE_MISSING, {'read_order_ids_type': type(declared).__name__}
    visible = set(int(x) for x in view.get('visible_order_ids'))
    unseen = sorted(int(x) for x in declared if int(x) not in visible)
    if unseen:
        return QUALITY_READ_UNREVEALED_ORDER, {'unrevealed_declared_read': unseen}
    # (7) only now may qok decide
    if not qok:
        return QUALITY_CHECKED_FAILED, {'reason': 'implementation returned False'}
    return QUALITY_CERTIFIED, {'evidence': dict(qdetail)}


# ==============================================================================================
# S-17: THE REGISTRY
# ==============================================================================================
class AdapterQualityRegistry(object):
    """The ONLY source of a quality implementation.  Refuses anything not fully audited."""

    def __init__(self):
        self._entries = {}

    def register(self, entry):
        self._entries[entry['impl_id']] = dict(entry)
        return entry['impl_id']

    def admit(self, impl_id):
        """Return (verdict, entry).  `verdict is None` means ADMITTED."""
        e = self._entries.get(impl_id)
        if e is None:
            return UNCERTIFIED_IMPL_REFUSED, None
        if e.get('test_only') is True or str(e.get('label', '')).upper() in (
                'TEST_ONLY', 'PROBE_ONLY', 'SYNTHETIC_ONLY'):
            return TEST_ONLY_REGISTRY_REFUSED, e
        if not e.get('conservativeness_proof_ref'):
            return UNCERTIFIED_IMPL_REFUSED, e
        if not e.get('applicable_domain_id') or not e.get('omega_check_fn'):
            return NO_OMEGA_DECLARATION_REFUSED, e
        actual = _sha256_of_file(e.get('impl_path')) if e.get('impl_path') else e.get('impl_sha256')
        if actual != e.get('impl_sha256'):
            return IMPL_SHA_MISMATCH_REFUSED, e
        return None, e


def _sha256_of_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:                                             # noqa: BLE001
        return None


# ==============================================================================================
# THE ADAPTER
# ==============================================================================================
class M1aCertificationAdapter(object):
    """Two-stage fail-closed certification.  Installs nothing; returns a receipt.

    STAGE 1  `compose_certification(..., quality_check_fn=None, project_quality=False)`
             judged by an INDEPENDENT predicate (`_stage1_passed`), never by `installable`.
    STAGE 2  reached ONLY when stage 1 passed: resolve the implementation through the registry,
             run it over the MINIMAL-VISIBILITY view, then strictly type-check its result.
    FINAL    binds the candidate plan identity to BOTH stage results.
    """

    def __init__(self, registry=None, registered_p0_sha256=None, composer=None):
        self.registry = registry
        self.registered_p0_sha256 = registered_p0_sha256
        self._composer = composer
        self.receipts = []

    # ------------------------------------------------------------------ composer indirection
    def _compose(self):
        if self._composer is not None:
            return self._composer
        from d3_composed_certifier import compose_certification
        return compose_certification

    # ------------------------------------------------------------------ STAGE 1
    def _stage1_passed(self, rep):
        """THE FIRST-STAGE PASS CONDITION.

        Must NOT consult `installable`: stage 1 always runs with `quality_check_fn=None`, so the
        composer always reports `quality.state == QUALITY_CAPABILITY_GAP` and
        `installable is False` even when every executable check passes.
        """
        if rep is None:
            return STAGE1_COMPOSER_EXCEPTION, {'why': 'composer returned None'}
        if not isinstance(rep, dict):
            return STAGE1_REPORT_SCHEMA_INVALID, {'got_type': type(rep).__name__}
        missing = [k for k in REQUIRED_REPORT_KEYS if k not in rep]
        if missing:
            return STAGE1_REPORT_SCHEMA_INVALID, {'missing_report_keys': missing}
        comps = rep.get('components')
        if not isinstance(comps, dict):
            return STAGE1_REPORT_SCHEMA_INVALID, {'components_type': type(comps).__name__}
        absent = [k for k in REQUIRED_STAGE1_COMPONENTS if k not in comps]
        if absent:
            return STAGE1_REPORT_SCHEMA_INVALID, {'missing_components': absent}
        for k in REQUIRED_STAGE1_COMPONENTS:
            v = comps[k]
            if not isinstance(v, dict) or v.get('ok') is not True:
                if k in REQUIRED_ORDER_SET_CHECKS:
                    return STAGE1_LEGALITY_GATE_FAILED, {'failed': k, 'detail': v}
                if k == 'repair_plan':
                    if v.get('error') == 'base_plans_missing':
                        return STAGE1_MISSING_BASE_PLANS, {'failed': k}
                    return STAGE1_REPAIR_PLAN_FAILED, {'failed': k, 'detail': v}
                if k == 'full_partition':
                    return STAGE1_FULL_PARTITION_FAILED, {'failed': k, 'detail': v}
                if k == 'route_feasible':
                    return STAGE1_ROUTE_INFEASIBLE, {'failed': k, 'detail': v}
                return STAGE1_ENERGY_BUDGET_EXCEEDED, {'failed': k, 'detail': v}
        # stage-1 must not have silently acquired a quality verdict
        if rep['quality'].get('state') != 'CAPABILITY_GAP_BLOCKING':
            return STAGE1_REPORT_SCHEMA_INVALID, {
                'why': 'stage 1 ran with quality_check_fn=None, so the quality state MUST be the '
                       'capability gap', 'got': rep['quality'].get('state')}
        return STAGE1_PASS, {'checks': list(REQUIRED_STAGE1_COMPONENTS)}

    def certify(self, ctx, candidate_plans, *, quality_impl_id=None):
        receipt = {'schema': 'm1a_certification_receipt_v1',
                   'candidate_plan_identity': plan_identity(candidate_plans),
                   'candidate_opaque_id': ctx.get('candidate_opaque_id'),
                   'state_id': ctx.get('state_id'),
                   # every receipt carries BOTH stage slots on EVERY return path, so a consumer
                   # never has to guess whether a missing key means "not run" or "not recorded"
                   'stage1': {'verdict': 'NOT_RUN'},
                   'stage2': {'verdict': 'NOT_RUN'},
                   'final': None,
                   'full_installability_available': False}
        # ---- S-12 provenance / aliasing (before anything else) ----
        if ctx.get('enforce_p0_provenance', True):
            v12, d12 = check_p0_provenance(
                ctx.get('P0'), candidate_plans,
                p0_provenance=ctx.get('P0_provenance'),
                p0_sha256=ctx.get('P0_sha256'),
                registered_p0_sha256=self.registered_p0_sha256)
            receipt['p0_provenance'] = {'verdict': v12, 'detail': d12}
            if v12 != S12_PROVENANCE_ISOLATED:
                receipt['final'] = FINAL_STAGE1_REFUSED
                receipt['stage1'] = {'verdict': STAGE1_P0_PROVENANCE_INVALID, 'detail': d12}
                self.receipts.append(receipt)
                return False, receipt
        else:
            receipt['p0_provenance'] = {'verdict': 'NOT_ENFORCED', 'detail': {}}
        p0_before = _structural_fingerprint(ctx['P0']) if ctx.get('P0') is not None else None

        # ---- adapter-enforced mutable scope, BEFORE the composer ----
        # The composer accepts `mutable_ids` but NEVER reads it (verified: one ast.arg, zero uses),
        # so immutability is this adapter's responsibility.  The check runs BEFORE the composer on
        # purpose: an out-of-scope candidate is invalid regardless of any other check, and running
        # the composer first would let `repair_plan` (which reconciles against P0) mask the scope
        # violation with a different reason -- the failure would be attributable to the wrong
        # cause.  Failing fast also means a scope-violating plan never reaches any check at all.
        v_mut, d_mut = self._enforce_mutable_scope(ctx, candidate_plans)
        receipt['mutable_scope'] = {'verdict': v_mut if v_mut is not True else 'ENFORCED',
                                    'detail': d_mut}
        if v_mut is not True:
            receipt['stage1'] = {'verdict': STAGE1_MUTABLE_SCOPE_VIOLATION, 'detail': d_mut}
            receipt['stage2'] = {'verdict': 'NOT_RUN_STAGE1_REFUSED'}
            receipt['final'] = FINAL_STAGE1_REFUSED
            self.receipts.append(receipt)
            return False, receipt

        # ---- STAGE 1: executable checks only, composer MUST NOT see a quality callback ----
        try:
            rep = self._compose()(ctx['env'], ctx['inst_idx'], ctx['clock'], ctx['vehicles'],
                                  ctx['served_mask'], candidate_plans,
                                  mask_set=ctx['mask_set'], protected=ctx['protected'],
                                  deferred=ctx['deferred'], B=ctx['B'], contract=ctx['contract'],
                                  base_plans=ctx['P0'], mutable_ids=ctx.get('mutable_ids'),
                                  quality_check_fn=None,           # routes around composer :291
                                  project_quality=False)           # ruling: projection is not a gate
            s1, d1 = self._stage1_passed(rep)
        except BaseException as exc:                                  # noqa: BLE001
            rep, s1 = None, STAGE1_COMPOSER_EXCEPTION
            d1 = {'exception': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
        receipt['stage1'] = {'verdict': s1, 'detail': d1}

        if s1 != STAGE1_PASS:
            receipt['final'] = FINAL_STAGE1_REFUSED
            receipt['stage2'] = {'verdict': 'NOT_RUN_STAGE1_REFUSED'}
            self.receipts.append(receipt)
            return False, receipt

        # ---- the composer must not have mutated P0 ----
        if p0_before is not None and _structural_fingerprint(ctx['P0']) != p0_before:
            receipt['final'] = FINAL_STAGE1_REFUSED
            receipt['stage1'] = {'verdict': STAGE1_P0_PROVENANCE_INVALID,
                                 'detail': {'why': S12_CANDIDATE_PATH_MUTATED_P0}}
            self.receipts.append(receipt)
            return False, receipt

        # ---- STAGE 2: independent quality certification ----
        if self.registry is None:
            receipt['stage2'] = {'verdict': REGISTRY_NOT_SUPPLIED}
            receipt['final'] = FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
            self.receipts.append(receipt)
            return False, receipt
        admit_v, entry = self.registry.admit(quality_impl_id)
        if admit_v is not None:
            receipt['stage2'] = {'verdict': admit_v, 'impl_id': quality_impl_id}
            receipt['final'] = FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
            self.receipts.append(receipt)
            return False, receipt
        # omega membership is checked HERE, at run time, on the minimal-visibility view
        view = MinimalVisibilityView.from_state(ctx['env'], ctx['inst_idx'], ctx['clock'],
                                                ctx['served_mask'], ctx['vehicles'],
                                                ctx['contract'])
        omega_v = self._check_omega(entry, ctx, view)
        if omega_v is not None:
            receipt['stage2'] = {'verdict': omega_v, 'impl_id': quality_impl_id}
            receipt['final'] = FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
            self.receipts.append(receipt)
            return False, receipt
        try:
            raw = entry['impl'](view, candidate_plans)
        except BaseException as exc:                                  # noqa: BLE001
            receipt['stage2'] = {'verdict': QUALITY_CHECKED_FAILED,
                                 'exception': '%s: %s' % (type(exc).__name__, str(exc)[:160])}
            receipt['final'] = FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
            self.receipts.append(receipt)
            return False, receipt
        qv, qd = strict_quality_verdict(raw, entry, view)
        receipt['stage2'] = {'verdict': qv, 'detail': qd, 'impl_id': quality_impl_id}

        # ---- FINAL: bind BOTH stages to the candidate plan identity ----
        if plan_identity(candidate_plans) != receipt['candidate_plan_identity']:
            receipt['final'] = FINAL_PLAN_IDENTITY_MISMATCH
        elif qv == QUALITY_CERTIFIED:
            receipt['final'] = FINAL_INSTALLABLE
        else:
            receipt['final'] = FINAL_QUALITY_REFUSED
        receipt['full_installability_available'] = receipt['final'] == FINAL_INSTALLABLE
        self.receipts.append(receipt)
        return receipt['final'] == FINAL_INSTALLABLE, receipt

    # ------------------------------------------------------------------ helpers
    def _check_omega(self, entry, ctx, view):
        try:
            ok = entry['omega_check_fn'](view, ctx.get('domain_facts') or {})
        except BaseException:                                         # noqa: BLE001
            return 'QUALITY_DOMAIN_MEMBERSHIP_FAILED'
        return None if ok is True else 'QUALITY_DOMAIN_MEMBERSHIP_FAILED'

    def _enforce_mutable_scope(self, ctx, candidate_plans):
        """The composer accepts `mutable_ids` but NEVER reads it, so this adapter enforces it.

        Compares EACH vehicle against the independent frozen P0, not against the candidate:
          * a vehicle listed in `mutable_ids` may differ freely;
          * any OTHER vehicle must be identical to P0 -- a vehicle is out of scope if it is
            absent from P0, or if its suffix, anchor node, anchor time, anchor load or vehicle id
            differs.  anchor_time is checked because it is part of physical state (a changed
            anchor time changes what an existing commitment means) even though it is not part of
            the order sequence.
        """
        mutable = ctx.get('mutable_ids')
        if mutable is None:
            return True, {'enforced': False, 'why': 'no mutable_ids supplied'}
        mut = set(int(x) for x in mutable)
        p0 = ctx.get('P0') or {}
        p0_by_vid = {int(k): v for k, v in p0.items()}
        candidates = {int(k): v for k, v in candidate_plans.items()}
        violations = []
        for vid, cand in sorted(candidates.items()):
            if vid in mut:
                continue
            base = p0_by_vid.get(vid)
            if base is None:
                violations.append({'vehicle': vid, 'why': 'immutable vehicle absent from P0'})
                continue
            if list(getattr(base, 'suffix', [])) != list(getattr(cand, 'suffix', [])):
                violations.append({'vehicle': vid, 'why': 'immutable vehicle route changed',
                                   'before': list(getattr(base, 'suffix', [])),
                                   'after': list(getattr(cand, 'suffix', []))})
            if int(getattr(base, 'anchor_node', -1)) != int(getattr(cand, 'anchor_node', -1)):
                violations.append({'vehicle': vid, 'why': 'immutable vehicle anchor node changed',
                                   'before': int(getattr(base, 'anchor_node', -1)),
                                   'after': int(getattr(cand, 'anchor_node', -1))})
            if abs(float(getattr(base, 'anchor_time', 0.0))
                   - float(getattr(cand, 'anchor_time', 0.0))) > 1e-9:
                violations.append({'vehicle': vid, 'why': 'immutable vehicle anchor time changed',
                                   'before': float(getattr(base, 'anchor_time', 0.0)),
                                   'after': float(getattr(cand, 'anchor_time', 0.0))})
            if abs(float(getattr(base, 'anchor_load', 0.0))
                   - float(getattr(cand, 'anchor_load', 0.0))) > 1e-9:
                violations.append({'vehicle': vid, 'why': 'immutable vehicle anchor load changed'})
        return ((True, {'enforced': True, 'n_mutable': len(mut), 'n_immutable': len(candidates) - len(
            mut & set(candidates))}) if not violations
            else (False, {'violations': violations}))


def build_context(env, inst_idx, clock, vehicles, served_mask, P0, *, B, contract, mask_set,
                  protected=frozenset(), deferred=frozenset(), mutable_ids=None,
                  P0_provenance='FROZEN_P0_ARTIFACT', P0_sha256=None, state_id=None,
                  candidate_opaque_id=None, domain_facts=None, enforce_p0_provenance=True):
    """Assemble the immutable CertificationContext.  P0 is fingerprinted, never copied by alias."""
    return {'env': env, 'inst_idx': int(inst_idx), 'clock': float(clock),
            'vehicles': vehicles, 'served_mask': served_mask,
            'P0': P0, 'P0_provenance': P0_provenance,
            'P0_sha256': P0_sha256 if P0_sha256 is not None else _structural_fingerprint(P0),
            'B': B, 'contract': contract, 'mask_set': tuple(mask_set),
            'protected': frozenset(protected), 'deferred': frozenset(deferred),
            'mutable_ids': None if mutable_ids is None else tuple(int(x) for x in mutable_ids),
            'state_id': state_id, 'candidate_opaque_id': candidate_opaque_id,
            'domain_facts': domain_facts or {},
            'enforce_p0_provenance': enforce_p0_provenance}
