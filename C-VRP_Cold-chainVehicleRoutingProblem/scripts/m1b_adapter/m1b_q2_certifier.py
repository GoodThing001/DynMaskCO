#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-TrustFix: the ENGINEERING CERTIFICATION ORCHESTRATOR (version M1B_Q2TRUSTFIX).

Changes relative to the Q2-Fix orchestrator
  * F-T1: the PRODUCTION class NO LONGER ACCEPTS a continuous-temperature proof hook at all.  The
    Q2-Fix-T audit (T1-1) showed that an arbitrary callable returning `(True, {})` produced
    `THERMAL_CONTINUOUS_OK` and `composite_pass = True`.  Until a real analytic proof exists and
    has been independently audited, the continuous level is ALWAYS
    `THERMAL_CONTINUOUS_NOT_ESTABLISHED`, and a test double can only be supplied through the
    SEPARATE test-only class, whose receipts are explicitly not production certificates.
  * F-T4: each vehicle is classified for thermal APPLICABILITY.  A vehicle that was never
    dispatched, is idle at the depot with no cargo, no commitment and no candidate work is
    `THERMAL_NOT_APPLICABLE` and neither proves nor sinks the fleet.  A vehicle holding cargo,
    a commitment or planned work WITHOUT a cold-chain state still REFUSES.
  * the trusted strict-online projection (F-T3) is produced by `m1b_public_projection`; this
    orchestrator only consumes it, and refuses inputs that are not a projection.

RECEIPT FIELDS (independent, non-substitutable)
    QUALITY_LB_OK            analytic quality lower bound above threshold
    THERMAL_SAMPLED_OK       sampled envelope inside the hard bounds
    THERMAL_CONTINUOUS_OK    continuous-time extremum PROVEN (ALWAYS False this round)
    THERMAL_CONTINUOUS_STATE THERMAL_CONTINUOUS_OK / _NOT_ESTABLISHED / _VIOLATED
    thermal_applicability    per-vehicle REQUIRED / NOT_APPLICABLE
    test_only                True only for the test-only class; such a receipt is NOT consumable

WHAT THIS IS NOT: engineering acceptance only.  NOT production installability, NOT scientific
evidence, NOT L2, and it does NOT establish FULL_INSTALLABILITY_ESTABLISHED.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import m1b_contract_identity as CI
import m1b_immutable_snapshot as IS
import m1b_public_projection as PP
import m1b_quality_certificate as QC
import m1b_return_feasibility as RF
import m1b_thermal_gate as TG
import m1b_timing as TM

SCHEMA = 'm1b_q2trustfix_engineering_receipt_v1'
IMPLEMENTATION_VERSION = 'M1B_Q2TRUSTFIX'

ENGINEERING_CERTIFIED = 'M1B_Q2TRUSTFIX_ENGINEERING_CERTIFIED'
ENGINEERING_REFUSED = 'M1B_Q2TRUSTFIX_ENGINEERING_REFUSED'
NOT_ESTABLISHED = 'M1B_Q2TRUSTFIX_NOT_ESTABLISHED'

PRODUCTION_PROOF_HOOK_REFUSED = 'PRODUCTION_CERTIFIER_REFUSES_PROOF_HOOK'

COMPOSITE_DISCLAIMER = (
    'engineering acceptance only: NOT production installability, NOT scientific evidence, NOT '
    'L2, and it does NOT establish FULL_INSTALLABILITY_ESTABLISHED')

TEST_ONLY_LABEL = 'TEST_ONLY_NOT_A_PRODUCTION_CERTIFICATE'

#: The sha256 of the PRE-FIX modules is recorded in the fix spec and the acceptance report
#: (`M1b_Q2TrustFix_修复规格_2026-10-16.md` §0 and `M1b_Q2TrustFix_验收报告_2026-10-16.md` §0),
#: NOT duplicated here.  A literal table of 64-hex digests inside a source file trips the
#: repository's staged secret scanner (rule R2_bare_64_hex_token) on a FALSE POSITIVE -- those
#: values are content hashes, not credentials.  Rather than exempt the scanner, the literals are
#: kept out of the source and the report remains the single recorded copy.
PRE_FIX_SOURCE_SHA256_DOC = ('see M1b_Q2TrustFix_修复规格_2026-10-16.md §0 and '
                             'M1b_Q2TrustFix_验收报告_2026-10-16.md §0')

_ADAPTER_DIR = Path(__file__).resolve().parent


def implementation_identity():
    out = {}
    for p in sorted(_ADAPTER_DIR.glob('m1b_*.py')):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class _Receipt(dict):
    """Receipt dict carrying ONE read-only compatibility property (see Q2-Fix disclosure)."""

    @property
    def THERMAL_HARD_OK(self):                                           # noqa: N802
        return self.get('THERMAL_SAMPLED_OK')

    def __missing__(self, key):
        if key == 'THERMAL_HARD_OK':
            return self.get('THERMAL_SAMPLED_OK')
        raise KeyError(key)


class M1bQ2TrustFixCertifier(object):
    """PRODUCTION-shaped engineering certifier.  NO continuous-proof hook exists here (F-T1)."""

    def __init__(self, *, contract, registered_contract_hash=None, T_end, t_min=0.0,
                 epsilon_init=QC.EPSILON_INIT, epsilon_return=RF.EPSILON_RETURN,
                 budget=None, budget_is_synthetic=True, closed_return_history=None,
                 continuous_extremum_proof=None, verified_t_min=None, env=None):
        # B (final-entry fix) + G-TMIN.  `t_min` must be an INDEPENDENTLY VERIFIABLE execution-time
        # lower bound.  The audited version accepted any float, and `t_min = 1e6` collapsed H_max to 0
        # so Q_LB degenerated to the initial value.  No trusted witness mechanism exists, so NO value
        # of `verified_t_min` changes a number and the bare `t_min` keyword stays IGNORED (retained
        # for signature compatibility only).
        #
        # The caller's object is NOT RETAINED.  Only the boolean fact "a claim was supplied" survives:
        # keeping the object would let a later state dump, a debugging export or an incidental
        # repr()/str() reach caller-controlled code AFTER the refusal decision, which is exactly the
        # hazard `_resolve_t_min`'s contract excludes.
        self.env = env
        self.t_min_requested = t_min
        self.t_min_claim_supplied = verified_t_min is not None
        self.t_min = self._resolve_t_min()
        # F-T1: a proof hook is NOT a supported input.  Rather than raising from the constructor --
        # which would make historical test files crash at construction and hide their real
        # divergences -- the value is REMEMBERED and the certification is REFUSED when it is not
        # None.  So no call pattern can reach THERMAL_CONTINUOUS_OK through this class.
        self._refused_proof_hook = continuous_extremum_proof
        self.contract = contract
        self.registered_contract_hash = registered_contract_hash
        self.T_end = T_end
        self.epsilon_init = epsilon_init
        self.epsilon_return = epsilon_return
        self.budget = budget
        self.budget_is_synthetic = budget_is_synthetic
        self.closed_return_history = dict(closed_return_history or {})
        self.receipts = []
        self.test_only = False

    def _resolve_t_min(self):
        """Return the t_min actually used: 0.0, unconditionally (G-TMIN fix, 2026-10-17).

        No trusted-witness mechanism for a positive execution-time lower bound exists, so NO value of
        `verified_t_min` may change a number.  A supplied claim is REMEMBERED as a BOOLEAN only, so
        that `certify()` can REFUSE the receipt: a conservative number is not a passing judgement.
        The supplied object itself is never stored, never formatted and never logged, because calling
        `repr()`/`str()` on caller-controlled data would execute caller code inside the certifier and
        could leak the value into a receipt.  `env` is not a witness and is not consulted here; it
        remains in use in `certify()` for the trusted-projection check, which this fix does not touch.
        The pre-fix label claimed a verification the code never performed; it is replaced by
        `UNVERIFIED_T_MIN_REFUSED`, which is also a refusal trigger.
        """
        self.t_min_source = 'CONSERVATIVE_CLOCK_ORIGIN'
        self.t_min_refused = self.t_min_claim_supplied
        if self.t_min_refused:
            self.t_min_source = 'UNVERIFIED_T_MIN_REFUSED'
        return 0.0

    # ------------------------------------------------------------------ main entry
    def certify(self, *, vehicles, served_mask, candidate_plans, clock, inst_idx,
                travel_fn, distance_fn, projection=None,
                per_class_constants=None, on_board_lots=None, committed_fish_nodes=None,
                candidate_fish_nodes=None, active_zone_mask=None, certified_order_ids=None,
                certify_route_fn=None,
                # ---- LEGACY (pre-fix) keyword arguments, translated, never trusted verbatim ----
                times=None, order_attrs=None, temp_class_of=None, quantity_of=None,
                service_time_of=None, initial_quality_by_node=None, lot_provider=None,
                env=None, **_ignored):
        if projection is None and times is not None and order_attrs is not None:
            # A caller may present the two provider halves directly; they are accepted ONLY if
            # they are the trusted projection type (F-T3), otherwise refused below.
            projection = times if isinstance(times, PP.TrustedPublicProjection) else None
            if projection is None and isinstance(order_attrs, PP.TrustedPublicProjection):
                projection = order_attrs
        if projection is None and (temp_class_of is not None or service_time_of is not None
                                   or initial_quality_by_node is not None):
            projection = _legacy_projection(temp_class_of, quantity_of, service_time_of,
                                            initial_quality_by_node, candidate_plans,
                                            contract_hash=getattr(self.contract, 'contract_hash',
                                                                  None))

        receipt = _Receipt({
            'schema': SCHEMA, 'implementation_version': IMPLEMENTATION_VERSION,
            'implementation_sha256': implementation_identity(),
            'pre_fix_source_sha256': PRE_FIX_SOURCE_SHA256_DOC,
            'test_only': bool(self.test_only),
            'inst_idx': int(inst_idx), 'clock': float(clock),
            'epsilon_init': float(self.epsilon_init), 'epsilon_return': float(self.epsilon_return),
            'budget_is_synthetic': bool(self.budget_is_synthetic),
            'QUALITY_LB_OK': None, 'THERMAL_SAMPLED_OK': None, 'THERMAL_CONTINUOUS_OK': None,
            'THERMAL_CONTINUOUS_STATE': None, 'thermal_applicability': None,
            'composite_pass': False, 'final': NOT_ESTABLISHED,
            'disclaimer': COMPOSITE_DISCLAIMER,
            'not_consumable_as_production_certificate': bool(self.test_only),
            'gates': {},
        })
        if self.test_only:
            receipt['TEST_ONLY_LABEL'] = TEST_ONLY_LABEL

        # ---- 0 F-T1: a proof hook on the PRODUCTION path is an immediate refusal ------------
        if self._refused_proof_hook is not None:
            receipt['gates']['proof_hook'] = {
                'verdict': PRODUCTION_PROOF_HOOK_REFUSED,
                'detail': {'why': 'the production certifier does not accept a continuous-extremum '
                                  'proof callable; until a real analytic proof exists and has been '
                                  'independently audited, THERMAL_CONTINUOUS_STATE stays '
                                  'THERMAL_CONTINUOUS_NOT_ESTABLISHED',
                           'use_instead': 'M1bQ2TrustFixTestOnlyCertifier for wiring tests'}}
            return self._refuse(receipt)

        # ---- 0b G-TMIN: an untrusted t_min claim REFUSES the certification --------------------
        # Authorised minimal fix (2026-10-17).  A conservative number is not a passing judgement:
        # a caller that supplied a `verified_t_min` must not read `composite_pass = True` as "the
        # certification is fine".  `t_min` itself is already 0.0 (see `_resolve_t_min`), so the
        # quality bound is the most conservative one; what changes here is only the verdict.
        #
        # The detail carries a FIXED reason code and booleans only.  The supplied value is never
        # rendered: `repr()` on caller-controlled data would run caller code inside the certifier
        # (side effects, exceptions, sensitive strings) and could leak it into a receipt.
        if getattr(self, 't_min_refused', False):
            receipt['gates']['t_min'] = {
                'verdict': 'UNVERIFIED_T_MIN_REFUSED',
                'detail': {'claim_supplied': True,
                           't_min_used': 0.0,
                           'reason_code': 'UNVERIFIED_T_MIN_WITHOUT_TRUSTED_WITNESS'}}
            return self._refuse(receipt)

        # ---- 1 contract identity -------------------------------------------------
        ident, cverd, cdet = CI.validate_contract_for_quality_certificate(
            self.contract, registered_contract_hash=self.registered_contract_hash)
        receipt['contract_identity'] = ident
        receipt['gates']['contract_identity'] = {'verdict': cverd, 'detail': cdet}
        if cverd != CI.CONTRACT_IDENTITY_OK:
            return self._refuse(receipt)
        per_class, pcverd, pcdet = CI.check_quality_classes_are_certifiable(
            self.contract, epsilon_init=self.epsilon_init)
        receipt['gates']['quality_class_scope'] = {'verdict': pcverd, 'detail': pcdet}
        if pcverd != CI.CONTRACT_IDENTITY_OK:
            return self._refuse(receipt)
        if per_class_constants is not None:
            # B (final-entry fix).  The quality parameters MUST come from the identity-certified
            # contract.  The previous version used a caller-supplied mapping VERBATIM
            # (`if per_class_constants is None: per_class_constants = per_class`), and the audit
            # showed `{q0_safe: 1e6, rate_per_hour: 0.0}` inflated the certified lower bound from
            # 8.548 to 1e6.  An override is now REFUSED rather than honoured.
            receipt['gates']['per_class_constants_override'] = {
                'verdict': 'PER_CLASS_CONSTANTS_OVERRIDE_REFUSED',
                'detail': {'why': 'quality parameters are taken from the contract that passed '
                                  'identity certification; an external override is not accepted'}}
            return self._refuse(receipt)
        per_class_constants = per_class
        receipt['gates']['quality_parameters'] = {
            'verdict': 'QUALITY_PARAMETERS_FROM_CERTIFIED_CONTRACT',
            'detail': {'source': 'CL.validate_contract_for_quality_certificate -> per_class',
                       'contract_hash': getattr(self.contract, 'contract_hash', None),
                       'n_classes': len(per_class)}}

        # ---- 2 the TRUSTED projection is mandatory (F-T3) ------------------------
        if not isinstance(projection, PP.TrustedPublicProjection):
            receipt['gates']['projection'] = {
                'verdict': PP.PROJECTION_DECLARED_SET_REJECTED,
                'detail': {'why': 'certification requires a TrustedPublicProjection built by the '
                                  'trusted construction boundary; externally declared visible '
                                  'sets and free-text source identities do not qualify'}}
            return self._refuse(receipt)
        # ---- F-A: PROVENANCE.  A directly-built or diagnostic projection is REFUSED. ---------
        prov_ok, prov_v, prov_d = PP.is_certifiable(projection)
        if not prov_ok:
            receipt['gates']['projection'] = {'verdict': prov_v, 'detail': prov_d}
            return self._refuse(receipt)
        # ---- F-A: the visible set is re-verified against the ENVIRONMENT, not self-declared. --
        env_ok, env_v, env_d = PP.assert_visible_set_matches_env(
            projection, env if env is not None else self.env, inst_idx=inst_idx, clock=clock)
        if not env_ok:
            receipt['gates']['projection'] = {'verdict': env_v, 'detail': env_d}
            return self._refuse(receipt)
        proj_check = {'declared_served': getattr(projection, 'served_nodes', None),
                      'declared_revealed': getattr(projection, 'revealed_nodes', None)}
        okp, vp, dp = PP.reject_declared_set(
            declared_served=proj_check['declared_served'],
            declared_revealed=proj_check['declared_revealed'],
            required=projection.revealed_nodes)
        receipt['gates']['projection'] = {
            'verdict': PP.PROJECTION_OK if okp else vp,
            'detail': {'identity': projection.identity(), 'reconciliation': dp,
                       'visible_set_reverified_against_env': env_d,
                       'provenance': 'factory-bound',
                       'instance_id': projection.instance_id,
                       'event_id': projection.event_id, 'clock': projection.clock,
                       'n_served_nodes': len(projection.revealed_nodes)}}
        if not okp:
            return self._refuse(receipt)
        receipt['projection_identity_before'] = projection.source_identity
        receipt['projection_challenge_before'] = projection.challenge()

        # ---- 3 immutable snapshot + aliasing -------------------------------------
        iq_map = {}
        for n in sorted(projection.revealed_nodes):
            try:
                iq_map[int(n)] = projection.initial_quality_of(n)
            except (PermissionError, KeyError):
                continue
        snap, sverd, sdet = IS.build_frozen_snapshot(
            vehicles, served_mask, candidate_plans, initial_quality_by_node=iq_map)
        receipt['gates']['snapshot'] = {'verdict': sverd, 'detail': sdet}
        if sverd != IS.SNAPSHOT_OK:
            return self._refuse(receipt)
        ali_ok, ali_det = IS.detect_aliasing(snap, vehicles, candidate_plans)
        receipt['gates']['aliasing'] = {
            'verdict': 'NO_ALIAS' if ali_ok else IS.SNAPSHOT_ALIAS_DETECTED, 'detail': ali_det}
        if not ali_ok:
            return self._refuse(receipt)
        receipt['candidate_plan_identity'] = snap['plan_identity']
        receipt['state_identity_before'] = snap['state_identity']

        # the projection must be bound to the snapshot actually being certified (no `None` skip)
        if projection.snapshot_identity != snap['state_identity']:
            receipt['gates']['projection_binding'] = {
                'verdict': 'PROJECTION_SNAPSHOT_MISMATCH',
                'detail': {'projection': projection.snapshot_identity,
                           'snapshot': snap['state_identity'],
                           'note': 'an unbound projection is already refused at construction, so '
                                   'this comparison is unconditional'}}
            return self._refuse(receipt)
        receipt['gates']['projection_binding'] = {'verdict': 'PROJECTION_BOUND_TO_SNAPSHOT',
                                                  'detail': {}}

        # ---- 4 Omega11 return feasibility ---------------------------------------
        tfn = (lambda a, b: float(travel_fn(int(a), int(b))))
        ok11, v11, d11 = RF.certify_return_feasibility(
            snap, T_end=self.T_end, clock=clock, travel_fn=tfn, times=projection,
            closed_return_history=self.closed_return_history,
            certify_route_fn=certify_route_fn, epsilon_return=self.epsilon_return)
        receipt['gates']['omega11_return_feasibility'] = {'verdict': v11, 'detail': d11}
        if not ok11:
            return self._refuse(receipt)

        # ---- 5 Omega10 COVERAGE --------------------------------------------------
        if candidate_fish_nodes is None:
            candidate_fish_nodes = _candidate_fish_nodes(snap)
        if on_board_lots is None:
            on_board_lots = _on_board_lots(snap)
        if lot_provider is not None:
            try:
                provided = list(lot_provider(snap))
            except Exception as exc:                                    # noqa: BLE001
                provided = []
                receipt['gates']['legacy_lot_provider'] = {
                    'verdict': 'LEGACY_PROVIDER_RAISED',
                    'detail': {'exception': type(exc).__name__}}
            if provided:
                have = {int(getattr(x, 'order_id', -1)) for x in on_board_lots}
                for x in provided:
                    if int(getattr(x, 'order_id', -1)) not in have:
                        on_board_lots = list(on_board_lots) + [x]
        if committed_fish_nodes is None:
            committed_fish_nodes = _committed_fish_nodes(snap)
        revealed_nodes = projection.revealed_nodes
        ok10, v10, d10 = QC.certify_omega10_coverage(
            on_board_lots=on_board_lots, committed_fish_nodes=committed_fish_nodes,
            candidate_fish_nodes=candidate_fish_nodes, revealed_nodes=revealed_nodes,
            provider=projection, per_class_constants=per_class_constants,
            epsilon_init=self.epsilon_init, verified_order_ids=certified_order_ids)
        receipt['gates']['omega10_coverage'] = {
            'verdict': v10, 'detail': d10,
            'required_set_E': sorted(int(x) for x in (d10.get('required_provenance') or [])),
            'n_required': d10.get('n_required'), 'n_covered': d10.get('n_covered')}
        receipt['gates']['omega10_initialization'] = {
            'verdict': v10, 'detail': d10,
            'COMPATIBILITY_MIRROR': 'pre-fix key name; authoritative gate is omega10_coverage'}
        if not ok10:
            receipt['gates']['quality_impl_call'] = {
                'verdict': 'NOT_CALLED_OMEGA10_REFUSED',
                'detail': {'why': 'the coverage gate precedes the quality stage'}}
            return self._refuse(receipt)

        # ---- 6 thermal: applicability per vehicle, then the sampled replay -------
        thermal_rows, sampled_flags, cont_states, applicability = {}, [], [], {}
        for vid in sorted(snap['vehicles']):
            fv = snap['vehicles'][vid]
            applies, averd, adet = TG.classify_thermal_applicability(
                fv, clock=clock, closed_return_history=self.closed_return_history)
            applicability[vid] = averd
            if applies is False:
                thermal_rows[vid] = {'verdict': TG.THERMAL_NOT_APPLICABLE, 'detail': adet,
                                     'counts_as_pass': False, 'counts_as_failure': False}
                continue
            if applies is None:
                thermal_rows[vid] = {'verdict': averd, 'detail': adet,
                                     'counts_as_pass': False, 'counts_as_failure': True}
                sampled_flags.append(averd)
                continue
            stops_override = None
            start_node_override = None
            depart_override = None
            fp = snap['candidate_plans'].get(vid)
            if fv.status == 'committed' and fv.committed_finish is not None:
                start_node_override = int(fv.committed_next)
                depart_override = float(fv.committed_finish)
                stops_override = [int(x) for x in (fp.suffix if fp is not None
                                                   else fv.mutable_suffix) if int(x) != 0]
            env, tverd, tdet = TG.replay_thermal_envelope(
                fv, contract=self.contract, inst_idx=inst_idx, clock=clock,
                travel_fn=tfn, distance_fn=(lambda a, b: float(distance_fn(int(a), int(b)))),
                temp_class_of=projection.temp_class_of, quantity_of=projection.quantity_of,
                service_time_of=projection, active_zone_mask=active_zone_mask,
                stops_override=stops_override, start_node_override=start_node_override,
                depart_override=depart_override,
                initial_quality_of=projection.initial_quality_of,
                continuous_extremum_proof=None)          # F-T1: no hook on this path, ever
            thermal_rows[vid] = {'verdict': tverd, 'detail': tdet,
                                 'counts_as_pass': tverd == TG.THERMAL_SAMPLED_OK,
                                 'counts_as_failure': tverd != TG.THERMAL_SAMPLED_OK}
            sampled_flags.append(tverd)
            cont_states.append(tdet.get('continuous_level') if isinstance(tdet, dict) else None)
        required = [v for v in thermal_rows.values() if v.get('counts_as_failure') is not None
                    and (v.get('counts_as_pass') or v.get('counts_as_failure'))]
        na = [vid for vid, v in thermal_rows.items()
              if v['verdict'] == TG.THERMAL_NOT_APPLICABLE]
        thermal_sampled_ok = all(r.get('counts_as_pass') is True for r in required) if required \
            else True
        cont_state = (TG.THERMAL_CONTINUOUS_OK if cont_states
                      and all(c == TG.THERMAL_CONTINUOUS_OK for c in cont_states)
                      else TG.THERMAL_CONTINUOUS_NOT_ESTABLISHED)
        receipt['gates']['thermal_replay'] = {
            'verdict': TG.THERMAL_SAMPLED_OK if thermal_sampled_ok
            else _first_bad([r['verdict'] for r in required]),
            'detail': {'per_vehicle': thermal_rows, 'not_applicable_vehicles': na},
            'sampled_level': TG.THERMAL_SAMPLED_OK if thermal_sampled_ok
            else TG.THERMAL_BOUNDS_VIOLATED,
            'continuous_level': cont_state,
            'n_required': len(required), 'n_not_applicable': len(na)}
        receipt['THERMAL_SAMPLED_OK'] = bool(thermal_sampled_ok)
        receipt['THERMAL_CONTINUOUS_OK'] = bool(cont_state == TG.THERMAL_CONTINUOUS_OK)
        receipt['THERMAL_CONTINUOUS_STATE'] = cont_state
        receipt['thermal_applicability'] = applicability

        # ---- 7 analytic quality certificate -------------------------------------
        units = getattr(self.contract, 'units', None)
        hpu = getattr(units, 'hours_per_time_unit', None)
        qres, qverd, qdet = QC.certify_quality_lb(
            per_class_constants, T_end=self.T_end, t_min=self.t_min, hours_per_time_unit=hpu,
            epsilon_init=self.epsilon_init)
        receipt['gates']['quality_lb'] = {'verdict': qverd, 'detail': qdet}
        receipt['QUALITY_LB_OK'] = bool(qverd == QC.QUALITY_LB_OK)

        # ---- 8 post-state + projection identity consistency ----------------------
        unch, uverd, udet = IS.verify_state_unchanged(snap, vehicles, served_mask,
                                                      candidate_plans)
        receipt['gates']['state_unchanged'] = {'verdict': uverd, 'detail': udet}
        receipt['state_identity_after'] = (udet or {}).get('after', snap['state_identity'])
        receipt['projection_identity_after'] = projection.source_identity
        same_proj = (receipt['projection_identity_before'] == receipt['projection_identity_after'])
        # a trusted FIELD must not be replaceable after construction: compare the content digest
        chal_after = projection.challenge()
        receipt['projection_challenge_after'] = chal_after
        tampered = (chal_after != receipt['projection_challenge_before'])
        receipt['gates']['projection_identity_stable'] = {
            'verdict': ('PROJECTION_IDENTITY_STABLE' if (same_proj and not tampered)
                        else 'PROJECTION_CHALLENGE_MISMATCH'), 'detail': {'tampered': tampered}}
        if not same_proj or tampered:
            return self._refuse(receipt)

        # ---- 9 composite ---------------------------------------------------------
        q_ok = receipt['QUALITY_LB_OK'] is True
        s_ok = receipt['THERMAL_SAMPLED_OK'] is True
        c_ok = receipt['THERMAL_CONTINUOUS_OK'] is True
        all_gates = (cverd == CI.CONTRACT_IDENTITY_OK and pcverd == CI.CONTRACT_IDENTITY_OK
                     and sverd == IS.SNAPSHOT_OK and ali_ok and ok11 and ok10 and unch and same_proj)
        receipt['composite_pass'] = bool(all_gates and q_ok and s_ok and c_ok)
        receipt['final'] = (ENGINEERING_CERTIFIED if receipt['composite_pass']
                            else ENGINEERING_REFUSED)
        receipt['non_substitutable'] = {
            'QUALITY_LB_OK': receipt['QUALITY_LB_OK'],
            'THERMAL_SAMPLED_OK': receipt['THERMAL_SAMPLED_OK'],
            'THERMAL_CONTINUOUS_OK': receipt['THERMAL_CONTINUOUS_OK'],
            'rule': 'each gate is independently required; NEITHER the sampled level NOR any '
                    'injected proof may substitute for the continuous level',
            'continuous_not_established_effect':
                'composite_pass is FAIL-CLOSED whenever THERMAL_CONTINUOUS_STATE is '
                'THERMAL_CONTINUOUS_NOT_ESTABLISHED',
            'not_applicable_vehicles': na}
        self.receipts.append(receipt)
        return receipt['composite_pass'], receipt

    def _refuse(self, receipt):
        receipt['final'] = ENGINEERING_REFUSED
        self.receipts.append(receipt)
        return False, receipt


class M1bQ2TrustFixTestOnlyCertifier(M1bQ2TrustFixCertifier):
    """ISOLATED TEST ENTRY (F-T1 requirement 4).

    The ONLY way to supply a continuous-extremum test double.  Every receipt from this class is
    stamped `test_only = True`, carries `TEST_ONLY_LABEL` and
    `not_consumable_as_production_certificate = True`, so a wiring test can exercise the
    TRUE branch of the continuous level without producing anything a production consumer may
    treat as a certificate.
    """

    def __init__(self, *, test_continuous_extremum_proof=None, **kw):
        kw.pop('continuous_extremum_proof', None)
        super(M1bQ2TrustFixTestOnlyCertifier, self).__init__(**kw)
        self.test_only = True
        self._test_proof = test_continuous_extremum_proof

    def certify(self, **kw):
        ok, rc = super(M1bQ2TrustFixTestOnlyCertifier, self).certify(**kw)
        rc['test_only'] = True
        rc['TEST_ONLY_LABEL'] = TEST_ONLY_LABEL
        rc['not_consumable_as_production_certificate'] = True
        if self._test_proof is not None:
            try:
                proven, _d = self._test_proof()
            except Exception:                                            # noqa: BLE001
                proven = False
            if proven is True:
                # record the wiring outcome WITHOUT storing the ambiguous continuous field as a
                # production value: the receipt stays test-only and non-consumable
                rc['test_only_continuous_wiring'] = 'TEST_DOUBLE_REACHED_TRUE_BRANCH'
            else:
                rc['test_only_continuous_wiring'] = 'TEST_DOUBLE_FALSE'
        return ok, rc


def _first_bad(verdicts):
    for v in verdicts:
        if v != TG.THERMAL_SAMPLED_OK:
            return v
    return TG.THERMAL_REPLAY_INCOMPLETE


def _on_board_lots(snap):
    seen, out = set(), []
    for vid in sorted(snap['vehicles']):
        fv = snap['vehicles'][vid]
        for lot in (getattr(fv.coldchain_state, 'cargo_manifest', ()) or ()):
            if int(lot.order_id) in seen:
                continue
            seen.add(int(lot.order_id))
            out.append(lot)
    return out


def _candidate_fish_nodes(snap):
    out = []
    for vid in sorted(snap['candidate_plans']):
        for n in snap['candidate_plans'][vid].suffix:
            n = int(n)
            if n > 0:
                out.append(n)
    return out


def _committed_fish_nodes(snap):
    out = []
    for vid in sorted(snap['vehicles']):
        fv = snap['vehicles'][vid]
        if fv.status == 'committed' and fv.committed_next:
            out.append(int(fv.committed_next))
    return out


def _legacy_projection(temp_class_of, quantity_of, service_time_of, initial_quality_by_node,
                       candidate_plans, contract_hash=None):
    """Translate the PRE-FIX loose callables into a TRUSTED PROJECTION (F-T3).

    The translation does NOT grant trust: nodes come only from the caller-named tables and the
    candidate suffix, every copied value is whatever the callable returns, and the identity is
    computed from the bound facts.  A node whose value is unavailable defaults to 1.0 so the
    COVERAGE gate REFUSES it rather than silently passing.
    """
    nodes = set(int(k) for k in (initial_quality_by_node or {}))
    for p in (candidate_plans or {}).values():
        nodes.update(int(x) for x in getattr(p, 'suffix', []) if int(x) > 0)
    nodes = sorted(nodes)
    iq, tc, qty, svc, tws, twe = {}, {}, {}, {}, {}, {}
    for n in nodes:
        try:
            iq[n] = float(initial_quality_by_node.get(n, 1.0)) if initial_quality_by_node else 1.0
        except Exception:                                                # noqa: BLE001
            iq[n] = 1.0
        try:
            tc[n] = int(temp_class_of(n)) if temp_class_of is not None else 1
        except Exception:                                                # noqa: BLE001
            tc[n] = 1
        try:
            qty[n] = float(quantity_of(n)) if quantity_of is not None else 1.0
        except Exception:                                                # noqa: BLE001
            qty[n] = 1.0
        try:
            svc[n] = float(service_time_of(n)) if service_time_of is not None else 0.0
        except Exception:                                                # noqa: BLE001
            svc[n] = 0.0
        tws[n] = 0.0
        twe[n] = 1e9
    return PP.projection_from_revealed_maps(iq=iq, tc=tc, qty=qty, svc=svc, tws=tws, twe=twe,
                                            revealed_nodes=nodes, instance_id=-1, event_id=-1,
                                            clock=0.0, contract_hash=contract_hash,
                                            snapshot_identity=None)
    # NOTE: `projection_from_revealed_maps` returns a DIAGNOSTIC projection (data_trusted=False,
    # foreign token, no real binding), so `PP.is_certifiable` REFUSES it.  The legacy callable path
    # therefore cannot certify anything -- an un-revealed node or a real late time window smuggled
    # through legacy callables is stopped at the provenance gate rather than by losing tw_start.


# ---- pre-fix symbol names, kept ONLY so historical test files remain importable -------------
M1bQ2FixCertifier = M1bQ2TrustFixCertifier
M1bQ2Certifier = M1bQ2TrustFixCertifier
ENGINEERING_CERTIFIED_PRE_FIX = 'M1B_Q2FIX_ENGINEERING_CERTIFIED'
ENGINEERING_REFUSED_PRE_FIX = 'M1B_Q2FIX_ENGINEERING_REFUSED'
